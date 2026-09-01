import os
import pandas as pd
import traceback
import sys


from typing import Dict
from datetime import datetime, timedelta
import multiprocessing as mp
from collections import defaultdict

from ptinfra import get_logger,intialize
from ptinfra.config import get_variable
from pymongo import MongoClient
from qdrant_client import QdrantClient

from src.request_processing import read_messages

from src.pipeline import AlbumContext, build_select

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from PIL import Image
import io
from utils.configs import CONFIGS

from ptinfra.pt_queue import Message
from main import ProcessStage

request_name = 'request0'
album_name = 'album1'


def _group_placements_by_composition(placements_img):
    """Bucket every placement under its compositionId."""
    grouped = defaultdict(list)
    for placement in placements_img:
        grouped[placement['compositionId']].append(placement)
    return grouped


def _resolve_design_boxes(comp, placements, box_id2data):
    """Return the box list aligned with placements: from the composition or looked up by boxId."""
    if comp['boxes'] is not None:
        return comp['boxes']
    return [box_id2data.get(placement['boxId']) for placement in placements]


def _find_image_for_photo(image_files, photo_id):
    """First filename whose name starts with `photo_id`, or None."""
    prefix = f"{photo_id}"
    for name in image_files:
        if name.startswith(prefix):
            return name
    return None


def _box_to_page_rect(box, page_width, page_height):
    """Convert relative box coordinates into a (x, y, w, h) pixel rect."""
    return (
        box['x'] * page_width,
        box['y'] * page_height,
        box['width'] * page_width,
        box['height'] * page_height,
    )


def _load_cropped_image(img_path, placement, box_w, box_h):
    """Open the source image, crop by placement ratios, resize to fit the box, return PNG bytes."""
    with Image.open(img_path) as img:
        width, height = img.size
        crop_x = int(placement['cropX'] * width)
        crop_y = int(placement['cropY'] * height)
        crop_w = int(placement['cropWidth'] * width)
        crop_h = int(placement['cropHeight'] * height)
        cropped = img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
        # Height is doubled to match the rest of the rendering pipeline.
        cropped = cropped.resize((int(box_w), int(box_h * 2)))
        buf = io.BytesIO()
        cropped.save(buf, format='PNG')
        buf.seek(0)
        return buf


def _draw_composition_header(c, comp_id, design_id, page_height, is_artificial_time=False):
    c.setFont("Helvetica", 10)
    c.drawString(30, page_height - 30, f"Composition ID: {comp_id}, Design ID: {design_id}")
    if is_artificial_time:
        c.setFont("Helvetica-Bold", 10)
        c.setFillColorRGB(1, 0, 0)
        c.drawString(30, page_height - 44, "ARTIFICIAL TIME APPLIED")
        c.setFillColorRGB(0, 0, 0)


def _draw_image_in_box(c, img_io, box_rect, page_height):
    """reportlab uses bottom-left origin, so flip y."""
    box_x, box_y, box_w, box_h = box_rect
    c.drawImage(ImageReader(img_io), box_x, page_height - box_y - box_h, width=box_w, height=box_h)


def _draw_error_in_box(c, photo_id, box_rect, page_height):
    box_x, box_y, _, _ = box_rect
    c.setFillColorRGB(1, 0, 0)
    c.drawString(box_x, page_height - box_y - 10, f"Error: {photo_id}")
    c.setFillColorRGB(0, 0, 0)


def _format_photo_time(row, is_artificial_time):
    """Human-readable per-photo time for the debug stamp.

    Real EXIF -> the actual datetime. Artificial-time galleries have a stale,
    identical image_time_date, so render the synthetic general_time (seconds
    from the first photo) as an elapsed H:MM:SS — readable and distinct.
    """
    if is_artificial_time:
        seconds = row.get('general_time', None)
        if pd.notnull(seconds):
            try:
                return str(timedelta(seconds=int(seconds)))
            except (ValueError, TypeError):
                return str(seconds)
        return ''
    return str(row.get('image_time_date', ''))


def _draw_photo_metadata(c, photo_id, gallery_photos_info, box_rect, page_height, is_artificial_time=False):
    """Stamp time, group key, and original context inside the box bottom — red, 8pt."""
    info_row = gallery_photos_info.loc[gallery_photos_info['image_id'] == photo_id]
    if info_row.empty:
        return

    row = info_row.iloc[0]
    general_time = _format_photo_time(row, is_artificial_time)
    original_context = row.get('original_context', '')
    group_key = (
        row.get('time_cluster', ''),
        row.get('cluster_context', ''),
        row.get('group_sub_index', ''),
    )

    box_x, box_y, _, box_h = box_rect
    base_y = page_height - box_y - box_h

    c.setFont("Helvetica", 8)
    c.setFillColorRGB(1, 0, 0)
    c.drawString(box_x, base_y + 18, f"{general_time}")
    c.drawString(box_x, base_y + 10, f"{group_key}")
    c.drawString(box_x, base_y + 2, f"{original_context}")
    c.setFillColorRGB(0, 0, 0)


def _render_placement(c, placement, box, image_files, images_path,
                      gallery_photos_info, page_width, page_height, is_artificial_time=False):
    """Render one photo placement: image (cropped & sized) plus its metadata, or an error label."""
    photo_id = placement['photoId']
    if photo_id is None or not box:
        return

    image_name = _find_image_for_photo(image_files, photo_id)
    if image_name is None:
        return

    img_path = os.path.join(images_path, image_name)
    box_rect = _box_to_page_rect(box, page_width, page_height)
    _, _, box_w, box_h = box_rect

    try:
        img_io = _load_cropped_image(img_path, placement, box_w, box_h)
        _draw_image_in_box(c, img_io, box_rect, page_height)
    except Exception:
        _draw_error_in_box(c, photo_id, box_rect, page_height)
        return

    _draw_photo_metadata(c, photo_id, gallery_photos_info, box_rect, page_height, is_artificial_time)


def _render_composition_page(c, comp, placements, box_id2data, image_files, images_path,
                             gallery_photos_info, page_width, page_height, is_artificial_time=False):
    """Render one composition as a single PDF page: header on top, then every placement."""
    design_boxes = _resolve_design_boxes(comp, placements, box_id2data)
    _draw_composition_header(c, comp['compositionId'], comp['designId'], page_height, is_artificial_time)
    for placement, box in zip(placements, design_boxes):
        _render_placement(c, placement, box, image_files, images_path,
                          gallery_photos_info, page_width, page_height, is_artificial_time)
    c.showPage()


def visualize_album_to_pdf(final_album, images_path, output_pdf_path, box_id2data, gallery_photos_info,
                           is_artificial_time=False):
    """
    Visualize the album in a PDF file: one composition per landscape A4 page.

    Args:
        final_album: dict, as returned by process_gallery.
        images_path: str, directory where images are stored.
        output_pdf_path: str, path to save the PDF.
        box_id2data: dict, mapping boxId to box info (with x, y, width, height).
        gallery_photos_info: pd.DataFrame with one row per photo (image_id, image_time_date,
            time_cluster, cluster_context, group_sub_index, original_context).
        is_artificial_time: bool, whether synthetic time was applied to this gallery (stamped
            as a banner on each page when True).
    """
    composition = final_album['composition']
    compositions = composition['compositions']
    placements_by_comp = _group_placements_by_composition(composition['placementsImg'])

    image_files = os.listdir(images_path)

    page_width, page_height = landscape(A4)
    c = canvas.Canvas(output_pdf_path, pagesize=(page_width, page_height))
    for comp in compositions:
        placements = placements_by_comp.get(comp['compositionId'], [])
        _render_composition_page(c, comp, placements, box_id2data, image_files, images_path,
                                 gallery_photos_info, page_width, page_height, is_artificial_time)
    c.save()


class Source:
    def __init__(self, id):
        self.id = id


def get_selection(message, logger):
    """Run the selection substages over one message.

    This used to be a near-copy of `SelectionStage.get_selection`, and the two
    had already drifted apart. Both now drive the same pipeline, so a local run
    exercises exactly what the service does.
    """
    start = datetime.now()

    try:
        context = build_select(logger=logger).run(
            AlbumContext.for_message(message, logger=logger)
        )
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        filename, lineno, func, text = tb[-1]
        logger.error(f"Error selection stage: {e}. Exception in function: {func}, line {lineno}, file {filename}.")
        raise Exception(f"Error selection stage: {e}. Exception in function: {func}, line {lineno}, file {filename}.")

    if context.failed:
        logger.error(f"Error for Selection images for this message {message}")
        message.error = f"Error for Selection images for this message {message}"
        return message

    message = context.sync_to_message()

    selection = context.selection
    if not selection.manual:
        logger.info('Photos selected: {}'.format(sorted(selection.photo_ids)))
        logger.info('Spreads dict sum: {}'.format(sum(selection.spreads.values())))
    logger.info('Selection took {}'.format(datetime.now() - start))

    return message


def process_gallery(input_request):
    message = Message(Source(1), input_request, None, datetime.now())
    msgs = [message]
    logger = get_logger(__name__, 'DEBUG')

    try:
        connection_string = get_variable(CONFIGS["DB_CONNECTION_STRING_VAR"])
        client = MongoClient(connection_string)
        db = client[CONFIGS["DB_NAME"]]
        project_status_collection = db[CONFIGS["STATUS_COLLECTION_NAME"]]
    except Exception as ex:
        logger.error(f"Failed to connect to database: {ex}")
    try:
        qdrant_client = QdrantClient(host=CONFIGS["QDRANT_HOST"],
                                          port=6333,
                                          # The HTTP port is often used for general access if not explicitly setting grpc_port
                                          grpc_port=6334,  # Explicitly define the gRPC port
                                          prefer_grpc=True
                                          # This forces the client to use gRPC for large operations like upsert
                                          )
        logger.info(f'Initialize qdrant client, host {CONFIGS["QDRANT_HOST"]}, port 6333, grpc_port 6334')
    except Exception as ex:
        logger.error(f"Failed to connect to Qdrant: {ex}")
    msgs, reading_error = read_messages(msgs, project_status_collection, qdrant_client, logger)
    if reading_error is not None:
        print(f"Reading error: {reading_error}")
        return reading_error, None
    message = get_selection(msgs[0], logger)

    process_stage = ProcessStage(logger=logger)
    message = process_stage.process_message(message)
    final_album_result = message.album_doc

    return final_album_result, message


def _build_arg_parser():
    import argparse

    ap = argparse.ArgumentParser(
        description="Run one album request locally and render the result to PDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
modes
  saved request (default)
    process_gallery.py <input_dir> <output_dir>
    process_gallery.py <input_dir> <output_dir> --request 53496523

  reproduce production
    process_gallery.py <input_dir> <output_dir> --from-datadog
    process_gallery.py <input_dir> <output_dir> --from-datadog --project-id 53496523

    Finds the newest request the service completed, saves it under
    files/test_requests/<projectId>.json, downloads that gallery's photos into
    <input_dir>/<projectId>/ and runs it. Needs DD_API_KEY and DD_APP_KEY for
    the log lookup, and the Azure network (VPN) for the photos.
""",
    )
    ap.add_argument("input_dir",
                    help="Root for gallery photos. Each gallery lives in <input_dir>/<projectId>/.")
    ap.add_argument("output_dir",
                    help="Root for rendered albums. Written to <output_dir>/<projectId>/.")

    source = ap.add_mutually_exclusive_group()
    source.add_argument("--request", metavar="NAME",
                        help="Saved request to run: a name under files/test_requests/ "
                             f"or a path to a .json. Default: {request_name}")
    source.add_argument("--from-datadog", action="store_true",
                        help="Reproduce the newest request the production service processed "
                             "successfully, from its logs.")

    dd = ap.add_argument_group("--from-datadog options")
    dd.add_argument("--project-id", type=int,
                    help="Reproduce this project's newest successful run instead of the newest overall.")
    dd.add_argument("--lookback-hours", type=int, default=48,
                    help="How far back to search the logs. Default: 48.")
    dd.add_argument("--no-download", action="store_true",
                    help="Skip the photo download (the PDF will have gaps unless they are already local).")
    dd.add_argument("--all-gallery-photos", action="store_true",
                    help="Download the whole gallery, not just the photos the request named.")
    dd.add_argument("--max-photos", type=int,
                    help="Cap how many photos to download.")

    ap.add_argument("--album-name", default=album_name,
                    help=f"Base name for the rendered PDF. Default: {album_name}")
    return ap


def _resolve_request(args, log):
    """Return (request dict, label used for the local folders)."""
    from tools import local_request as lr

    if not args.from_datadog:
        name = args.request or request_name
        request = lr.load_request(name)
        log(f"request: {lr.request_path(name)}")
        return request, str(request["projectId"])

    log("searching Datadog for the newest successful album run...")
    client = lr.DatadogLogs()
    run = lr.find_latest_successful_request(
        client, lookback_hours=args.lookback_hours, project_id=args.project_id
    )
    log(f"found: {run.summary()}")

    saved = lr.save_request(run.request, str(run.project_id))
    log(f"saved request: {saved}  (re-run it with --request {run.project_id})")
    return run.request, str(run.project_id)


def _ensure_photos(args, request, project_dir, log):
    """Download the gallery's photos into <input_dir>/<projectId>/."""
    from tools import local_request as lr

    if args.no_download:
        log("photo download skipped (--no-download)")
        return

    base_url = request.get("base_url")
    if not base_url:
        log("! request has no base_url; cannot download photos")
        return

    photo_ids = None if args.all_gallery_photos else (request.get("photos") or None)
    log(f"downloading photos -> {project_dir}")
    try:
        counts = lr.download_gallery_photos(
            base_url, project_dir, photo_ids=photo_ids,
            max_photos=args.max_photos, log=log,
        )
    except Exception as ex:  # noqa: BLE001 - the run may still work off local files
        log(f"! photo download failed ({type(ex).__name__}: {ex})")
        log("  the album will still be built; the PDF will have gaps for missing photos")
        return

    log("  " + ", ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == '__main__':
    args = _build_arg_parser().parse_args()
    log = print

    settings_filename = os.environ.get('HostingSettingsPath',
                                       '/ptinternal/pictures/hosting/ai_settings_audiobeat.json.txt')
    intialize('AlbumDesigner', settings_filename)

    _input_request, project_label = _resolve_request(args, log)

    _images_path = os.path.join(args.input_dir, project_label)
    _output_dir = os.path.join(args.output_dir, project_label)
    os.makedirs(_images_path, exist_ok=True)
    os.makedirs(_output_dir, exist_ok=True)

    if args.from_datadog:
        _ensure_photos(args, _input_request, _images_path, log)

    # Run request
    final_album, _message = process_gallery(_input_request)
    if _message is None:
        raise SystemExit(f"process_gallery failed: {final_album}")

    gallery_photos_info = _message.content['gallery_photos_info']
    box_id2data = _message.designsInfo['anyPagebox_id2data']

    is_artificial_time = _message.content.get('is_artificial_time', False)
    print('ARTIFICIAL TIME APPLIED:', is_artificial_time)

    print('FINAL SPREADS', len(final_album['composition']['compositions']))
    print(final_album)

    # Debug with Plotting
    _output_pdf_path = os.path.join(_output_dir, args.album_name + '.pdf')

    visualize_album_to_pdf(final_album, _images_path, _output_pdf_path, box_id2data, gallery_photos_info,
                           is_artificial_time)
    print('album saved locally:', _output_pdf_path)
