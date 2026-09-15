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

from src.pipeline import AlbumContext, build_select, compose_albums

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from PIL import Image
import io
from utils.configs import CONFIGS
from src.predefined.models import PredefinedLayoutInput

from ptinfra.pt_queue import Message
from main import ProcessStage

#: Set by --non-wedding; read in `process_gallery`.
FORCE_NON_WEDDING = False

#: Every album composed by the last run, for --albums verification.
ALBUM_RUNS = []

request_name = 'request_cameron_predefined'
album_name = 'album_predefined'


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
        # Third route, beside manual and AI: an external service fixed the
        # spreads, so selection is not run at all. Mirrors SelectionStage in
        # main.py, which is the point of this function -- a local run must
        # exercise what the service does.
        predefined = PredefinedLayoutInput.from_request(message.content)
        if predefined is not None:
            df = message.content.get('gallery_photos_info', pd.DataFrame())
            if df.empty:
                raise Exception(f"Gallery photos info DataFrame is empty for message {message}")
            message.content['predefined_layout'] = predefined
            message.content['gallery_all_photos_info'] = df.copy()
            message.content['gallery_photos_info'] = df[df['image_id'].isin(predefined.all_photo_ids())]
            logger.info(f"Predefined layout: {len(predefined.spreads)} spreads, skipping selection.")
            return message

        # Everything the branch had inline after this point is the pipeline's
        # now: `select.route` splits manual from AI and narrows the pool,
        # `select.budget` sets the spread counts, and `select.publish` writes
        # `photos`, `spreads_dict`, the spread bounds and the `bride and groom`
        # frame ProcessStage reads back.
        # Mirrors SelectionStage: one album per variant, each from its own
        # copy of the base. N=1 here until enrich.variants lands.
        settings = CONFIGS.get('albums', {})
        runs = compose_albums(message, build_select(logger=logger),
                              count=int(settings.get('count', 1)),
                              logger=logger, seed=settings.get('seed'))
        for run in runs:
            if not run.context.failed:
                run.context.sync_to_message()
        ALBUM_RUNS.extend(runs)
        context = runs[0].context
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

    # Local-only override for testing the non-wedding path. `is_wedding` comes
    # from `classify_gallery_type` over the photos, and it is wrong on some
    # genuinely non-wedding galleries -- a debutante ball reads as a wedding
    # (white gowns, tuxedos, bouquets, garden portraits). Until such galleries
    # route correctly on their own, this is how the non-wedding path gets
    # exercised on real content. Never set in the service.
    if FORCE_NON_WEDDING:
        was = msgs[0].content.get('is_wedding')
        msgs[0].content['is_wedding'] = False
        # `content` alone is not enough. The read stage leaves its AlbumContext
        # attached to the message, and `AlbumContext.for_message` reuses that
        # object rather than rebuilding from content -- so selection reads
        # `facts.is_wedding` from the context, and a content-only override is
        # silently ignored: budget and preselect (both wedding-gated) still run
        # and `select.narrator` still declines.
        attached = getattr(msgs[0], AlbumContext._MESSAGE_SLOT, None)
        if isinstance(attached, AlbumContext):
            attached.facts.is_wedding = False
        logger.warning(f"--non-wedding: is_wedding {was} -> False "
                       f"(local override; context {'patched' if attached else 'absent'})")

    message = get_selection(msgs[0], logger)

    # Lay out every album, not just the first. ProcessStage already accepts a
    # list -- which is how the service receives sibling messages -- so passing
    # the whole set is both what production does and the only way each album
    # ends up with an `album_doc` for the reply. One album passes a single
    # message, exactly as before.
    process_stage = ProcessStage(logger=logger)
    to_lay_out = [run.message for run in ALBUM_RUNS] if len(ALBUM_RUNS) > 1 else message
    laid_out = process_stage.process_message(to_lay_out)
    message = laid_out[0] if isinstance(laid_out, list) else laid_out
    final_album_result = message.album_doc

    return final_album_result, message


def _albums_to_render(final_album, message):
    """``(filename suffix, album_doc, message)`` for every album composed.

    A single-album run keeps the bare `--album-name`, so the file a normal run
    writes is exactly the file it always wrote. A multi-album run suffixes each
    with its variant name, because "the album" is no longer a single thing and
    overwriting one PDF with the next would hide that.
    """
    if len(ALBUM_RUNS) <= 1:
        return [("", final_album, message)]

    rendered = []
    for run in ALBUM_RUNS:
        name = run.context.variant_name or f"album{run.index}"
        rendered.append((f"_{name}", getattr(run.message, 'album_doc', None), run.message))
    return rendered


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

    Finds the newest request the service completed and saves it under
    files/test_requests/<projectId>.json, so it can be replayed later with
    --request <projectId>. Needs DD_API_KEY and DD_APP_KEY.

Both modes download the gallery's photos into <input_dir>/<projectId>/ (the
Azure network, i.e. VPN, is needed for that) and render the album to
<output_dir>/<projectId>/. Photos already on disk are skipped, so re-runs are
cheap. Pass --no-download to work purely off what is already local.
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

    ph = ap.add_argument_group("photos (both modes)")
    ph.add_argument("--no-download", action="store_true",
                    help="Skip the photo download (the PDF will have gaps unless they are already local).")
    ph.add_argument("--all-gallery-photos", action="store_true",
                    help="Download the whole gallery, not just the photos the request named.")
    ph.add_argument("--max-photos", type=int,
                    help="Cap how many photos to download.")

    ap.add_argument("--album-name", default=album_name,
                    help=f"Base name for the rendered PDF. Default: {album_name}")
    picker = ap.add_mutually_exclusive_group()
    picker.add_argument("--cp-sat", action="store_true",
                        help="Pick with the one-shot CP-SAT model. This is the default now, so "
                             "the flag only makes it explicit.")
    ap.add_argument("--narrator", action="store_true",
                    help="Enable select.narrator (the albumNarrator policy) for this run. "
                         "It only serves non-wedding galleries with 768-d embeddings.")
    ap.add_argument("--albums", type=int, default=None,
                    help="Compose N albums from the one gallery read (Phase 2 of "
                         "docs/multi_album_plan.md). Implies --seed-albums so the "
                         "N are comparable; only the first is laid out and rendered.")
    ap.add_argument("--seed-albums", type=int, default=12345,
                    help="Seed both global RNGs identically before each album, so "
                         "identical variants are comparable. Only used with --albums.")
    ap.add_argument("--non-wedding", action="store_true",
                    help="Force is_wedding=False after the read. For testing the "
                         "non-wedding path on a gallery the content classifier calls a "
                         "wedding; never used by the service.")
    picker.add_argument("--loop", action="store_true",
                        help="Pick with the per-category loop instead (WeddingPicker and its "
                             "strategies) -- the old default, for a side-by-side comparison.")
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


def _exit_now(code=0, message=None):
    """End the process without waiting for ptinfra's queue thread.

    `ptinfra.intialize` registers a **non-daemon** `ElasticQueueThread`, so a
    normal return -- or a `SystemExit` -- unwinds the main thread and then
    blocks forever waiting for that one to finish. The album is already on
    disk by then, so the run looks finished and simply never exits, and each
    one leaves an interpreter alive holding the whole photo table and its
    embeddings. Fifteen albums in a session is enough to exhaust memory.

    `os._exit` skips interpreter shutdown entirely, which is why the streams
    are flushed by hand first: it does not run `atexit`, and anything sitting
    in a buffer would be lost.

    Local driver only. The service's own shutdown is ptinfra's business and
    `main.py` does not do this -- there the queue thread is the point.
    """
    if message:
        print(message, file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == '__main__':
    args = _build_arg_parser().parse_args()
    log = print

    if args.albums:
        CONFIGS['albums'] = {**CONFIGS.get('albums', {}), 'count': args.albums,
                             'seed': args.seed_albums}
        log(f"composing {args.albums} albums (seed {args.seed_albums})")

    if args.narrator:
        CONFIGS['narrator'] = {**CONFIGS.get('narrator', {}), 'enabled': True}
        log("select.narrator: enabled")
    if args.non_wedding:
        FORCE_NON_WEDDING = True
        log("is_wedding will be forced False after the read (--non-wedding)")

    if args.loop:
        CONFIGS['pick_cpsat'] = {**CONFIGS.get('pick_cpsat', {}), 'enabled': False}
        log("select.pick: per-category loop (CP-SAT disabled for this run)")
    elif args.cp_sat:
        CONFIGS['pick_cpsat'] = {**CONFIGS.get('pick_cpsat', {}), 'enabled': True}
        log("select.pick: CP-SAT model (the default)")

    settings_filename = os.environ.get('HostingSettingsPath',
                                       '/ptinternal/pictures/hosting/ai_settings_audiobeat.json.txt')
    intialize('AlbumDesigner', settings_filename)

    _input_request, project_label = _resolve_request(args, log)

    _images_path = os.path.join(args.input_dir, project_label)
    _output_dir = os.path.join(args.output_dir, project_label)
    os.makedirs(_images_path, exist_ok=True)
    os.makedirs(_output_dir, exist_ok=True)

    # Replaying a saved request needs the photos just as much as reproducing a
    # fresh one; the request carries base_url either way.
    _ensure_photos(args, _input_request, _images_path, log)

    # Run request
    final_album, _message = process_gallery(_input_request)
    if _message is None:
        _exit_now(1, f"process_gallery failed: {final_album}")

    is_artificial_time = _message.content.get('is_artificial_time', False)
    print('ARTIFICIAL TIME APPLIED:', is_artificial_time)

    if len(ALBUM_RUNS) > 1:
        chosen = [tuple(run.photo_ids) for run in ALBUM_RUNS]
        distinct = len(set(chosen))
        for run in ALBUM_RUNS:
            name = run.context.variant_name or f'album {run.index}'
            focus = run.context.hints.focus
            print(f'ALBUM {run.index} variant={name} focus={focus} '
                  f'photos={len(run.photo_ids)}')
        print(f'ALBUMS COMPOSED {len(ALBUM_RUNS)}, '
              f'photos each {[len(c) for c in chosen]}, '
              f'distinct selections {distinct}')
        print('ALBUMS IDENTICAL' if distinct == 1 else 'ALBUMS DIFFER')
        # The reply that would go on the queue for this request.
        from src.album_response import build_reply, encoded_size
        reply = build_reply(
            [getattr(r.message, 'album_doc', None) for r in ALBUM_RUNS],
            [getattr(r.message, 'variant_name', None) for r in ALBUM_RUNS])
        if reply is not None:
            print('REPLY keys', sorted(reply))
            entries = reply.get('albums') or []
            for entry in entries:
                print('REPLY album', entry['albumIndex'],
                      'variant', entry.get('variant'),
                      'spreads', len(entry['composition']['compositions']))
            if not entries:
                print('REPLY single album (no `albums` key)')
            print('REPLY encoded', encoded_size(reply), 'bytes of 65536 cap,',
                  'omitted', reply.get('albumsOmitted', 0))

    print('FINAL SPREADS', len(final_album['composition']['compositions']))
    print(final_album)

    # Debug with Plotting. One PDF per album composed, not just the first:
    # rendering `runs[0]` alone meant the *variant* album -- the whole point of
    # a multi-album run -- could only be inspected through the log, so the one
    # artifact a local run produces was always the album the variant changed
    # least. Each album carries its own `album_doc` and its own selected frame,
    # so each renders from its own message.
    for suffix, album_doc, album_message in _albums_to_render(final_album, _message):
        if album_doc is None:
            print(f'! no album_doc for {suffix or "the album"}; nothing to render')
            continue
        _output_pdf_path = os.path.join(_output_dir, args.album_name + suffix + '.pdf')
        visualize_album_to_pdf(
            album_doc, _images_path, _output_pdf_path,
            album_message.designsInfo['anyPagebox_id2data'],
            album_message.content['gallery_photos_info'],
            is_artificial_time)
        print('album saved locally:', _output_pdf_path)

    # The PDF is written and closed, so there is nothing left to wait for.
    _exit_now(0)
