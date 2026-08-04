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

from src.selection.auto_selection import ai_selection

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from PIL import Image
import io
from utils.lookup_table_tools import wedding_lookup_table
from utils.configs import CONFIGS
from src.predefined.models import PredefinedLayoutInput

from ptinfra.pt_queue import Message
from main import ProcessStage

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
    start = datetime.now()
    # Iterate over message and start the selection process
    try:
        predefined = PredefinedLayoutInput.from_request(message.content)
        if predefined is not None:
            # Predefined-spreads mode: skip AI selection, narrow gallery to the
            # union of spread + cover photos.
            df = message.content.get('gallery_photos_info', pd.DataFrame())
            if df.empty:
                raise Exception(f"Gallery photos info DataFrame is empty for message {message}")
            message.content['predefined_layout'] = predefined
            message.content['gallery_all_photos_info'] = df.copy()
            message.content['gallery_photos_info'] = df[df['image_id'].isin(predefined.all_photo_ids())]
            logger.info(f"Predefined layout: {len(predefined.spreads)} spreads, skipping selection.")
            return message

        ai_metadata = message.content.get('aiMetadata', None)
        # condition for  manual selection
        if ai_metadata is None or ai_metadata['photoIds'] is None:
            logger.info(f"aiMetadata not found for message {message}. Continue with chosen photos.")
            photos = message.content.get('photos', [])
            df = pd.DataFrame(photos, columns=['image_id'])
            message.content['gallery_photos_info'] = df.merge(message.content['gallery_photos_info'], how='inner', on='image_id')
            # handle LUT for manual selection
            is_wedding = message.content.get('is_wedding', False)
            if is_wedding:
                modified_lut = wedding_lookup_table.copy()  # Create a copy to avoid modifying the original LUT
                modified_lut['Other'] = (24, 4)  # Set 'Other' event to have max spreads
                modified_lut['None'] = (24, 4)
                message.content['modified_lut'] = modified_lut
            message.content['manual_selection'] = True
            return message

        available_photos = message.content.get('photos', [])
        df = message.content.get('gallery_photos_info', pd.DataFrame())
        if df.empty:
            logger.error(f"Gallery photos info DataFrame is empty for message {message}")
            message.content['error'] = f"Gallery photos info DataFrame is empty for message {message}"
            raise Exception(f"Gallery photos info DataFrame is empty for message {message}")
        if len(available_photos) != 0:
            df = df[df['image_id'].isin(available_photos)]
            message.content['gallery_photos_info'] = df

        message.content['gallery_all_photos_info'] = df.copy()

        ten_photos = ai_metadata.get('photoIds', [])
        people_ids = ai_metadata.get('personIds', [])
        focus = ai_metadata.get('focus', ['everyoneElse'])
        tags = ai_metadata.get('subjects', ['Wedding dress', 'ceremony', 'bride', 'dancing', 'bride getting ready',
                                            'groom getting ready', 'table setting', 'flowers', 'decorations', 'family',
                                            'baby', 'kids', 'mother', 'father', 'Romance', 'affection', 'Intimacy',
                                            'Happiness', 'Holding hands', 'smiling', 'Hugging', 'Kissing', 'ring',
                                            'veil', 'soft light', 'portrait'])
        density = ai_metadata.get('density', 3)
        rating = message.content.get('rating', [])

        is_wedding = message.content.get('is_wedding', False)

        if df.empty:
            logger.error(f"Gallery photos info DataFrame is empty for message {message}")
            message.content['error'] = f"Gallery photos info DataFrame is empty for message {message}"
            return message

        if is_wedding:
            modified_lut = wedding_lookup_table.copy()  # Create a copy to avoid modifying the original LUT

            density_factor = CONFIGS['density_factors'][density] if density in CONFIGS['density_factors'] else 1
            for event, pair in modified_lut.items():
                modified_lut[event] = (min(24, max(1, pair[0] * density_factor)), pair[1])  # Ensure base spreads are at least 1 and not above 24
        else:
            modified_lut = None

        message.content['modified_lut'] = modified_lut
        is_artificial_time = message.content['is_artificial_time']

        ai_photos_selected, spreads_dict, min_total_spreads, max_total_spreads, errors = ai_selection(df, ten_photos, people_ids, focus, tags, is_wedding, density,is_artificial_time,
                                                  logger)

        if errors:
            logger.error(f"Error for Selection images for this message {message}")
            message.error = f"Error for Selection images for this message {message}"
            return message

        filtered_df = df[df['image_id'].isin(ai_photos_selected)]
        message.content['gallery_photos_info'] = filtered_df
        message.content['photos'] = ai_photos_selected
        message.content['spreads_dict'] = spreads_dict
        message.content['min_total_spreads'] = min_total_spreads
        message.content['max_total_spreads'] = max_total_spreads
        logger.info('Photos selected: {}'.format(sorted(ai_photos_selected)))
        logger.info('Spreads dict sum: {}'.format(sum([item for key, item in spreads_dict.items()])))

        if message.pagesInfo.get("firstPage"):
            if message.content.get('is_wedding', True):
                all_bride_groom = df[
                    (df["cluster_context"] == "bride and groom")]
                message.content['bride and groom'] = all_bride_groom
        else:
            message.content['bride and groom'] = None

        return message

    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        filename, lineno, func, text = tb[-1]
        logger.error(f"Error selection stage: {e}. Exception in function: {func}, line {lineno}, file {filename}.")
        raise Exception(f"Error selection stage: {e}. Exception in function: {func}, line {lineno}, file {filename}.")


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


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir",
                        help="The path to the directory on your system where the photos are stored. Each set inside should be named with the projectId number.")
    parser.add_argument("output_dir",
                        help="The path to the directory where created album should be saved.")
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    # PyCharm: Run -> Edit Configurations -> Script parameters
    # C:\Users\user\Desktop\PicTime\AlbumDesigner\dataset\ C:\Users\user\Desktop\PicTime\AlbumDesigner\output
    # add paths without argument names

    settings_filename = os.environ.get('HostingSettingsPath',
                                       '/ptinternal/pictures/hosting/ai_settings_audiobeat.json.txt')
    intialize('AlbumDesigner', settings_filename)

    with open(f'files/test_requests/{request_name}.json', 'r') as f:
        _input_request = json.load(f)

    # Run request
    final_album, _message = process_gallery(_input_request)
    if _message is None:
        raise SystemExit(f"process_gallery failed: {final_album}")

    gallery_photos_info = _message.content['gallery_photos_info']
    box_id2data = _message.designsInfo['anyPagebox_id2data']  # if 'designsInfo' in _message and 'anyPagebox_id2data' in _message['designsInfo'] else {}

    is_artificial_time = _message.content.get('is_artificial_time', False)
    print('ARTIFICIAL TIME APPLIED:', is_artificial_time)

    print('FINAL SPREADS', len(final_album['composition']['compositions']))
    print(final_album)

    # Debug with Plotting
    id = str(_input_request["projectId"])
    _images_path = os.path.join(input_dir, id)
    _output_pdf_path = os.path.join(output_dir, id)
    os.makedirs(_output_pdf_path, exist_ok=True)
    _output_pdf_path = os.path.join(_output_pdf_path, album_name + '.pdf')

    visualize_album_to_pdf(final_album, _images_path, _output_pdf_path, box_id2data, gallery_photos_info,
                           is_artificial_time)
    print('album saved locally')


