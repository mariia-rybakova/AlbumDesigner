import os
import pandas as pd

from typing import Dict
from datetime import datetime
import multiprocessing as mp
from collections import defaultdict

from ptinfra import get_logger

from src.request_processing import read_messages

from src.smart_cropping import process_crop_images
from src.core.key_pages import generate_first_last_pages
from utils.time_processing import process_image_time, get_time_clusters, merge_time_clusters_by_context
from src.album_processing import album_processing
from src.request_processing import assembly_output
from src.selection.auto_selection import ai_selection

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from PIL import Image
import io
from utils.lookup_table_tools import wedding_lookup_table
from utils.configs import CONFIGS

def visualize_album_to_pdf(final_album, images_path, output_pdf_path, box_id2data, gallery_photos_info):
    """
    Visualize the album in a PDF file.
    Args:
        final_album: dict, as returned by process_gallery
        images_path: str, directory where images are stored
        output_pdf_path: str, path to save the PDF
        box_id2data: dict, mapping boxId to box info (with x, y, width, height)
        gallery_photos_info: pd.DataFrame, contains info for each photo (general_time, cluster_context)
    """

    composition = final_album['composition']
    compositions = composition['compositions']
    placementsImg = composition['placementsImg']

    placements_by_comp = defaultdict(list)
    for placement in placementsImg:
        placements_by_comp[placement['compositionId']].append(placement)

    # Use landscape A4
    page_width, page_height = landscape(A4)
    c = canvas.Canvas(output_pdf_path, pagesize=(page_width, page_height))

    for comp in compositions:
        comp_id = comp['compositionId']
        placements = placements_by_comp.get(comp_id, [])
        c.setFont("Helvetica", 10)
        c.drawString(30, page_height - 30, f"Composition ID: {comp_id}")
        for placement in placements:
            photo_id = placement['photoId']
            box_id = placement['boxId']
            all_images_names = os.listdir(images_path)
            all_images_with_this_name = [img for img in all_images_names if img.startswith(f"{photo_id}")]
            img_path = os.path.join(images_path, all_images_with_this_name[0])
            box = box_id2data.get(box_id)
            if not box:
                continue
            # Box coordinates and size (relative to page)
            box_x = box['x'] * page_width
            box_y = box['y'] * page_height
            box_w = box['width'] * page_width
            box_h = box['height'] * page_height
            try:

                with Image.open(img_path) as img:
                    # Crop image according to cropX, cropY, cropWidth, cropHeight (assume these are ratios)
                    width, height = img.size
                    crop_x = int(placement['cropX'] * width)
                    crop_y = int(placement['cropY'] * height)
                    crop_w = int(placement['cropWidth'] * width)
                    crop_h = int(placement['cropHeight'] * height)
                    crop_box = (crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)
                    cropped_img = img.crop(crop_box)
                    # Resize to fit the box, but divide width by 2
                    cropped_img = cropped_img.resize((int(box_w), int(box_h*2)))
                    img_io = io.BytesIO()
                    cropped_img.save(img_io, format='PNG')
                    img_io.seek(0)
                    # In reportlab, (0,0) is at the bottom-left
                    c.drawImage(ImageReader(img_io), box_x, page_height - box_y - box_h, width=box_w, height=box_h)
            except Exception as e:
                c.setFillColorRGB(1, 0, 0)
                c.drawString(box_x, page_height - box_y - 10, f"Error: {photo_id}")
                c.setFillColorRGB(0, 0, 0)
                continue
            # Print general_time and cluster_context at the bottom of the image (inside the box)
            info_row = gallery_photos_info.loc[gallery_photos_info['image_id'] == photo_id]
            if not info_row.empty:
                general_time = info_row.iloc[0].get('image_time_date', '')
                cluster_context = info_row.iloc[0].get('cluster_context', '')
                c.setFont("Helvetica", 8)
                c.setFillColorRGB(1, 0, 0)  # White color for bright text
                c.drawString(box_x, page_height - box_y - box_h + 10, f"{general_time}")
                c.drawString(box_x, page_height - box_y - box_h + 2, f"{cluster_context}")
                c.setFillColorRGB(0, 0, 0)  # Reset to black
        c.showPage()
    c.save()

class Source:
    def __init__(self, id):
        self.id = id

class Message:
    body: Dict
    source: Source

    def __init__(self, body, id):
        self.body = body
        self.source = Source(id)
        self.error = None

    @property
    def content(self) -> Dict:
        return self.body


def get_selection(message, logger):
    start = datetime.now()
    # Iterate over message and start the selection process
    try:
        ai_metadata = message.content.get('aiMetadata', None)
        if ai_metadata is None:
            logger.info(f"aiMetadata not found for message {message}. Continue with chosen photos.")
            photos = message.content.get('photos', [])
            df = pd.DataFrame(photos, columns=['image_id'])
            message.content['gallery_photos_info'] = df.merge(message.content['gallery_photos_info'], how='inner', on='image_id')
            return message
        photos = message.content.get('photos', [])
        if len(photos) != 0:
            return message

        ten_photos = ai_metadata.get('photoIds', [])
        people_ids = ai_metadata.get('personIds', [])
        focus = ai_metadata.get('focus', ['everyoneElse'])
        tags = ai_metadata.get('subjects', ['Wedding dress', 'ceremony', 'bride', 'dancing', 'bride getting ready',
                                            'groom getting ready', 'table setting', 'flowers', 'decorations', 'family',
                                            'baby', 'kids', 'mother', 'father', 'Romance', 'affection', 'Intimacy',
                                            'Happiness', 'Holding hands', 'smiling', 'Hugging', 'Kissing', 'ring',
                                            'veil', 'soft light', 'portrait'])
        density = ai_metadata.get('density', 3)
        df = message.content.get('gallery_photos_info', pd.DataFrame())
        is_wedding = message.content.get('is_wedding', False)

        if df.empty:
            logger.error(f"Gallery photos info DataFrame is empty for message {message}")
            message.content['error'] = f"Gallery photos info DataFrame is empty for message {message}"
            return message

        modified_lut = wedding_lookup_table.copy()  # Create a copy to avoid modifying the original LUT
        density_factor = CONFIGS['density_factors'][density] if density in CONFIGS['density_factors'] else 1
        for event, pair in modified_lut.items():
            modified_lut[event] = (min(24, max(1, pair[0] * density_factor)),
                                   pair[1])  # Ensure base spreads are at least 1 and not above 24
        message.content['modified_lut'] = modified_lut

        ai_photos_selected, spreads_dict, errors = ai_selection(df, ten_photos, people_ids, focus, tags, is_wedding, density,
                                                  logger)

        if errors:
            logger.error(f"Error for Selection images for this message {message}")
            message.error = f"Error for Selection images for this message {message}"
            return message

        filtered_df = df[df['image_id'].isin(ai_photos_selected)]
        message.content['gallery_photos_info'] = filtered_df
        message.content['photos'] = ai_photos_selected
        message.content['spreads_dict'] = spreads_dict
        logger.info('Photos selected: {}'.format(sorted(ai_photos_selected)))
        logger.info('Spreads dict sum: {}'.format(sum([item for key, item in spreads_dict.items()])))
        return message

    except Exception as e:
        # self.logger.error(f"Error reading messages: {e}")
        raise (e)
        # return []

def process_message(message, logger):
    # check if its single message or list
    params = [0.01, 100, 1000, 100, 300, 12]
    logger.debug("Params for this Gallery are: {}".format(params))

    df = message.content.get('gallery_photos_info', pd.DataFrame())
    df_serializable = df.copy()  # Make a copy to avoid modifying original
    df_serializable = df_serializable[['image_id', 'faces_info', 'background_centroid', 'diameter', 'image_as']]

    q = mp.Queue()
    p = mp.Process(target=process_crop_images, args=(q, df_serializable))
    p.start()

    try:
        stage_start = datetime.now()
        # Extract gallery photo info safely
        df = message.content.get('gallery_photos_info', pd.DataFrame())
        if df.empty:
            logger.error(f"Gallery photos info DataFrame is empty for message {message}")
            message.content['error'] = f"Gallery photos info DataFrame is empty for message {message}"
            return None

        # Sorting the DataFrame by "image_order" column
        sorted_df = df.sort_values(by="image_order", ascending=False)

        # Process time
        sorted_df, image_id2general_time = process_image_time(sorted_df)
        sorted_df['time_cluster'] = get_time_clusters(sorted_df['general_time'])
        if message.content['is_wedding']:
            sorted_df = merge_time_clusters_by_context(sorted_df, ['dancing'], logger)

        df, first_last_pages_data_dict = generate_first_last_pages(message, sorted_df, logger)

        if message.content.get('aiMetadata', None) is not None:
            density = message.content['aiMetadata'].get('density', 3)
        else:
            density = 3

        # Handle the processing time logging
        start = datetime.now()
        message.content['gallery_photos_info'] = df
        modified_lut = message.content['modified_lut'] if message.content.get('modified_lut', None) is not None else None
        album_result = album_processing(df, message.designsInfo, message.content['is_wedding'], modified_lut, params, logger=logger, density=density)

        wait_start = datetime.now()
        try:
            cropped_df = q.get(timeout=200)
        except Exception as e:
            p.terminate()
            raise Exception('cropping process not completed: {}'.format(e))
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
            logger.error('cropping process not completed 2')
            raise Exception('cropping process not completed.')

        df = df.merge(cropped_df, how='inner', on='image_id')
        for key, value in first_last_pages_data_dict.items():
            if first_last_pages_data_dict[key]['last_images_df'] is not None or first_last_pages_data_dict[key]['first_images_df'] is not None and first_last_pages_data_dict[key]['last_images_df'].shape[0] != 0 or first_last_pages_data_dict[key]['first_images_df'].shape[0] != 0:
                first_last_pages_data_dict[key]['last_images_df'] = value['last_images_df'].merge(cropped_df, how='inner', on='image_id')
                first_last_pages_data_dict[key]['first_images_df'] = value['first_images_df'].merge(cropped_df, how='inner', on='image_id')

        logger.debug('waited for cropping process: {}'.format(datetime.now() - wait_start))

        final_response = assembly_output(album_result, message, df, first_last_pages_data_dict, logger)

        processing_time = datetime.now() - start

        logger.debug('Lay-outing time: {}.For Processed album id: {}'.format(processing_time,
                                                                             message.content.get('projectURL', True)))
        logger.debug(
            'Processing Stage time: {}.For Processed album id: {}'.format(datetime.now() - stage_start,
                                                                          message.content.get('projectURL',True)))

    except Exception as e:
        logger.error(f"Unexpected error in message processing: {e}")
        raise Exception(f"Unexpected error in message processing: {e}")

    return final_response, message


def process_gallery(input_request):
    message = Message(input_request, id=1)
    msgs = [message]
    logger = get_logger(__name__, 'DEBUG')
    msgs, reading_error = read_messages(msgs, logger)
    if reading_error is not None:
        print(f"Reading error: {reading_error}")
        return reading_error, None
    message = get_selection(msgs[0], logger)

    final_album_result, message = process_message(message, logger)
    return final_album_result, message


if __name__ == '__main__':
    #_input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310, 'projectId': 46105850, 'userId': 548864249, 'userJobId': 1069943370, 'base_url': 'ptstorage_32://pictures/46/105/46105850/g59f42f8oml2n45x4s', 'photos': [10772592874, 10772592890, 10772592931, 10772592939, 10772592913, 10772593214, 10772593221, 10772593257, 10772593189, 10772593224, 10772593206, 10772593216, 10772593260, 10772593311, 10772593308, 10772593261, 10772593263, 10772593688, 10772593314, 10772593695, 10772593633, 10772593612, 10772593754, 10772593445, 10772593467, 10772594359, 10772594274, 10772594268, 10772594277, 10772593563, 10772593585, 10772594384, 10772594387, 10772594380, 10772594272, 10772594381, 10772594392], 'projectCategory': 0, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/91qecwibrecpki6p_wcwffdo.json', 'conditionId': 'AAD_46105850_86e53b01-758e-4d5c-8bcd-d43b4e02ec07.326.101', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}
    # _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310, 'projectId': 46245951, 'userId': 548224517, 'userJobId': 1069781153, 'base_url': 'ptstorage_32://pictures/46/245/46245951/ii52fnki40jq0i3xvu', 'photos': [10803725905, 10803725910, 10803725924, 10803725963, 10803725967, 10803725969, 10803725978, 10803725994, 10803725996, 10803725997, 10803726027, 10803726045, 10803726043, 10803726137, 10803726140, 10803726128, 10803726109, 10803726085, 10803726190, 10803726150, 10803726149, 10803726182, 10803726055, 10803726068, 10803726056, 10803726220, 10803726223, 10803726103, 10803726104, 10803726078, 10803726077, 10803726596, 10803726605, 10803726314, 10803726615, 10803726626, 10803726336, 10803726638, 10803726533, 10803726530, 10803726499, 10803726552, 10803726431, 10803726433, 10803726492], 'projectCategory': 0, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/henypeix2kyn2jrspfcd6wae.json', 'conditionId': 'AAD_46245951_bb226506-e333-4b98-bcee-e0ab624b64c9.208.429', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}
    # _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310, 'projectId': 46245951, 'userId': 547128248, 'userJobId': 1068714614, 'base_url': 'ptstorage_32://pictures/46/245/46245951/ii52fnki40jq0i3xvu', 'photos': [10803728384, 10803728502, 10803728538, 10803728560, 10803728527, 10803728549, 10803728582, 10803728547, 10803728593, 10803728385, 10803728396, 10803728407, 10803728418, 10803728429, 10803728440, 10803728571, 10803728451, 10803728462, 10803728473, 10803728484, 10803728503, 10803728514, 10803728519, 10803728520, 10803728521, 10803728522, 10803728523, 10803728524, 10803728525, 10803728526, 10803728528, 10803728529, 10803728530, 10803728531, 10803728532, 10803728533, 10803728534, 10803728550, 10803728535, 10803728536, 10803728537, 10803728548, 10803728539, 10803728540, 10803728541, 10803728542, 10803728543, 10803728551, 10803728544, 10803728545, 10803728546, 10803728552, 10803728555, 10803728553, 10803728554, 10803728556, 10803728558, 10803728419, 10803728557, 10803728420, 10803728559, 10803728561, 10803728562, 10803728563, 10803728564, 10803728565, 10803728566, 10803728567, 10803728568, 10803728569, 10803728570, 10803728572, 10803728573, 10803728577, 10803728578, 10803728579, 10803728580, 10803728574, 10803728581, 10803728576, 10803728583, 10803728584, 10803728585, 10803728586, 10803728587, 10803728588, 10803728589, 10803728575, 10803728590, 10803728591, 10803728592, 10803728594, 10803728595, 10803728596, 10803728597, 10803728598, 10803728599, 10803728600, 10803728601, 10803728386, 10803728602, 10803728603, 10803728387, 10803728388, 10803728389, 10803728390, 10803728391, 10803728392, 10803728393, 10803728394, 10803728397, 10803728398, 10803728399, 10803728400, 10803728401, 10803728402, 10803728403, 10803728404, 10803728405, 10803728395, 10803728406, 10803728408, 10803728409, 10803728410, 10803728411, 10803728412, 10803728413, 10803728414, 10803728415, 10803728416, 10803728417, 10803728421, 10803728422, 10803728423, 10803728424, 10803728425, 10803728426, 10803728439, 10803728441, 10803728442, 10803728428, 10803728427, 10803728432, 10803728436, 10803728430, 10803728435, 10803728433, 10803728434, 10803728431, 10803728437, 10803728438, 10803728443, 10803728444, 10803728445, 10803728446, 10803728447, 10803728448, 10803728449, 10803728450, 10803728452, 10803728453, 10803728454, 10803728455, 10803728456, 10803728457, 10803728458, 10803728459, 10803728460, 10803728461, 10803728463, 10803728464, 10803728465, 10803728466, 10803728467, 10803728468, 10803728469, 10803728470, 10803728471, 10803728472, 10803728474, 10803728475, 10803728476, 10803728477, 10803728478, 10803728479, 10803728480, 10803728481, 10803728482, 10803728483, 10803728485, 10803728486, 10803728487, 10803728488, 10803728489, 10803728490, 10803728491, 10803728492, 10803728501, 10803728493, 10803728504, 10803728505, 10803728511, 10803728509, 10803728510, 10803728512, 10803728513, 10803728517, 10803728515, 10803728518, 10803728516, 10803728506, 10803728507, 10803728508, 10803728383], 'projectCategory': 0, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/xzycustiue6zobqvnora4fpa.json', 'aiMetadata': None, 'conditionId': 'AAD_46245951_4038cbc7-0743-464c-af1d-cc8bc280e6bb.171.255', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}

    #_input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 9092, 'projectId': 3441383, 'userId': 390948694, 'userJobId': 1061794486, 'base_url': 'pictures/3/441/3441383', 'photos': [], 'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/1s_jejgcjk-60shpbclxhpjz.json', 'aiMetadata': {'photoIds': [99950443, 119964528, 119964534, 99950444, 99950442, 99950441, 99950445, 99950446, 119964536, 99950447], 'focus': ['everyoneElse'], 'personIds': [25, 9], 'subjects': ['decorations', 'flowers'], 'density': 4}, 'conditionId': 'AAD_3441383_93203b73-c3e2-4543-bf3d-748e3185c920.143.157', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 2}
    #_input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 9092, 'projectId': 3441383, 'userId': 504175649, 'userJobId': 1088111892, 'base_url': 'pictures/3/441/3441383', 'photos': [99950441, 99950442, 99950443, 99950444, 99950445, 99950446, 99950447, 119964528, 119964529, 119964530, 119964531, 119964532, 119964533, 119964534, 119964535, 119964536, 119964537, 119964538, 119964539, 119964540, 119964541, 119964542, 119964543, 119964544, 119964545, 119964546, 119964547, 119964548, 119964549, 119964550, 119964551, 119964552, 119964553, 119964554, 119964555, 119964556, 119964557, 119964558, 119964559, 119964560, 119964561, 119964562, 119964563, 119964564, 119964565, 119964566, 119964567, 119964568, 119964569, 119964570, 119964571, 119964572, 119964573, 119964574, 119964575, 119964576, 119964577, 119964578, 119964579, 119964580, 119964581, 119964582, 119964583, 119964584, 119964585, 119964586, 119964587, 119964588, 119964589, 119964590, 119964591, 119964592, 119964593, 119964594, 119964595, 119964596, 119964597, 119964598, 119964599, 119964600, 119964601, 119964602, 119964603, 119964604, 119964605, 119964606, 119964607, 119964608, 119964609, 119964610, 119964611, 119964612, 119964613, 119964614, 119964615, 119964616, 119964617, 119964618, 119964619, 119964620, 119964621, 119964622, 119964623, 119964624, 119964625, 119964626, 119964628, 119964629, 119964630, 119964631, 119964632, 119964633, 8352432186], 'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/hw5g4j4wfecqqlqyjwaui9_y.json', 'aiMetadata': None, 'conditionId': 'AAD_3441383_4a089a05-47dd-46d3-b42f-e5841f856f57.90.1181', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}


    # _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 9092, 'projectId': 3441383, 'userId': 319454276, 'userJobId': 1023171710, 'base_url': 'pictures/3/441/3441383', 'photos': [99950441, 99950442, 99950443, 99950444, 99950445, 99950446, 99950447, 119964528, 119964529, 119964530, 119964531, 119964532, 119964533, 119964534, 119964535, 119964536, 119964537, 119964538, 119964539, 119964540, 119964541, 119964542, 119964543, 119964544, 119964545, 119964546, 119964547, 119964548, 119964549, 119964550, 119964551, 119964552, 119964553, 119964554, 119964555, 119964556, 119964557, 119964558, 119964559, 119964560, 119964561, 119964562, 119964563, 119964564, 119964565, 119964566, 119964567, 119964568, 119964569, 119964570, 119964571, 119964572, 119964573, 119964574, 119964575, 119964576, 119964577, 119964578, 119964579, 119964580, 119964581, 119964582, 119964583, 119964584, 119964585, 119964586, 119964587, 119964588, 119964589, 119964590, 119964591, 119964592, 119964593, 119964594, 119964595, 119964596, 119964597, 119964598, 119964599, 119964600, 119964601, 119964602, 119964603, 119964604, 119964605, 119964606, 119964607, 119964608, 119964609, 119964610, 119964611, 119964612, 119964613, 119964614, 119964615, 119964616, 119964617, 119964618, 119964619, 119964620, 119964621, 119964622, 119964623, 119964624, 119964625, 119964626, 119964628, 119964629, 119964630, 119964631, 119964632, 119964633, 8352432186], 'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/lbun0-a_reo3oxmwkt5vejof.json', 'aiMetadata': None, 'conditionId': 'AAD_3441383_d1033d5e-7c5e-4e04-b417-69235508bbb2.124.113', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}
    # _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 9092, 'projectId': 35536445, 'userId': 390948694, 'userJobId': 1061794486, 'base_url': 'ptstorage_33://pictures/35/536/35536445/zxtwfklbicxv', 'photos': [8892687183, 8892687267, 8892687359, 8892687357, 8892687150, 8892687159, 8892687058, 8892687170, 8892687083, 8892687248], 'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/8f549nelb0ydbgqxpik3bbu0.json', 'aiMetadata': None, 'conditionId': 'AAD_35536445_b0ad223d-a811-4805-956d-a04cbcb18f04.181.175', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 1}
    #_input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 294318, 'projectId': 44992408, 'userId': 304034801, 'userJobId': 1048365853, 'base_url': 'ptstorage_32://pictures/44/992/44992408/hcw2txvloznymv7rhj', 'photos': [], 'projectCategory': 0, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/ykqfjiw-uem2gqa4pqwzczad.json', 'aiMetadata': {'photoIds': [], 'focus': ['brideAndGroom'], 'personIds': [1, 2], 'subjects': [], 'density': 3}, 'conditionId': 'AAD_44992408_9840a4d1-d718-41cf-b042-31db2a7a93b0.107.319', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}
    #_input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310, 'projectId': 46229129, 'userId': 548224517, 'userJobId': 1069781153, 'base_url': 'ptstorage_32://pictures/46/229/46229129/66cam5evxasipdko4j', 'photos': [], 'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/oktfy0khskwyu1fbn-wqn0km.json', 'aiMetadata': {'photoIds': [], 'focus': ['brideAndGroom'], 'personIds': [1, 13], 'subjects': [], 'density': 3}, 'conditionId': 'AAD_46229129_fd0e84ae-9e89-4f01-9e6e-14c2b32f894e.102.120', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}
    #_input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310, 'projectId': 46229129, 'userId': 570976104, 'userJobId': 1111385993, 'base_url': 'ptstorage_32://pictures/46/229/46229129/66cam5evxasipdko4j', 'photos': [], 'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/_kcpazfkceoznlsokfwtiwv_.json', 'aiMetadata': {'photoIds': [], 'focus': [], 'personIds': [], 'subjects': [], 'density': 3}, 'conditionId': 'AAD_46229129_33583b7f-7d08-4400-bb1c-4a53d3745294.179.125', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}
    #_input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 294318, 'projectId': 44992408, 'userId': 304034801, 'userJobId': 1048365853, 'base_url': 'ptstorage_32://pictures/44/992/44992408/hcw2txvloznymv7rhj', 'photos': [], 'projectCategory': 0, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/ykqfjiw-uem2gqa4pqwzczad.json', 'aiMetadata': {'photoIds': [], 'focus': ['brideAndGroom'], 'personIds': [1, 2], 'subjects': [], 'density': 3}, 'conditionId': 'AAD_44992408_9840a4d1-d718-41cf-b042-31db2a7a93b0.107.319', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}

    # _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 9092, 'projectId': 3441383, 'userId': 570949576, 'userJobId': 1111336211, 'base_url': 'pictures/3/441/3441383', 'photos': [99950439], 'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/so6n5gcrheeyjssetnruep3p.json', 'aiMetadata': None, 'conditionId': 'AAD_3441383_980e1fef-6991-4d0c-b4ac-e8236d8c7125.15.160', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 1}
    # _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 412600, 'projectId': 46669751, 'userId': 571418748, 'userJobId': 1112205013, 'base_url': 'ptstorage_32://pictures/46/669/46669751/wojva6irswydalysu8', 'photos': [10900588461], 'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/rydr02rmkkwutmqnrhumlrwj.json', 'aiMetadata': None, 'conditionId': 'AAD_46669751_3ada6beb-86ea-40fb-8a29-d5b8c68632c4.51.6', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 2}
    #_input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 412600, 'projectId': 46669751, 'userId': 571418748, 'userJobId': 1112205013, 'base_url': 'ptstorage_32://pictures/46/669/46669751/wojva6irswydalysu8', 'photos': [], 'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/isfiexrxae2zhxr8vq_arepm.json', 'aiMetadata': {'photoIds': [], 'focus': [], 'personIds': [], 'subjects': [], 'density': 3}, 'conditionId': 'AAD_46669751_04fe69a6-5e1e-4cd1-841b-33682d77b082.133.145', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}
    #_input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 412600, 'projectId': 46670335, 'userId': 570939077, 'userJobId': 1111317004, 'base_url': 'ptstorage_32://pictures/46/670/46670335/hykjp6q0hzt8zh2m7a', 'photos': [], 'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/euig11nnqecmi6q_estp67dx.json', 'aiMetadata': {'photoIds': [], 'focus': [], 'personIds': [], 'subjects': [], 'density': 3}, 'conditionId': 'AAD_46670335_a67372bb-349e-46f0-a83b-8481de60f7d8.85.64', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 1}
    #_input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 9092, 'projectId': 46655667, 'userId': 394330771, 'userJobId': 1094178825, 'base_url': 'ptstorage_33://pictures/46/655/46655667/v7akooxrl3d5ro3elc', 'photos': [], 'projectCategory': 0, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/e3wqdod8v0yhbvrqgbn9jruk.json', 'aiMetadata': {'photoIds': [10897564846, 10897564845], 'focus': ['brideAndGroom'], 'personIds': [], 'subjects': ['brideGettingReady', 'decorations', 'family'], 'density': 1}, 'conditionId': 'AAD_46655667_7b565c60-661b-447f-bcfd-a1f17c3efc0b.116.47', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 2}

    _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 464800, 'projectId': 46881120,
     'userId': 575306713, 'userJobId': 1119470734,
     'base_url': 'ptstorage_32://pictures/46/881/46881120/6v9uakdy047lyg0lwj', 'photos': [], 'projectCategory': 0,
     'compositionPackageId': -1, 'designInfo': None,
     'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/deaypp34wkqtsrtcrpnurieq.json',
     'aiMetadata': {'photoIds': [], 'focus': ['brideAndGroom'], 'personIds': [7, 1], 'subjects': [], 'density': 3},
     'conditionId': 'AAD_46881120_359e68f5-5162-401b-b713-25899f6196de.54.83', 'timedOut': False,
     'dependencyDeleted': False, 'retryCount': 0}
    _images_path_karmel = fr'C:\Users\karmel\Desktop\AlbumDesigner\dataset\newest_wedding_galleries/{_input_request["projectId"]}/'
    _output_pdf_path_karmel = fr'C:\Users\karmel\Desktop\AlbumDesigner\output/{_input_request["projectId"]}'
    os.makedirs(_output_pdf_path_karmel, exist_ok=True)
    _output_pdf_path = os.path.join(_output_pdf_path_karmel, 'album1.pdf')

    _images_path = _images_path_karmel
    final_album, _message = process_gallery(_input_request)
    gallery_photos_info = _message.content['gallery_photos_info']

    box_id2data = _message.designsInfo['anyPagebox_id2data'] # if 'designsInfo' in _message and 'anyPagebox_id2data' in _message['designsInfo'] else {}
    visualize_album_to_pdf(final_album, _images_path, _output_pdf_path, box_id2data, gallery_photos_info)

    print(final_album)

