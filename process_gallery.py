import os
import pandas as pd

from typing import Dict
from datetime import datetime
import multiprocessing as mp
from collections import defaultdict

from ptinfra import get_logger

from src.request_processing import read_messages

from src.core.photos import update_photos_ranks
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
        design_id = comp['designId']
        design_boxes = comp['boxes']
        placements = placements_by_comp.get(comp_id, [])
        if design_boxes is None:
            design_boxes = [box_id2data.get(placement['boxId']) for placement in placements]
        c.setFont("Helvetica", 10)
        c.drawString(30, page_height - 30, f"Composition ID: {comp_id}, Design ID: {design_id}")
        for placement, box in zip(placements, design_boxes):
            photo_id = placement['photoId']
            all_images_names = os.listdir(images_path)
            all_images_with_this_name = [img for img in all_images_names if img.startswith(f"{photo_id}")]
            img_path = os.path.join(images_path, all_images_with_this_name[0])
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

        available_photos = message.content.get('photos', [])
        df = message.content.get('gallery_photos_info', pd.DataFrame())
        if df.empty:
            logger.error(f"Gallery photos info DataFrame is empty for message {message}")
            message.content['error'] = f"Gallery photos info DataFrame is empty for message {message}"
            raise Exception(f"Gallery photos info DataFrame is empty for message {message}")
        if len(available_photos) != 0:
            df = df[df['image_id'].isin(available_photos)]
            message.content['gallery_photos_info'] = df

        ten_photos = ai_metadata.get('photoIds', [])
        people_ids = ai_metadata.get('personIds', [])
        focus = ai_metadata.get('focus', ['everyoneElse'])
        tags = ai_metadata.get('subjects', ['Wedding dress', 'ceremony', 'bride', 'dancing', 'bride getting ready',
                                            'groom getting ready', 'table setting', 'flowers', 'decorations', 'family',
                                            'baby', 'kids', 'mother', 'father', 'Romance', 'affection', 'Intimacy',
                                            'Happiness', 'Holding hands', 'smiling', 'Hugging', 'Kissing', 'ring',
                                            'veil', 'soft light', 'portrait'])
        density = ai_metadata.get('density', 3)
        
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
        ai_metadata = message.content.get('aiMetadata', {})
        if ai_metadata is not None:
            chosen_photos = ai_metadata.get('photoIds', [])
        else:
            chosen_photos = []
        df = update_photos_ranks(df, chosen_photos)
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
        # for key, value in first_last_pages_data_dict.items():
        #     if first_last_pages_data_dict[key].get('last_images_df',None) is not None or first_last_pages_data_dict[key].get('first_images_df',None) is not None :
        #         if len(first_last_pages_data_dict[key]['last_images_df']) != 0 or len(first_last_pages_data_dict[key]['first_images_df']) != 0:
        #             first_last_pages_data_dict[key]['last_images_df'] = value['last_images_df'].merge(cropped_df, how='inner', on='image_id')
        #             first_last_pages_data_dict[key]['first_images_df'] = value['first_images_df'].merge(cropped_df, how='inner', on='image_id')

        _IMAGE_DF_FIELDS = ("first_images_df", "last_images_df")
        for page_key, page_data in first_last_pages_data_dict.items():
            for field in _IMAGE_DF_FIELDS:
                if field in page_data:
                    if not page_data[field].empty:
                        page_data[field] = page_data[field].merge(cropped_df, how="inner", on="image_id")

        logger.debug('waited for cropping process: {}'.format(datetime.now() - wait_start))

        final_response = assembly_output(album_result, message, df, first_last_pages_data_dict,message.content.get('album_ar',
                                                                                                                   {'anyPage':2})['anyPage'], logger)

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

    # _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 464800, 'projectId': 46881120,
    #  'userId': 575306713, 'userJobId': 1119470734,
    #  'base_url': 'ptstorage_32://pictures/46/881/46881120/6v9uakdy047lyg0lwj', 'photos': [], 'projectCategory': 0,
    #  'compositionPackageId': -1, 'designInfo': None,
    #  'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/deaypp34wkqtsrtcrpnurieq.json',
    #  'aiMetadata': {'photoIds': [], 'focus': ['brideAndGroom'], 'personIds': [7, 1], 'subjects': [], 'density': 3},
    #  'conditionId': 'AAD_46881120_359e68f5-5162-401b-b713-25899f6196de.54.83', 'timedOut': False,
    #  'dependencyDeleted': False, 'retryCount': 0}

    # _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310, 'projectId': 46229129,
    #  'userId': 576349956, 'userJobId': 1121483286,
    #  'base_url': 'ptstorage_32://pictures/46/229/46229129/66cam5evxasipdko4j', 'photos': [], 'projectCategory': 1,
    #  'compositionPackageId': -1, 'designInfo': None,
    #  'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/u7agsvfzvuwzotvuhiadfdjb.json',
    #  'aiMetadata': {'photoIds': [], 'focus': ['brideAndGroom'], 'personIds': [1, 13], 'subjects': [], 'density': 3},
    #  'conditionId': 'AAD_46229129_a47f9c01-976e-4690-a163-c2b1dcf7f37c.77.55', 'timedOut': False,
    #  'dependencyDeleted': False, 'retryCount': 0}


    #  Galleries I have locally


    #_input_request =  {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310, 'projectId': 37141824, 'userId': 576349956, 'userJobId': 1121483286, 'base_url': 'ptstorage_17://pictures/37/141/37141824/dmgb4onqc3hm', 'photos': [], 'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/84fuwtbvy0cwkmw5yhuflcat.json', 'aiMetadata': {'photoIds': [], 'focus': ['brideAndGroom'], 'personIds': [], 'subjects': [], 'density': 3}, 'conditionId': 'AAD_46229128_18628955-182e-465d-bff4-3d4f6bdc120c.151.212', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}
    # _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310,
    #                   'projectId': 38978635, 'userId': 576349956, 'userJobId': 1121483286,
    #                   'base_url': 'ptstorage_18://pictures/38/978/38978635/ldvo72xf7pop', 'photos': [],
    #                   'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None,
    #                   'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/84fuwtbvy0cwkmw5yhuflcat.json',
    #                   'aiMetadata': {'photoIds': [], 'focus': ['brideAndGroom'], 'personIds': [], 'subjects': [],
    #                                  'density': 3},
    #                   'conditionId': 'AAD_46229128_18628955-182e-465d-bff4-3d4f6bdc120c.151.212', 'timedOut': False,
    #                   'dependencyDeleted': False, 'retryCount': 0}

    #_input_request =  {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 464800, 'projectId': 46881120, 'userId': 578739502, 'userJobId': 1126092023, 'base_url': 'ptstorage_32://pictures/46/881/46881120/6v9uakdy047lyg0lwj', 'photos': [], 'projectCategory': 0, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/sesrnulcv0ogvflvc69mjqrd.json', 'aiMetadata': {'photoIds': [], 'focus': ['brideAndGroom'], 'personIds': [], 'subjects': [], 'density': 4}, 'conditionId': 'AAD_46881120_9b2cecf2-1abd-4088-b945-798fdb83a1bb.257.32', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}

    # _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310,
    #                   'projectId': 38122574, 'userId': 576349956, 'userJobId': 1121483286,
    #                   'base_url': 'ptstorage_18://pictures/38/122/38122574/jn4cl65tg2gf', 'photos': [],
    #                   'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None,
    #                   'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/84fuwtbvy0cwkmw5yhuflcat.json',
    #                   'aiMetadata': {'photoIds': [], 'focus': ['brideAndGroom'], 'personIds': [], 'subjects': [],
    #                                  'density': 3},
    #                   'conditionId': 'AAD_46229128_18628955-182e-465d-bff4-3d4f6bdc120c.151.212', 'timedOut': False,
    #                   'dependencyDeleted': False, 'retryCount': 0}

    # _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310,
    #                   'projectId': 37036946, 'userId': 576349956, 'userJobId': 1121483286,
    #                   'base_url': 'ptstorage_18://pictures/37/36/37036946/0j7cgl13spuo', 'photos': [],
    #                   'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None,
    #                   'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/84fuwtbvy0cwkmw5yhuflcat.json',
    #                   'aiMetadata': {'photoIds': [], 'focus': ['brideAndGroom'], 'personIds': [], 'subjects': [],
    #                                  'density': 3},
    #                   'conditionId': 'AAD_46229128_18628955-182e-465d-bff4-3d4f6bdc120c.151.212', 'timedOut': False,
    #                   'dependencyDeleted': False, 'retryCount': 0}
    #
    # _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310,
    #                   'projectId': 32900972, 'userId': 576349956, 'userJobId': 1121483286,
    #                   'base_url': 'ptstorage_18://pictures/32/900/32900972/1teshu0uhg8u', 'photos': [],
    #                   'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None,
    #                   'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/84fuwtbvy0cwkmw5yhuflcat.json',
    #                   'aiMetadata': {'photoIds': [], 'focus': ['brideAndGroom'], 'personIds': [], 'subjects': [],
    #                                  'density': 3},
    #                   'conditionId': 'AAD_46229128_18628955-182e-465d-bff4-3d4f6bdc120c.151.212', 'timedOut': False,
    #                   'dependencyDeleted': False, 'retryCount': 0}

    # _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310, 'projectId': 46229129, 'userId': 548224517, 'userJobId': 1069781153, 'base_url': 'ptstorage_32://pictures/46/229/46229129/66cam5evxasipdko4j', 'photos': [], 'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/queuesdevaigeneratealbumdto/-gfmztztreqf2dwbzag7zmwn.json', 'aiMetadata': {'photoIds': [], 'focus': [], 'personIds': [], 'subjects': [], 'density': 3}, 'conditionId': 'AAD_46229129_fd48ca38-1b12-4eae-9512-08a6cbb12af6.105.22', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}
    # _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310, 'projectId': 46229128, 'userId': 548224517, 'userJobId': 1069781153, 'base_url': 'ptstorage_32://pictures/46/229/46229128/hbltfcpcopx67ta3tc', 'photos': [], 'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/queuesdevaigeneratealbumdto/dgd6mspvukylkgkckicfkmd1.json', 'aiMetadata': {'photoIds': [], 'focus': [], 'personIds': [], 'subjects': [], 'density': 3}, 'conditionId': 'AAD_46229128_46370b95-88f7-415c-9c5a-3ac6f154dd80.147.79', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}
    # _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310, 'projectId': 46245951, 'userId': 548224517, 'userJobId': 1069781153, 'base_url': 'ptstorage_32://pictures/46/245/46245951/ii52fnki40jq0i3xvu', 'photos': [], 'projectCategory': 0, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/queuesdevaigeneratealbumdto/diohv7hckkk0y-4xbyb11fj6.json', 'aiMetadata': {'photoIds': [], 'focus': [], 'personIds': [], 'subjects': [], 'density': 3}, 'conditionId': 'AAD_46245951_13cf3cd5-7d8c-4a3a-a71f-3f27de9c8564.205.83', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}
    #_input_request =  {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310, 'projectId': 46229128, 'userId': 576349956, 'userJobId': 1121483286, 'base_url': 'ptstorage_32://pictures/46/229/46229128/hbltfcpcopx67ta3tc', 'photos': [], 'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/_1ggmdmcjkgqrmzkie0yerot.json', 'aiMetadata': {'photoIds': [], 'focus': ['brideAndGroom'], 'personIds': [4, 11], 'subjects': [], 'density': 3}, 'conditionId': 'AAD_46229128_fead6bde-73b6-4804-84d3-0f2f838a9855.236.98', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}
    _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310, 'projectId': 46227775, 'userId': 548224517, 'userJobId': 1069781153, 'base_url': 'ptstorage_32://pictures/46/227/46227775/wrk9apb6882nqsb96p', 'photos': [10799339727, 10799339736, 10799339746, 10799339763, 10799339772, 10799339781, 10799339790, 10799339799, 10799339808, 10799339817, 10799339825, 10799339830, 10799339836, 10799339845, 10799339855, 10799339864, 10799339871, 10799339878, 10799339887, 10799339896, 10799339902, 10799339915, 10799339924, 10799339931, 10799339939, 10799339947, 10799339957, 10799339964, 10799339977, 10799339987, 10799339999, 10799340005, 10799340014, 10799340023, 10799340032, 10799340043, 10799340052, 10799340064, 10799340086, 10799340097, 10799340114, 10799340126, 10799340136, 10799340245, 10799340258, 10799340271, 10799340283, 10799340296, 10799340310, 10799340320, 10799340334, 10799340345, 10799340355, 10799340366, 10799340376, 10799340387, 10799340396, 10799340408, 10799340420, 10799340433, 10799340445, 10799340454, 10799340462, 10799340471, 10799340479, 10799340488, 10799340497, 10799340504, 10799340515, 10799340523, 10799340531, 10799340540, 10799340553, 10799340561, 10799340570, 10799340576, 10799340583, 10799340587, 10799340594, 10799340600, 10799340609, 10799340618, 10799340626, 10799340635, 10799340644, 10799340655, 10799340662, 10799340670, 10799340677, 10799340685, 10799340693, 10799340702, 10799340711, 10799340719, 10799340726, 10799340735, 10799340741, 10799340749, 10799340755, 10799340759, 10799340763, 10799340773, 10799340781, 10799340791, 10799340801, 10799340811, 10799340820, 10799340829, 10799340840, 10799340849, 10799340858, 10799340867, 10799340875, 10799340885, 10799340894, 10799340904, 10799340914, 10799340922, 10799340928, 10799340935, 10799340941, 10799340950, 10799340960, 10799340971, 10799340980, 10799340994, 10799341006, 10799341013, 10799341020, 10799341028, 10799341036, 10799341043, 10799341050, 10799341060, 10799341067, 10799341073, 10799341080, 10799341088, 10799341096, 10799341104, 10799341112, 10799341121, 10799341146, 10799341154, 10799341163, 10799341171, 10799341180, 10799341188, 10799341198, 10799341207, 10799341214, 10799341223, 10799341232, 10799341239, 10799341247, 10799341254, 10799341262, 10799341268, 10799341277, 10799341285, 10799341292, 10799341302, 10799341309, 10799341332, 10799341341, 10799341352, 10799341360, 10799341368, 10799341378, 10799341387, 10799341397, 10799341406, 10799341412, 10799341421, 10799341431, 10799341440, 10799341449, 10799341459, 10799341468, 10799341478, 10799341489, 10799341499, 10799341509, 10799341518, 10799341527, 10799341536, 10799341545, 10799341553, 10799341563, 10799341574, 10799341582, 10799341591, 10799341599, 10799341609, 10799341618, 10799341627, 10799341636, 10799341645, 10799341655, 10799341664, 10799341672, 10799341682, 10799341692, 10799341701, 10799341710, 10799341717, 10799341729, 10799341736, 10799341745, 10799341755, 10799341767, 10799341776, 10799341786, 10799341797, 10799341806, 10799341815, 10799341824, 10799341832, 10799341841, 10799341853, 10799341863, 10799341873, 10799341882, 10799341892, 10799341901, 10799341910, 10799341919, 10799341926, 10799341935, 10799341945, 10799341953, 10799341964, 10799341973, 10799342001, 10799342009, 10799342018, 10799342024, 10799342033, 10799342041, 10799342050, 10799342059, 10799342067, 10799342077, 10799342085, 10799342094, 10799342105, 10799342115, 10799342120, 10799342128, 10799342138, 10799342148, 10799342157, 10799342165, 10799342174, 10799342183, 10799342192, 10799342201, 10799342211, 10799342219, 10799342231, 10799342239, 10799342247, 10799342257, 10799342267, 10799342275, 10799342283, 10799342292, 10799342302, 10799342311, 10799342320, 10799342329, 10799342337, 10799342346, 10799342356, 10799342363, 10799342372, 10799342379, 10799342388, 10799342397, 10799342406, 10799342415, 10799342424, 10799342434, 10799342444, 10799342454, 10799342464, 10799342473, 10799342482, 10799342490, 10799342500, 10799342507, 10799342517, 10799342527, 10799342537, 10799342546, 10799342564, 10799342573, 10799342582, 10799342590, 10799342597, 10799342609, 10799342617, 10799342625, 10799342635, 10799342645, 10799342652, 10799342663, 10799342672, 10799342680, 10799342689, 10799342698, 10799342707, 10799342717, 10799342725, 10799342734, 10799342742, 10799342751, 10799342757, 10799342768, 10799342778, 10799342786, 10799342795, 10799342804, 10799342813, 10799342822, 10799342832, 10799342841, 10799342850, 10799342859, 10799342868, 10799342877, 10799342885, 10799342893, 10799342555, 10799342900, 10799342910, 10799342920, 10799342930, 10799342936, 10799342950, 10799342957, 10799342966, 10799342975, 10799342984, 10799342992, 10799343001, 10799343010, 10799343019, 10799343024, 10799343032, 10799343042, 10799343050, 10799343058, 10799343067, 10799343077, 10799343086, 10799343094, 10799343102, 10799343109, 10799343114, 10799343121, 10799343127, 10799343135, 10799343143, 10799343152, 10799343161, 10799343169, 10799343176, 10799343184, 10799343192, 10799343200, 10799343208, 10799343217, 10799343226, 10799343235, 10799343244, 10799343253, 10799343262, 10799343271, 10799343280, 10799343288, 10799343296, 10799343306, 10799343315, 10799343322, 10799343333, 10799343343, 10799343351, 10799343360, 10799343368, 10799343378, 10799343387, 10799343396, 10799343405, 10799343414, 10799343423, 10799343430, 10799343440, 10799343448, 10799343459, 10799343467, 10799343475, 10799343484, 10799343493, 10799343502, 10799343511, 10799343520, 10799343529, 10799343537, 10799343546, 10799343555, 10799343564, 10799343573, 10799343582, 10799343591, 10799343599, 10799343608, 10799343616, 10799343625, 10799343636, 10799343647, 10799343655, 10799343662, 10799343669, 10799343680, 10799343692, 10799343699, 10799343708, 10799343718, 10799343727, 10799343736, 10799343744, 10799343753, 10799343764, 10799343773, 10799343782, 10799343792, 10799343800, 10799343808, 10799343818, 10799343828, 10799343837, 10799343846, 10799343853, 10799343860, 10799343868, 10799343877, 10799343886, 10799343894, 10799343903, 10799343912, 10799343921, 10799343928, 10799343938, 10799343947, 10799343960, 10799343969, 10799343979, 10799343987, 10799343997, 10799344006, 10799344017, 10799344025, 10799344038, 10799344047, 10799344054, 10799344065, 10799344076, 10799344082, 10799344091, 10799344100, 10799344109, 10799344118, 10799344127, 10799344133, 10799344142, 10799344151, 10799344160, 10799344169, 10799344178, 10799344186, 10799344195, 10799344205, 10799344213, 10799344221, 10799344230, 10799344240, 10799344249, 10799344257, 10799344266, 10799344275, 10799344284, 10799344293, 10799344302, 10799344310, 10799344319, 10799344328, 10799344337, 10799344345, 10799344361, 10799344368, 10799344373, 10799344381, 10799344386, 10799344394, 10799344403, 10799344411, 10799344420, 10799344429, 10799344438, 10799344445, 10799344455, 10799344464, 10799344473, 10799344482, 10799344491, 10799344502, 10799344508, 10799344514, 10799344521, 10799344530, 10799344539, 10799344547, 10799344556, 10799344562, 10799344568, 10799344573, 10799344580, 10799344589, 10799344596, 10799344604, 10799344612, 10799344619, 10799344628, 10799344637, 10799344646, 10799344654, 10799344663, 10799344672, 10799344682, 10799344354, 10799344690, 10799344699, 10799344708, 10799344717, 10799344726, 10799344737, 10799344745, 10799344754, 10799344764, 10799344774, 10799344781, 10799344790, 10799344799, 10799344809, 10799344817, 10799344823, 10799344833, 10799344845, 10799344853, 10799344862, 10799344869, 10799344878, 10799344887, 10799344896, 10799344905, 10799344913, 10799344923, 10799344932, 10799344942, 10799344951, 10799344960, 10799344971, 10799344980, 10799344988, 10799344999, 10799345008, 10799345017, 10799345025, 10799345035, 10799345044, 10799345052, 10799345061, 10799345069, 10799345078, 10799345086, 10799345094, 10799345105, 10799345115, 10799345123, 10799345133, 10799345143, 10799345152, 10799345162, 10799345170, 10799345180, 10799345188, 10799345197, 10799345206, 10799345213, 10799345222, 10799345232, 10799345241, 10799345248, 10799345258, 10799345268, 10799345276, 10799345284, 10799345291, 10799345300, 10799345310, 10799345319, 10799345328, 10799345338, 10799345350, 10799345358, 10799345363, 10799345367, 10799345370, 10799345379, 10799345385, 10799345396, 10799345404, 10799345412, 10799345420, 10799345429, 10799345437, 10799345446, 10799345455, 10799345464, 10799345473, 10799345482, 10799345491, 10799345499, 10799345507, 10799345516, 10799345525, 10799345533, 10799345542, 10799345551, 10799345560, 10799345570, 10799345579, 10799345589, 10799345598, 10799345607, 10799345616, 10799345624, 10799345633, 10799345642, 10799345653, 10799345662, 10799345671, 10799345680, 10799345689, 10799345698, 10799345708, 10799345717, 10799345725, 10799345733, 10799345742, 10799345750, 10799345757, 10799345766, 10799345775, 10799345784, 10799345793, 10799345801, 10799345810, 10799345819, 10799345830, 10799345838, 10799345847, 10799345856, 10799345865, 10799345874, 10799345883, 10799345893, 10799345903, 10799345912, 10799345921, 10799345930, 10799345939, 10799345948, 10799345957, 10799345966, 10799345975, 10799345984, 10799345993, 10799346002, 10799346011, 10799346021, 10799346030, 10799346039, 10799346047, 10799346056, 10799346071, 10799346080, 10799346089, 10799346098, 10799346116, 10799346123, 10799346132, 10799346140, 10799346147, 10799346154, 10799346157, 10799346166, 10799346175, 10799346184, 10799346191, 10799346202, 10799346209, 10799346217, 10799346226, 10799346234, 10799346242, 10799346251, 10799346257, 10799346266, 10799346276, 10799346281, 10799346284, 10799346292, 10799346300, 10799346305, 10799346313, 10799346321, 10799346328, 10799346335, 10799346343, 10799346350, 10799346357, 10799346364, 10799346371, 10799346378, 10799346385, 10799346391, 10799346398, 10799346403, 10799346410, 10799346416, 10799346423, 10799346430, 10799346437, 10799346445, 10799346451, 10799346458, 10799346465, 10799346472, 10799346480, 10799346488, 10799346495, 10799346502, 10799346509, 10799346517, 10799346524, 10799346530, 10799346538, 10799346545, 10799346552, 10799346558, 10799346565, 10799346572, 10799346579, 10799346586, 10799346593, 10799346600, 10799346607, 10799346614, 10799346621, 10799346628, 10799346635, 10799346641, 10799346650, 10799346107, 10799346725, 10799346732, 10799346739, 10799346746, 10799346753, 10799346760, 10799346767, 10799346774, 10799346781, 10799346788, 10799346795, 10799346802, 10799346718, 10799346810, 10799346817, 10799346824, 10799346829, 10799346838, 10799346847, 10799346852, 10799346865, 10799346874, 10799346882, 10799346888, 10799346894, 10799346901, 10799346908, 10799346916, 10799346923, 10799346930, 10799346937, 10799346945, 10799346951, 10799346960, 10799346967, 10799346973, 10799346980, 10799346987, 10799346994, 10799347001, 10799347008, 10799347016, 10799347022, 10799347029, 10799347036, 10799338699, 10799339718, 10799338708, 10799339700, 10799338717, 10799339709, 10799338726, 10799338736, 10799338744, 10799338754, 10799338763, 10799338771, 10799338780, 10799338791, 10799338798, 10799338807, 10799338818, 10799338828, 10799338836, 10799338840, 10799338852, 10799338860, 10799338869, 10799338877, 10799338886, 10799338895, 10799338904, 10799338913, 10799338922, 10799338931, 10799338940, 10799338949, 10799338959, 10799338967, 10799338975, 10799338984, 10799338993, 10799338999, 10799339008, 10799339017, 10799339026, 10799339035, 10799339043, 10799339055, 10799339064, 10799339071, 10799339081, 10799339089, 10799339098, 10799339107, 10799339115, 10799339124, 10799339135, 10799339146, 10799339156, 10799339166, 10799339177, 10799339184, 10799339193, 10799339202, 10799339211, 10799339220, 10799339229, 10799339239, 10799339248, 10799339257, 10799339266, 10799339276, 10799339284, 10799339293, 10799339302, 10799339311, 10799339319, 10799339327, 10799339337, 10799339348, 10799339356, 10799339363, 10799339369, 10799339378, 10799339386, 10799339395, 10799339403, 10799339430, 10799339439, 10799339448, 10799339457, 10799339466, 10799339475, 10799339484, 10799339493, 10799339503, 10799339511, 10799339519, 10799339530, 10799339537, 10799339544, 10799339550, 10799339559, 10799339565, 10799339574, 10799339584, 10799339593, 10799339602, 10799339611, 10799339621, 10799339630, 10799339639, 10799339647, 10799339657, 10799339666, 10799339675, 10799339683, 10799339693, 10799346663, 10799346671, 10799346678, 10799346685, 10799346692, 10799346699, 10799346706, 10799346711, 10799346657], 'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/queuesdevaigeneratealbumdto/kffbjcqynkasd20gadrs5pwq.json', 'aiMetadata': {'photoIds': [], 'focus': [], 'personIds': [], 'subjects': [], 'density': 3}, 'conditionId': 'AAD_46227775_f594122c-1bc9-4ed0-b18d-5744b37a160c.85.17', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}

    # _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310,
    #                   'projectId': 36048323, 'userId': 576349956, 'userJobId': 1121483286,
    #                   'base_url': 'ptstorage_16://pictures/36/48/36048323/vxgigtmzwxsv', 'photos': [],
    #                   'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None,
    #                   'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/84fuwtbvy0cwkmw5yhuflcat.json',
    #                   'aiMetadata': {'photoIds': [], 'focus': ['brideAndGroom'], 'personIds': [], 'subjects': [],
    #                                  'density': 3},
    #                   'conditionId': 'AAD_46229128_18628955-182e-465d-bff4-3d4f6bdc120c.151.212', 'timedOut': False,
    #                   'dependencyDeleted': False, 'retryCount': 0}
    #

    #_input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310, 'projectId': 46227780, 'userId': 548224517, 'userJobId': 1069781153, 'base_url': 'ptstorage_32://pictures/46/227/46227780/njev8ankt8x9b7ynth', 'photos': [], 'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/bxqwa2at_eq6resxfgqctxw7.json', 'aiMetadata': {'photoIds': [], 'focus': ['brideAndGroom'], 'personIds': [7, 1], 'subjects': [], 'density': 3}, 'conditionId': 'AAD_46227780_c69bf3c0-43ea-4b96-be82-a7ae256281f1.100.10', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}

    # _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310, 'projectId': 46229128,
    #  'userId': 548224517, 'userJobId': 1069781153,
    #  'base_url': 'ptstorage_32://pictures/46/229/46229128/hbltfcpcopx67ta3tc', 'photos': [], 'projectCategory': 1,
    #  'compositionPackageId': -1, 'designInfo': None,
    #  'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/aqa34npt60-jil9d9ahgrljv.json',
    #  'aiMetadata': {'photoIds': [], 'focus': ['brideAndGroom'], 'personIds': [], 'subjects': [], 'density': 3},
    #  'conditionId': 'AAD_46229128_c69bf3c0-43ea-4b96-be82-a7ae256281f1.98.148', 'timedOut': False,
    #  'dependencyDeleted': False, 'retryCount': 0}

    #_input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310, 'projectId': 46227780, 'userId': 576349956, 'userJobId': 1121483286, 'base_url': 'ptstorage_32://pictures/46/227/46227780/njev8ankt8x9b7ynth', 'photos': [], 'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/43rqeed08u2xtk_vwexyl8ut.json', 'aiMetadata': {'photoIds': [], 'focus': ['brideAndGroom'], 'personIds': [], 'subjects': [], 'density': 3}, 'conditionId': 'AAD_46227780_e049886f-504e-42a9-8a2c-40d13b404436.127.172', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}

    # Debug with Plotting
    _images_path_karmel = fr'C:\Users\ZivRotman\PycharmProjects\logAnalysis\galleries_pbs2/{_input_request["projectId"]}/'
    _output_pdf_path_karmel = fr'C:\temp'
    os.makedirs(_output_pdf_path_karmel, exist_ok=True)
    _output_pdf_path = os.path.join(_output_pdf_path_karmel, 'album1.pdf')

    _images_path = _images_path_karmel
    final_album, _message = process_gallery(_input_request)
    gallery_photos_info = _message.content['gallery_photos_info']

    box_id2data = _message.designsInfo['anyPagebox_id2data'] # if 'designsInfo' in _message and 'anyPagebox_id2data' in _message['designsInfo'] else {}
    visualize_album_to_pdf(final_album, _images_path, _output_pdf_path, box_id2data, gallery_photos_info)

    print(final_album)



