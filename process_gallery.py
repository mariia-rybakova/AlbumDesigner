import os
import pandas as pd
import traceback
import sys


from typing import Dict
from datetime import datetime
import multiprocessing as mp
from collections import defaultdict

from ptinfra import get_logger,intialize
from ptinfra.config import get_variable
from pymongo import MongoClient
from qdrant_client import QdrantClient

from src.request_processing import read_messages

from src.core.photos import update_photos_ranks
from src.smart_cropping import process_crop_images
from src.core.key_pages import generate_first_last_pages
from utils.time_processing import generate_time_clusters
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
            if photo_id is None:
                continue
            all_images_names = os.listdir(images_path)
            all_images_with_this_name = [img for img in all_images_names if img.startswith(f"{photo_id}")]
            if len(all_images_with_this_name) == 0:
                continue
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
        ai_photos_selected, spreads_dict, errors = ai_selection(df, ten_photos, people_ids, focus, tags, is_wedding, density,is_artificial_time,
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


def process_message(message, logger):
    # check if its single message or list
    params = [0.01, 100, 1000, 100, 300, 12]
    logger.debug("Params for this Gallery are: {}".format(params))

    df = message.content.get('gallery_photos_info', pd.DataFrame())
    bride_and_groom_df = message.content.get('bride and groom', pd.DataFrame())
    df_serializable = pd.concat([df.copy(), bride_and_groom_df])  # Make a copy to avoid modifying original
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

        # generate time clusters for the gallery photos
        sorted_df = generate_time_clusters(message, sorted_df, logger)

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
        tb = traceback.extract_tb(e.__traceback__)
        filename, lineno, func, text = tb[-1]
        logger.error(f"Unexpected error in message processing: {e}. Exception in function: {func}, line {lineno}, file {filename}.")
        raise Exception(f"Unexpected error in message processing: {e}. Exception in function: {func}, line {lineno}, file {filename}.")

    return final_response, message


def process_gallery(input_request):
    message = Message(input_request, id=1)
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
    #_input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310, 'projectId': 46227780, 'userId': 548224517, 'userJobId': 1069781153, 'base_url': 'ptstorage_32://pictures/46/227/46227780/njev8ankt8x9b7ynth', 'photos': [10799337783, 10799337786, 10799337789, 10799337792, 10799337796, 10799337798, 10799337802, 10799337805, 10799337808, 10799337811, 10799337817, 10799337820, 10799337824, 10799337828, 10799337832, 10799337836, 10799337840, 10799337845, 10799337849, 10799337853, 10799337857, 10799337861, 10799337865, 10799337869, 10799337873, 10799337876, 10799337878, 10799337882, 10799337886, 10799337889, 10799337893, 10799337897, 10799337901, 10799337905, 10799337909, 10799337913, 10799337918, 10799337921, 10799337926, 10799337929, 10799337933, 10799337938, 10799337942, 10799337946, 10799337950, 10799337954, 10799337958, 10799337962, 10799337966, 10799337970, 10799337975, 10799337980, 10799337985, 10799337990, 10799337995, 10799338000, 10799338005, 10799338010, 10799338014, 10799338020, 10799338025, 10799338031, 10799338036, 10799338041, 10799338046, 10799338051, 10799338056, 10799338061, 10799338066, 10799338071, 10799338075, 10799338080, 10799338085, 10799338089, 10799338094, 10799338099, 10799338104, 10799338110, 10799338120, 10799338125, 10799338130, 10799338135, 10799338139, 10799338144, 10799338149, 10799338154, 10799338159, 10799338164, 10799338169, 10799338173, 10799338177, 10799338182, 10799338187, 10799338192, 10799338197, 10799338202, 10799338207, 10799338212, 10799338217, 10799338222, 10799338227, 10799338231, 10799338236, 10799338241, 10799338247, 10799338253, 10799338259, 10799338265, 10799338271, 10799338277, 10799338283, 10799338289, 10799338295, 10799338301, 10799338307, 10799338314, 10799338321, 10799338328, 10799338334, 10799338340, 10799338345, 10799338350, 10799338358, 10799338366, 10799338372, 10799338379, 10799338386, 10799338394, 10799338402, 10799338409, 10799338416, 10799338423, 10799338429, 10799338436, 10799338443, 10799338451, 10799338460, 10799338468, 10799338475, 10799338484, 10799338493, 10799338501, 10799338509, 10799338518, 10799338530, 10799338539, 10799338547, 10799338555, 10799338562, 10799338570, 10799338578, 10799338585, 10799338593, 10799338601, 10799338605, 10799338613, 10799338621, 10799338629, 10799338637, 10799338645, 10799338653, 10799338662, 10799338670, 10799338677, 10799338683, 10799338691, 10799338700, 10799338709, 10799338718, 10799338727, 10799338735, 10799338745, 10799338753, 10799338762, 10799338772, 10799338781, 10799338789, 10799338800, 10799338809, 10799338817, 10799338825, 10799338835, 10799338844, 10799338855, 10799338867, 10799338878, 10799338887, 10799338896, 10799338905, 10799338914, 10799338923, 10799338932, 10799338941, 10799338951, 10799338960, 10799338969, 10799338978, 10799338987, 10799338996, 10799339001, 10799339010, 10799339019, 10799339032, 10799339041, 10799339050, 10799339060, 10799339069, 10799339078, 10799339088, 10799339096, 10799339105, 10799339114, 10799339123, 10799339132, 10799339142, 10799339152, 10799339161, 10799339171, 10799339180, 10799339189, 10799339198, 10799339209, 10799339218, 10799339227, 10799339237, 10799339246, 10799339255, 10799339264, 10799339273, 10799339281, 10799339291, 10799339300, 10799339309, 10799339317, 10799339326, 10799339335, 10799339344, 10799339354, 10799339361, 10799339370, 10799339379, 10799339388, 10799339396, 10799339407, 10799339418, 10799346928, 10799346935, 10799346942, 10799346949, 10799346956, 10799346962, 10799346969, 10799346976, 10799346983, 10799346990, 10799346997, 10799347004, 10799347012, 10799347018, 10799347026, 10799347033, 10799347039, 10799347045, 10799347051, 10799347057, 10799347063, 10799347069, 10799347075, 10799347080, 10799347086, 10799347092, 10799347098, 10799347104, 10799347110, 10799347116, 10799347122, 10799347128, 10799347134, 10799347140, 10799347145, 10799347151, 10799347157, 10799347163, 10799339426, 10799339435, 10799339443, 10799339453, 10799339462, 10799339471, 10799339479, 10799339488, 10799339497, 10799339505, 10799339516, 10799339526, 10799339534, 10799339541, 10799339548, 10799339556, 10799339562, 10799339568, 10799339577, 10799339586, 10799339595, 10799339604, 10799339612, 10799339620, 10799339629, 10799339637, 10799339646, 10799339654, 10799339662, 10799339682, 10799339689, 10799339706, 10799339716, 10799339724, 10799339734, 10799339744, 10799339749, 10799339754, 10799339761, 10799339771, 10799339780, 10799339788, 10799339797, 10799339807, 10799339816, 10799339826, 10799339840, 10799339851, 10799339861, 10799339870, 10799339881, 10799339890, 10799339899, 10799339907, 10799339917, 10799339926, 10799339935, 10799339943, 10799339950, 10799339960, 10799339969, 10799339976, 10799339985, 10799339991, 10799340000, 10799340009, 10799340018, 10799340028, 10799340036, 10799340045, 10799340054, 10799340060, 10799340069, 10799340076, 10799340078, 10799340081, 10799340087, 10799340095, 10799340101, 10799340106, 10799340113, 10799340120, 10799340128, 10799340138, 10799340144, 10799340149, 10799340154, 10799340158, 10799340164, 10799340169, 10799340176, 10799340181, 10799340186, 10799340190, 10799340194, 10799340199, 10799340204, 10799340210, 10799340214, 10799340218, 10799340223, 10799340227, 10799340232, 10799340237, 10799340244, 10799340252, 10799340259, 10799340266, 10799340274, 10799340281, 10799340288, 10799340295, 10799340302, 10799340309, 10799340318, 10799340327, 10799340333, 10799340341, 10799340348, 10799340357, 10799340364, 10799340373, 10799340383, 10799340390, 10799340398, 10799340407, 10799340413, 10799340421, 10799340430, 10799340437, 10799340444, 10799340453, 10799340463, 10799340473, 10799340483, 10799340492, 10799340501, 10799340510, 10799340517, 10799340525, 10799340533, 10799340542, 10799340549, 10799340556, 10799340564, 10799340573, 10799340580, 10799340589, 10799340597, 10799340606, 10799340614, 10799340622, 10799340630, 10799340640, 10799340650, 10799340660, 10799340667, 10799340675, 10799340682, 10799340690, 10799340699, 10799340709, 10799340717, 10799340727, 10799340732, 10799340739, 10799340747, 10799340765, 10799340771, 10799340778, 10799340785, 10799340794, 10799340802, 10799340810, 10799340819, 10799340828, 10799340838, 10799340847, 10799340856, 10799340865, 10799340874, 10799340883, 10799340892, 10799340901, 10799340910, 10799340919, 10799340925, 10799340936, 10799340948, 10799340956, 10799340966, 10799340974, 10799340983, 10799340989, 10799340996, 10799341003, 10799341011, 10799341019, 10799341027, 10799341035, 10799341044, 10799341052, 10799341059, 10799341066, 10799341075, 10799341083, 10799341090, 10799341098, 10799341106, 10799341114, 10799341124, 10799341131, 10799341136, 10799341141, 10799341147, 10799341155, 10799341166, 10799341174, 10799341183, 10799341193, 10799341202, 10799341211, 10799341218, 10799341226, 10799341233, 10799341241, 10799341246, 10799341252, 10799341259, 10799341266, 10799341273, 10799341280, 10799341286, 10799341295, 10799341303, 10799341311, 10799341318, 10799341326, 10799341335, 10799341343, 10799341351, 10799341358, 10799341367, 10799341375, 10799341384, 10799341393, 10799341402, 10799341409, 10799341418, 10799341428, 10799341435, 10799341444, 10799341456, 10799341461, 10799341469, 10799341477, 10799341485, 10799341494, 10799341502, 10799341512, 10799341520, 10799341529, 10799341538, 10799341546, 10799341556, 10799341565, 10799341573, 10799341580, 10799341589, 10799341598, 10799341607, 10799341615, 10799341626, 10799341635, 10799341644, 10799341653, 10799341662, 10799341670, 10799341677, 10799341686, 10799341695, 10799341704, 10799341713, 10799341725, 10799341734, 10799341743, 10799341752, 10799341760, 10799341769, 10799341777, 10799341785, 10799341795, 10799341802, 10799341812, 10799341820, 10799341829, 10799341846, 10799341856, 10799341865, 10799341875, 10799341883, 10799341893, 10799341903, 10799341911, 10799341917, 10799341927, 10799341936, 10799341944, 10799341954, 10799341962, 10799341972, 10799341981, 10799341982, 10799341983, 10799341984, 10799341990, 10799341996, 10799342006, 10799342014, 10799342021, 10799342030, 10799342039, 10799342049, 10799342057, 10799342065, 10799342073, 10799342083, 10799342092, 10799342101, 10799342108, 10799342117, 10799342125, 10799342134, 10799342143, 10799342153, 10799342162, 10799342171, 10799342180, 10799342189, 10799342198, 10799342206, 10799342215, 10799342223, 10799342232, 10799342241, 10799342250, 10799342259, 10799342269, 10799342277, 10799342285, 10799342295, 10799342303, 10799342313, 10799342321, 10799342331, 10799342340, 10799342349, 10799342358, 10799342367, 10799342376, 10799342385, 10799342394, 10799342402, 10799342410, 10799342420, 10799342429, 10799342439, 10799342448, 10799342457, 10799342466, 10799342476, 10799342485, 10799342492, 10799342501, 10799342511, 10799342521, 10799342530, 10799342541, 10799342550, 10799342557, 10799342565, 10799342574, 10799342583, 10799342601, 10799342611, 10799342620, 10799342629, 10799342634, 10799342642, 10799342651, 10799342661, 10799342670, 10799342678, 10799342687, 10799342696, 10799342704, 10799342714, 10799342722, 10799342732, 10799342743, 10799342749, 10799342759, 10799342769, 10799342777, 10799342787, 10799342796, 10799342805, 10799342815, 10799342823, 10799342831, 10799342840, 10799342849, 10799342858, 10799342867, 10799342876, 10799342886, 10799342895, 10799342905, 10799342915, 10799342923, 10799342932, 10799342941, 10799342949, 10799342959, 10799342968, 10799342978, 10799342985, 10799342995, 10799343004, 10799343013, 10799343022, 10799343026, 10799343036, 10799343049, 10799343059, 10799343068, 10799343075, 10799343084, 10799343093, 10799343104, 10799343115, 10799343124, 10799343133, 10799343144, 10799343151, 10799343159, 10799343170, 10799343180, 10799343188, 10799343206, 10799343215, 10799343223, 10799343232, 10799343241, 10799343249, 10799343258, 10799343267, 10799343276, 10799343286, 10799343294, 10799343303, 10799343312, 10799343321, 10799343330, 10799343339, 10799343348, 10799343357, 10799343366, 10799343375, 10799343384, 10799343393, 10799343402, 10799343412, 10799343420, 10799343429, 10799343438, 10799343447, 10799343456, 10799343466, 10799343478, 10799343487, 10799343496, 10799343505, 10799343513, 10799343524, 10799343533, 10799343542, 10799343550, 10799343558, 10799343568, 10799343577, 10799343585, 10799343594, 10799343603, 10799343611, 10799343620, 10799343629, 10799343637, 10799343644, 10799343652, 10799343661, 10799343671, 10799343678, 10799343687, 10799343696, 10799343705, 10799343714, 10799343723, 10799343732, 10799343740, 10799343747, 10799343756, 10799343767, 10799343778, 10799343786, 10799343793, 10799343802, 10799343811, 10799343820, 10799343829, 10799343838, 10799343847, 10799343855, 10799343863, 10799343871, 10799343880, 10799343889, 10799343898, 10799343907, 10799343916, 10799343925, 10799343932, 10799343940, 10799343950, 10799343958, 10799343968, 10799343976, 10799343985, 10799343994, 10799344003, 10799344012, 10799344022, 10799344031, 10799344039, 10799344048, 10799344058, 10799344064, 10799344073, 10799344087, 10799344094, 10799344103, 10799344112, 10799344121, 10799344131, 10799344140, 10799344149, 10799344158, 10799344167, 10799344176, 10799344185, 10799344194, 10799344204, 10799344214, 10799344223, 10799344232, 10799344241, 10799344250, 10799344259, 10799344267, 10799344276, 10799344285, 10799344297, 10799344307, 10799344315, 10799344323, 10799344330, 10799344340, 10799344349, 10799344357, 10799344365, 10799344371, 10799344376, 10799344383, 10799344388, 10799344396, 10799344405, 10799344412, 10799344419, 10799344428, 10799344437, 10799344447, 10799344456, 10799344465, 10799344474, 10799344484, 10799344495, 10799344506, 10799344511, 10799344517, 10799344526, 10799344535, 10799344543, 10799344553, 10799344559, 10799344565, 10799344570, 10799344577, 10799344585, 10799344592, 10799344603, 10799344613, 10799344621, 10799344629, 10799344636, 10799344644, 10799344653, 10799344662, 10799344671, 10799344680, 10799344688, 10799344697, 10799344706, 10799344714, 10799344723, 10799344732, 10799344741, 10799344749, 10799344758, 10799344768, 10799344778, 10799344788, 10799344795, 10799344807, 10799344814, 10799344829, 10799344834, 10799344842, 10799344851, 10799344860, 10799344875, 10799344884, 10799344893, 10799344901, 10799344910, 10799344919, 10799344928, 10799344937, 10799344946, 10799344955, 10799344965, 10799344974, 10799344983, 10799344992, 10799345001, 10799345011, 10799345021, 10799345031, 10799345040, 10799345048, 10799345057, 10799345065, 10799345076, 10799345088, 10799345097, 10799345106, 10799345113, 10799345122, 10799345131, 10799345140, 10799345149, 10799345156, 10799345166, 10799345174, 10799345182, 10799345191, 10799345200, 10799345207, 10799345217, 10799345227, 10799345236, 10799345245, 10799345254, 10799345263, 10799345272, 10799345281, 10799345289, 10799345297, 10799345304, 10799345312, 10799345321, 10799345330, 10799345336, 10799345345, 10799345351, 10799345378, 10799345388, 10799345395, 10799345406, 10799345413, 10799345421, 10799345430, 10799345439, 10799345448, 10799345458, 10799345466, 10799345475, 10799345483, 10799345494, 10799345502, 10799345511, 10799345520, 10799345529, 10799345538, 10799345549, 10799345558, 10799345566, 10799345575, 10799345583, 10799345593, 10799345602, 10799345611, 10799345619, 10799345629, 10799345638, 10799345647, 10799345656, 10799345665, 10799345674, 10799345683, 10799345693, 10799345702, 10799345710, 10799345719, 10799345727, 10799345737, 10799345746, 10799345755, 10799345764, 10799345772, 10799345781, 10799345799, 10799345808, 10799345816, 10799345826, 10799345835, 10799345844, 10799345854, 10799345863, 10799345873, 10799345882, 10799345891, 10799345900, 10799345911, 10799345920, 10799345929, 10799345938, 10799345947, 10799345956, 10799345965, 10799345973, 10799345982, 10799345990, 10799346000, 10799346009, 10799346017, 10799346027, 10799346036, 10799346046, 10799346055, 10799346064, 10799346072, 10799346082, 10799346092, 10799346100, 10799346109, 10799346117, 10799346125, 10799346133, 10799346141, 10799346148, 10799346155, 10799346159, 10799346165, 10799346173, 10799346181, 10799346189, 10799346196, 10799346204, 10799346212, 10799346222, 10799346229, 10799346237, 10799346245, 10799346253, 10799346261, 10799346269, 10799346278, 10799346291, 10799346298, 10799346309, 10799346316, 10799346324, 10799346332, 10799346338, 10799346345, 10799346352, 10799346359, 10799346365, 10799346373, 10799346383, 10799346388, 10799346395, 10799346405, 10799346408, 10799345790, 10799346418, 10799346424, 10799346431, 10799346436, 10799346443, 10799346450, 10799346457, 10799346463, 10799346471, 10799346478, 10799346485, 10799346491, 10799346498, 10799346505, 10799346511, 10799346520, 10799346528, 10799346535, 10799346542, 10799346550, 10799346564, 10799346571, 10799346578, 10799346585, 10799346592, 10799346599, 10799346605, 10799346612, 10799346619, 10799346626, 10799346633, 10799346640, 10799346647, 10799346654, 10799346664, 10799346670, 10799346677, 10799346684, 10799346691, 10799346698, 10799346705, 10799346713, 10799346719, 10799346726, 10799346733, 10799346740, 10799346747, 10799346754, 10799346761, 10799346768, 10799346775, 10799346782, 10799346789, 10799346796, 10799346803, 10799346809, 10799346816, 10799346823, 10799346830, 10799346834, 10799346843, 10799346850, 10799346858, 10799346862, 10799346872, 10799346880, 10799346886, 10799346893, 10799346900, 10799346907, 10799346914, 10799346921], 'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/queuesdevaigeneratealbumdto/a0hspykt0kuq8vgkyownfa35.json', 'aiMetadata': {'photoIds': [], 'focus': [], 'personIds': [], 'subjects': [], 'density': 3}, 'conditionId': 'AAD_46227780_e6170478-0645-4166-b9ec-0a10c28acbee.191.113', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}

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

    #_input_request ={'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310, 'projectId': 46245951, 'userId': 548224517, 'userJobId': 1069781153, 'base_url': 'ptstorage_32://pictures/46/245/46245951/ii52fnki40jq0i3xvu', 'photos': [10803725943, 10803725940, 10803725941, 10803725942, 10803725895, 10803725897, 10803725902, 10803725898, 10803725899, 10803725901, 10803725903, 10803725904, 10803725915, 10803725893, 10803725910, 10803725911, 10803725905, 10803725900, 10803725906, 10803725931, 10803725932, 10803725907, 10803725933, 10803725912, 10803725913, 10803725914, 10803725908, 10803725909, 10803725916, 10803725917, 10803725918, 10803725919, 10803725926, 10803725930, 10803725920, 10803725921, 10803725922, 10803725937, 10803725923, 10803725924, 10803725925, 10803725927, 10803725928, 10803725929, 10803725935, 10803725936, 10803725938, 10803725939, 10803725934, 10803725896, 10803725894, 10803726026, 10803726015, 10803726037, 10803725945, 10803725951, 10803725952, 10803725953, 10803725944, 10803725954, 10803725955, 10803725956, 10803725971, 10803725960, 10803725957, 10803725958, 10803725959, 10803725961, 10803726004, 10803725982, 10803725962, 10803725964, 10803725963, 10803725993, 10803725965, 10803725966, 10803725967, 10803725968, 10803725969, 10803725970, 10803725972, 10803725973, 10803725974, 10803725975, 10803725976, 10803725977, 10803725978, 10803725979, 10803725980, 10803725981, 10803725983, 10803725984, 10803725985, 10803725986, 10803725987, 10803725988, 10803725989, 10803725990, 10803725991, 10803725992, 10803725994, 10803725995, 10803725996, 10803725997, 10803725998, 10803725999, 10803726000, 10803726001, 10803726002, 10803726003, 10803726005, 10803726006, 10803726007, 10803726008, 10803726009, 10803726010, 10803726011, 10803726012, 10803726013, 10803726014, 10803726016, 10803726017, 10803726018, 10803726019, 10803726020, 10803726021, 10803726022, 10803726023, 10803726024, 10803726025, 10803726027, 10803726028, 10803726029, 10803726030, 10803726031, 10803726032, 10803726033, 10803726034, 10803726035, 10803726036, 10803726038, 10803726039, 10803726040, 10803726041, 10803726042, 10803726044, 10803726043, 10803726045, 10803726046, 10803726047, 10803725946, 10803725947, 10803725948, 10803725949, 10803725950, 10803726051, 10803726113, 10803726124, 10803726135, 10803726146, 10803726157, 10803726168, 10803726179, 10803726190, 10803726052, 10803726063, 10803726074, 10803726085, 10803726096, 10803726107, 10803726109, 10803726110, 10803726111, 10803726112, 10803726114, 10803726115, 10803726116, 10803726117, 10803726118, 10803726119, 10803726120, 10803726121, 10803726122, 10803726123, 10803726125, 10803726126, 10803726127, 10803726128, 10803726129, 10803726130, 10803726131, 10803726132, 10803726133, 10803726134, 10803726136, 10803726137, 10803726138, 10803726139, 10803726140, 10803726141, 10803726142, 10803726143, 10803726144, 10803726145, 10803726147, 10803726148, 10803726149, 10803726150, 10803726151, 10803726152, 10803726153, 10803726154, 10803726155, 10803726156, 10803726158, 10803726159, 10803726160, 10803726161, 10803726162, 10803726163, 10803726164, 10803726165, 10803726166, 10803726167, 10803726169, 10803726170, 10803726171, 10803726172, 10803726173, 10803726174, 10803726177, 10803726175, 10803726176, 10803726178, 10803726180, 10803726181, 10803726182, 10803726183, 10803726184, 10803726185, 10803726186, 10803726187, 10803726188, 10803726189, 10803726191, 10803726192, 10803726193, 10803726194, 10803726195, 10803726197, 10803726196, 10803726198, 10803726199, 10803726200, 10803726053, 10803726054, 10803726055, 10803726056, 10803726057, 10803726058, 10803726059, 10803726060, 10803726061, 10803726062, 10803726064, 10803726065, 10803726066, 10803726067, 10803726068, 10803726069, 10803726070, 10803726071, 10803726072, 10803726073, 10803726075, 10803726077, 10803726076, 10803726078, 10803726079, 10803726080, 10803726081, 10803726082, 10803726083, 10803726084, 10803726086, 10803726087, 10803726089, 10803726088, 10803726090, 10803726091, 10803726092, 10803726093, 10803726094, 10803726095, 10803726097, 10803726098, 10803726099, 10803726100, 10803726101, 10803726102, 10803726103, 10803726104, 10803726105, 10803726106, 10803726108, 10803726202, 10803726213, 10803726223, 10803726234, 10803726245, 10803726252, 10803726253, 10803726255, 10803726254, 10803726203, 10803726204, 10803726205, 10803726206, 10803726207, 10803726208, 10803726209, 10803726210, 10803726211, 10803726212, 10803726214, 10803726215, 10803726216, 10803726217, 10803726218, 10803726219, 10803726220, 10803726221, 10803726222, 10803726224, 10803726225, 10803726226, 10803726227, 10803726228, 10803726229, 10803726230, 10803726231, 10803726232, 10803726233, 10803726235, 10803726236, 10803726237, 10803726238, 10803726239, 10803726240, 10803726241, 10803726242, 10803726243, 10803726244, 10803726246, 10803726247, 10803726248, 10803726249, 10803726250, 10803726251, 10803726259, 10803726373, 10803726484, 10803726575, 10803726586, 10803726597, 10803726607, 10803726618, 10803726629, 10803726260, 10803726271, 10803726282, 10803726293, 10803726307, 10803726318, 10803726329, 10803726340, 10803726351, 10803726362, 10803726374, 10803726385, 10803726396, 10803726407, 10803726418, 10803726429, 10803726440, 10803726451, 10803726462, 10803726473, 10803726485, 10803726496, 10803726507, 10803726518, 10803726529, 10803726540, 10803726551, 10803726562, 10803726574, 10803726576, 10803726577, 10803726578, 10803726579, 10803726580, 10803726581, 10803726582, 10803726583, 10803726584, 10803726585, 10803726587, 10803726588, 10803726589, 10803726590, 10803726591, 10803726592, 10803726593, 10803726594, 10803726595, 10803726596, 10803726598, 10803726599, 10803726600, 10803726601, 10803726602, 10803726603, 10803726604, 10803726605, 10803726606, 10803726608, 10803726609, 10803726610, 10803726611, 10803726612, 10803726613, 10803726614, 10803726615, 10803726616, 10803726617, 10803726619, 10803726620, 10803726621, 10803726622, 10803726623, 10803726624, 10803726625, 10803726626, 10803726627, 10803726628, 10803726630, 10803726631, 10803726632, 10803726633, 10803726634, 10803726635, 10803726636, 10803726637, 10803726638, 10803726639, 10803726261, 10803726262, 10803726263, 10803726264, 10803726265, 10803726266, 10803726267, 10803726268, 10803726269, 10803726270, 10803726272, 10803726273, 10803726274, 10803726275, 10803726276, 10803726277, 10803726278, 10803726279, 10803726280, 10803726281, 10803726283, 10803726284, 10803726285, 10803726286, 10803726287, 10803726288, 10803726289, 10803726290, 10803726291, 10803726292, 10803726294, 10803726295, 10803726296, 10803726297, 10803726298, 10803726299, 10803726303, 10803726304, 10803726305, 10803726306, 10803726308, 10803726309, 10803726310, 10803726311, 10803726312, 10803726313, 10803726314, 10803726315, 10803726316, 10803726317, 10803726319, 10803726320, 10803726321, 10803726322, 10803726323, 10803726324, 10803726325, 10803726326, 10803726327, 10803726328, 10803726330, 10803726331, 10803726332, 10803726333, 10803726334, 10803726335, 10803726336, 10803726337, 10803726338, 10803726339, 10803726341, 10803726342, 10803726343, 10803726344, 10803726345, 10803726346, 10803726347, 10803726348, 10803726349, 10803726350, 10803726352, 10803726353, 10803726354, 10803726355, 10803726356, 10803726357, 10803726358, 10803726359, 10803726360, 10803726361, 10803726363, 10803726364, 10803726365, 10803726366, 10803726367, 10803726368, 10803726369, 10803726370, 10803726371, 10803726372, 10803726375, 10803726376, 10803726377, 10803726378, 10803726379, 10803726380, 10803726381, 10803726382, 10803726383, 10803726384, 10803726386, 10803726387, 10803726388, 10803726389, 10803726390, 10803726391, 10803726392, 10803726393, 10803726394, 10803726395, 10803726397, 10803726398, 10803726399, 10803726400, 10803726401, 10803726402, 10803726403, 10803726404, 10803726405, 10803726406, 10803726408, 10803726409, 10803726410, 10803726411, 10803726412, 10803726413, 10803726414, 10803726415, 10803726416, 10803726417, 10803726419, 10803726420, 10803726421, 10803726422, 10803726423, 10803726424, 10803726425, 10803726426, 10803726427, 10803726428, 10803726430, 10803726431, 10803726432, 10803726433, 10803726434, 10803726435, 10803726436, 10803726437, 10803726438, 10803726439, 10803726441, 10803726442, 10803726443, 10803726444, 10803726445, 10803726446, 10803726447, 10803726448, 10803726449, 10803726450, 10803726452, 10803726453, 10803726454, 10803726455, 10803726456, 10803726457, 10803726458, 10803726459, 10803726460, 10803726461, 10803726463, 10803726464, 10803726465, 10803726466, 10803726467, 10803726468, 10803726469, 10803726470, 10803726471, 10803726472, 10803726474, 10803726475, 10803726476, 10803726477, 10803726478, 10803726479, 10803726480, 10803726481, 10803726482, 10803726483, 10803726486, 10803726487, 10803726488, 10803726489, 10803726490, 10803726491, 10803726492, 10803726493, 10803726494, 10803726495, 10803726497, 10803726498, 10803726499, 10803726500, 10803726501, 10803726502, 10803726503, 10803726504, 10803726505, 10803726506, 10803726508, 10803726509, 10803726510, 10803726511, 10803726512, 10803726513, 10803726514, 10803726515, 10803726516, 10803726517, 10803726519, 10803726520, 10803726521, 10803726522, 10803726523, 10803726524, 10803726525, 10803726526, 10803726527, 10803726528, 10803726530, 10803726531, 10803726532, 10803726533, 10803726534, 10803726535, 10803726536, 10803726537, 10803726538, 10803726539, 10803726541, 10803726542, 10803726543, 10803726544, 10803726545, 10803726546, 10803726547, 10803726548, 10803726549, 10803726550, 10803726552, 10803726553, 10803726554, 10803726555, 10803726556, 10803726557, 10803726558, 10803726559, 10803726560, 10803726561, 10803726563, 10803726564, 10803726565, 10803726566, 10803726567, 10803726568, 10803726569, 10803726571, 10803726570, 10803726572, 10803726573, 10803726693, 10803726646, 10803726704, 10803726721, 10803726732, 10803726743, 10803726754, 10803726765, 10803726776, 10803726647, 10803726658, 10803726669, 10803726680, 10803726687, 10803726688, 10803726689, 10803726690, 10803726691, 10803726692, 10803726694, 10803726695, 10803726696, 10803726697, 10803726698, 10803726699, 10803726700, 10803726701, 10803726702, 10803726703, 10803726705, 10803726706, 10803726707, 10803726708, 10803726715, 10803726716, 10803726717, 10803726718, 10803726719, 10803726720, 10803726722, 10803726723, 10803726724, 10803726725, 10803726726, 10803726727, 10803726728, 10803726729, 10803726730, 10803726731, 10803726733, 10803726734, 10803726735, 10803726736, 10803726737, 10803726738, 10803726739, 10803726740, 10803726741, 10803726742, 10803726744, 10803726745, 10803726746, 10803726747, 10803726748, 10803726749, 10803726750, 10803726751, 10803726752, 10803726753, 10803726755, 10803726756, 10803726757, 10803726758, 10803726759, 10803726760, 10803726761, 10803726762, 10803726763, 10803726764, 10803726766, 10803726767, 10803726768, 10803726769, 10803726770, 10803726771, 10803726772, 10803726773, 10803726774, 10803726775, 10803726777, 10803726778, 10803726779, 10803726780, 10803726781, 10803726782, 10803726783, 10803726784, 10803726785, 10803726786, 10803726648, 10803726649, 10803726650, 10803726651, 10803726652, 10803726653, 10803726654, 10803726655, 10803726656, 10803726657, 10803726659, 10803726660, 10803726661, 10803726663, 10803726662, 10803726664, 10803726665, 10803726666, 10803726667, 10803726668, 10803726670, 10803726671, 10803726672, 10803726673, 10803726674, 10803726675, 10803726676, 10803726677, 10803726678, 10803726679, 10803726681, 10803726682, 10803726683, 10803726684, 10803726685, 10803726686, 10803726870, 10803726802, 10803726881, 10803726892, 10803726859, 10803726903, 10803726914, 10803726925, 10803726936, 10803726803, 10803726814, 10803726825, 10803726836, 10803726847, 10803726854, 10803726855, 10803726856, 10803726857, 10803726858, 10803726860, 10803726861, 10803726862, 10803726863, 10803726864, 10803726865, 10803726866, 10803726867, 10803726868, 10803726869, 10803726871, 10803726872, 10803726873, 10803726874, 10803726875, 10803726876, 10803726877, 10803726878, 10803726879, 10803726880, 10803726882, 10803726883, 10803726884, 10803726885, 10803726886, 10803726887, 10803726888, 10803726889, 10803726890, 10803726891, 10803726893, 10803726894, 10803726895, 10803726896, 10803726897, 10803726898, 10803726899, 10803726900, 10803726901, 10803726902, 10803726904, 10803726905, 10803726906, 10803726907, 10803726908, 10803726909, 10803726910, 10803726911, 10803726912, 10803726913, 10803726915, 10803726916, 10803726917, 10803726918, 10803726919, 10803726920, 10803726921, 10803726922, 10803726923, 10803726924, 10803726926, 10803726927, 10803726928, 10803726929, 10803726930, 10803726931, 10803726932, 10803726933, 10803726934, 10803726935, 10803726937, 10803726938, 10803726939, 10803726940, 10803726941, 10803726942, 10803726943, 10803726944, 10803726945, 10803726946, 10803726804, 10803726805, 10803726806, 10803726807, 10803726808, 10803726809, 10803726810, 10803726811, 10803726812, 10803726813, 10803726815, 10803726816, 10803726817, 10803726818, 10803726819, 10803726820, 10803726821, 10803726822, 10803726823, 10803726831, 10803726832, 10803726833, 10803726829, 10803726828, 10803726834, 10803726835, 10803726830, 10803726837, 10803726826, 10803726827, 10803726838, 10803726839, 10803726840, 10803726841, 10803726842, 10803726843, 10803726844, 10803726845, 10803726846, 10803726848, 10803726849, 10803726824, 10803726850, 10803726851, 10803726852, 10803726853, 10803727174, 10803727185, 10803727196, 10803727207, 10803727216, 10803727223, 10803727232, 10803727233, 10803727234, 10803727175, 10803727176, 10803727177, 10803727178, 10803727179, 10803727180, 10803727181, 10803727182, 10803727183, 10803727184, 10803727186, 10803727187, 10803727188, 10803727189, 10803727190, 10803727191, 10803727192, 10803727193, 10803727194, 10803727195, 10803727197, 10803727198, 10803727199, 10803727200, 10803727201, 10803727202, 10803727203, 10803727204, 10803727205, 10803727206, 10803727208, 10803727209, 10803727210, 10803727211, 10803727212, 10803727213, 10803727214, 10803727215, 10803727217, 10803727218, 10803727219, 10803727220, 10803727221, 10803727222, 10803727224, 10803727225, 10803727226, 10803727227, 10803727228, 10803727229, 10803727230, 10803727231, 10803727384, 10803727395, 10803727406, 10803727417, 10803727428, 10803727439, 10803727450, 10803727461, 10803727464, 10803727385, 10803727386, 10803727387, 10803727388, 10803727389, 10803727390, 10803727391, 10803727392, 10803727393, 10803727394, 10803727396, 10803727397, 10803727398, 10803727399, 10803727400, 10803727401, 10803727402, 10803727403, 10803727404, 10803727405, 10803727407, 10803727408, 10803727409, 10803727410, 10803727411, 10803727412, 10803727413, 10803727414, 10803727415, 10803727416, 10803727418, 10803727419, 10803727420, 10803727421, 10803727422, 10803727423, 10803727424, 10803727425, 10803727426, 10803727427, 10803727429, 10803727430, 10803727431, 10803727432, 10803727433, 10803727434, 10803727435, 10803727436, 10803727437, 10803727438, 10803727440, 10803727441, 10803727442, 10803727443, 10803727444, 10803727445, 10803727446, 10803727447, 10803727448, 10803727449, 10803727451, 10803727452, 10803727453, 10803727454, 10803727455, 10803727456, 10803727457, 10803727458, 10803727459, 10803727460, 10803727462, 10803727463, 10803727468, 10803727745, 10803727855, 10803727914, 10803727925, 10803727936, 10803727947, 10803727958, 10803727469, 10803727969, 10803727480, 10803727491, 10803727502, 10803727679, 10803727690, 10803727701, 10803727712, 10803727723, 10803727734, 10803727746, 10803727757, 10803727768, 10803727779, 10803727790, 10803727801, 10803727812, 10803727823, 10803727834, 10803727844, 10803727856, 10803727867, 10803727878, 10803727889, 10803727900, 10803727909, 10803727910, 10803727911, 10803727912, 10803727913, 10803727915, 10803727916, 10803727917, 10803727918, 10803727919, 10803727920, 10803727921, 10803727922, 10803727923, 10803727924, 10803727926, 10803727927, 10803727928, 10803727929, 10803727930, 10803727931, 10803727932, 10803727933, 10803727934, 10803727935, 10803727937, 10803727938, 10803727939, 10803727940, 10803727941, 10803727942, 10803727943, 10803727944, 10803727945, 10803727946, 10803727948, 10803727949, 10803727950, 10803727951, 10803727952, 10803727953, 10803727954, 10803727955, 10803727956, 10803727957, 10803727963, 10803727959, 10803727964, 10803727960, 10803727961, 10803727962, 10803727966, 10803727965, 10803727967, 10803727968, 10803727970, 10803727971, 10803727972, 10803727973, 10803727974, 10803727976, 10803727977, 10803727978, 10803727979, 10803727470, 10803727471, 10803727472, 10803727975, 10803727473, 10803727474, 10803727476, 10803727475, 10803727477, 10803727478, 10803727479, 10803727481, 10803727482, 10803727483, 10803727484, 10803727485, 10803727486, 10803727487, 10803727488, 10803727489, 10803727490, 10803727492, 10803727493, 10803727494, 10803727496, 10803727497, 10803727498, 10803727495, 10803727499, 10803727500, 10803727501, 10803727503, 10803727504, 10803727505, 10803727506, 10803727507, 10803727508, 10803727675, 10803727676, 10803727677, 10803727678, 10803727680, 10803727681, 10803727682, 10803727683, 10803727684, 10803727685, 10803727686, 10803727687, 10803727688, 10803727689, 10803727691, 10803727692, 10803727693, 10803727694, 10803727695, 10803727696, 10803727697, 10803727698, 10803727699, 10803727700, 10803727702, 10803727703, 10803727704, 10803727705, 10803727706, 10803727707, 10803727708, 10803727709, 10803727710, 10803727711, 10803727713, 10803727714, 10803727715, 10803727716, 10803727717, 10803727718, 10803727719, 10803727720, 10803727721, 10803727722, 10803727724, 10803727729, 10803727725, 10803727726, 10803727727, 10803727728, 10803727730, 10803727731, 10803727732, 10803727733, 10803727735, 10803727736, 10803727737, 10803727738, 10803727739, 10803727740, 10803727741, 10803727742, 10803727743, 10803727744, 10803727747, 10803727748, 10803727749, 10803727750, 10803727751, 10803727752, 10803727753, 10803727754, 10803727755, 10803727756, 10803727758, 10803727759, 10803727760, 10803727761, 10803727762, 10803727763, 10803727764, 10803727765, 10803727766, 10803727767, 10803727769, 10803727770, 10803727771, 10803727772, 10803727773, 10803727774, 10803727775, 10803727776, 10803727777, 10803727778, 10803727780, 10803727781, 10803727782, 10803727783, 10803727784, 10803727785, 10803727786, 10803727787, 10803727788, 10803727789, 10803727791, 10803727792, 10803727793, 10803727794, 10803727795, 10803727796, 10803727797, 10803727798, 10803727799, 10803727800, 10803727802, 10803727803, 10803727804, 10803727805, 10803727806, 10803727807, 10803727808, 10803727809, 10803727810, 10803727811, 10803727813, 10803727814, 10803727815, 10803727816, 10803727817, 10803727818, 10803727819, 10803727820, 10803727821, 10803727822, 10803727824, 10803727825, 10803727826, 10803727827, 10803727828, 10803727829, 10803727830, 10803727831, 10803727832, 10803727833, 10803727835, 10803727836, 10803727837, 10803727838, 10803727839, 10803727840, 10803727841, 10803727842, 10803727843, 10803727845, 10803727846, 10803727847, 10803727848, 10803727849, 10803727850, 10803727851, 10803727852, 10803727853, 10803727854, 10803727857, 10803727858, 10803727859, 10803727860, 10803727861, 10803727862, 10803727863, 10803727864, 10803727865, 10803727866, 10803727868, 10803727869, 10803727870, 10803727871, 10803727872, 10803727873, 10803727874, 10803727875, 10803727876, 10803727877, 10803727879, 10803727880, 10803727881, 10803727882, 10803727883, 10803727884, 10803727885, 10803727886, 10803727887, 10803727888, 10803727890, 10803727891, 10803727892, 10803727893, 10803727894, 10803727895, 10803727896, 10803727897, 10803727898, 10803727899, 10803727901, 10803727902, 10803727903, 10803727904, 10803727905, 10803727906, 10803727907, 10803727908, 10803727986, 10803728100, 10803728211, 10803728297, 10803728308, 10803728319, 10803728330, 10803728341, 10803728352, 10803727987, 10803727998, 10803728009, 10803728020, 10803728031, 10803728042, 10803728053, 10803728064, 10803728075, 10803728089, 10803728101, 10803728112, 10803728123, 10803728134, 10803728145, 10803728156, 10803728167, 10803728178, 10803728189, 10803728200, 10803728212, 10803728222, 10803728233, 10803728244, 10803728255, 10803728266, 10803728277, 10803728288, 10803728295, 10803728296, 10803728298, 10803728299, 10803728300, 10803728301, 10803728302, 10803728303, 10803728304, 10803728305, 10803728306, 10803728307, 10803728309, 10803728310, 10803728311, 10803728312, 10803728313, 10803728314, 10803728315, 10803728316, 10803728317, 10803728318, 10803728320, 10803728321, 10803728322, 10803728323, 10803728324, 10803728325, 10803728326, 10803728327, 10803728328, 10803728329, 10803728331, 10803728332, 10803728333, 10803728334, 10803728335, 10803728336, 10803728337, 10803728338, 10803728339, 10803728340, 10803728342, 10803728343, 10803728344, 10803728345, 10803728346, 10803728347, 10803728348, 10803728349, 10803728350, 10803728351, 10803728353, 10803728354, 10803728355, 10803728356, 10803728357, 10803728358, 10803728359, 10803728360, 10803728361, 10803728362, 10803727988, 10803727989, 10803727990, 10803727991, 10803727992, 10803727993, 10803727994, 10803727995, 10803727996, 10803727997, 10803727999, 10803728000, 10803728001, 10803728002, 10803728003, 10803728004, 10803728005, 10803728006, 10803728007, 10803728008, 10803728010, 10803728011, 10803728012, 10803728013, 10803728014, 10803728015, 10803728016, 10803728017, 10803728018, 10803728019, 10803728021, 10803728022, 10803728023, 10803728024, 10803728025, 10803728026, 10803728027, 10803728028, 10803728029, 10803728030, 10803728032, 10803728033, 10803728034, 10803728035, 10803728036, 10803728037, 10803728038, 10803728039, 10803728040, 10803728041, 10803728043, 10803728044, 10803728045, 10803728046, 10803728047, 10803728048, 10803728049, 10803728050, 10803728051, 10803728052, 10803728054, 10803728055, 10803728056, 10803728057, 10803728058, 10803728059, 10803728060, 10803728061, 10803728062, 10803728063, 10803728065, 10803728066, 10803728067, 10803728068, 10803728069, 10803728070, 10803728071, 10803728072, 10803728073, 10803728074, 10803728076, 10803728077, 10803728078, 10803728079, 10803728080, 10803728084, 10803728085, 10803728086, 10803728087, 10803728088, 10803728090, 10803728091, 10803728092, 10803728093, 10803728094, 10803728095, 10803728096, 10803728097, 10803728098, 10803728099, 10803728102, 10803728103, 10803728104, 10803728105, 10803728106, 10803728107, 10803728108, 10803728109, 10803728110, 10803728111, 10803728113, 10803728114, 10803728115, 10803728116, 10803728117, 10803728118, 10803728119, 10803728120, 10803728121, 10803728122, 10803728124, 10803728125, 10803728126, 10803728127, 10803728128, 10803728129, 10803728130, 10803728131, 10803728132, 10803728133, 10803728135, 10803728136, 10803728137, 10803728138, 10803728139, 10803728140, 10803728141, 10803728142, 10803728143, 10803728144, 10803728146, 10803728147, 10803728148, 10803728149, 10803728150, 10803728151, 10803728152, 10803728153, 10803728154, 10803728155, 10803728157, 10803728158, 10803728159, 10803728160, 10803728161, 10803728162, 10803728163, 10803728164, 10803728165, 10803728166, 10803728168, 10803728169, 10803728170, 10803728171, 10803728172, 10803728173, 10803728174, 10803728175, 10803728176, 10803728177, 10803728179, 10803728180, 10803728181, 10803728182, 10803728183, 10803728184, 10803728185, 10803728186, 10803728187, 10803728188, 10803728190, 10803728191, 10803728192, 10803728193, 10803728194, 10803728195, 10803728196, 10803728197, 10803728198, 10803728199, 10803728201, 10803728202, 10803728203, 10803728204, 10803728205, 10803728206, 10803728207, 10803728208, 10803728209, 10803728210, 10803728213, 10803728214, 10803728215, 10803728216, 10803728217, 10803728218, 10803728219, 10803728220, 10803728221, 10803728223, 10803728224, 10803728225, 10803728226, 10803728227, 10803728228, 10803728229, 10803728230, 10803728231, 10803728232, 10803728234, 10803728235, 10803728236, 10803728237, 10803728238, 10803728239, 10803728240, 10803728241, 10803728242, 10803728243, 10803728245, 10803728246, 10803728247, 10803728248, 10803728249, 10803728250, 10803728251, 10803728252, 10803728253, 10803728254, 10803728256, 10803728257, 10803728258, 10803728259, 10803728260, 10803728261, 10803728262, 10803728263, 10803728264, 10803728265, 10803728267, 10803728268, 10803728269, 10803728270, 10803728271, 10803728272, 10803728273, 10803728274, 10803728275, 10803728276, 10803728278, 10803728279, 10803728280, 10803728281, 10803728282, 10803728283, 10803728284, 10803728285, 10803728286, 10803728287, 10803728289, 10803728290, 10803728291, 10803728292, 10803728293, 10803728294, 10803728373, 10803728380, 10803728381, 10803728372, 10803728374, 10803728375, 10803728376, 10803728377, 10803728378, 10803728379, 10803728384, 10803728502, 10803728538, 10803728560, 10803728527, 10803728549, 10803728582, 10803728547, 10803728593, 10803728385, 10803728396, 10803728407, 10803728418, 10803728429, 10803728440, 10803728571, 10803728451, 10803728462, 10803728473, 10803728484, 10803728503, 10803728514, 10803728519, 10803728520, 10803728521, 10803728522, 10803728523, 10803728524, 10803728525, 10803728526, 10803728528, 10803728529, 10803728530, 10803728531, 10803728532, 10803728533, 10803728534, 10803728550, 10803728535, 10803728536, 10803728537, 10803728548, 10803728539, 10803728540, 10803728541, 10803728542, 10803728543, 10803728551, 10803728544, 10803728545, 10803728546, 10803728552, 10803728555, 10803728553, 10803728554, 10803728556, 10803728558, 10803728419, 10803728557, 10803728420, 10803728559, 10803728561, 10803728562, 10803728563, 10803728564, 10803728565, 10803728566, 10803728567, 10803728568, 10803728569, 10803728570, 10803728572, 10803728573, 10803728577, 10803728578, 10803728579, 10803728580, 10803728574, 10803728581, 10803728576, 10803728583, 10803728584, 10803728585, 10803728586, 10803728587, 10803728588, 10803728589, 10803728575, 10803728590, 10803728591, 10803728592, 10803728594, 10803728595, 10803728596, 10803728597, 10803728598, 10803728599, 10803728600, 10803728601, 10803728386, 10803728602, 10803728603, 10803728387, 10803728388, 10803728389, 10803728390, 10803728391, 10803728392, 10803728393, 10803728394, 10803728397, 10803728398, 10803728399, 10803728400, 10803728401, 10803728402, 10803728403, 10803728404, 10803728405, 10803728395, 10803728406, 10803728408, 10803728409, 10803728410, 10803728411, 10803728412, 10803728413, 10803728414, 10803728415, 10803728416, 10803728417, 10803728421, 10803728422, 10803728423, 10803728424, 10803728425, 10803728426, 10803728439, 10803728441, 10803728442, 10803728428, 10803728427, 10803728432, 10803728436, 10803728430, 10803728435, 10803728433, 10803728434, 10803728431, 10803728437, 10803728438, 10803728443, 10803728444, 10803728445, 10803728446, 10803728447, 10803728448, 10803728449, 10803728450, 10803728452, 10803728453, 10803728454, 10803728455, 10803728456, 10803728457, 10803728458, 10803728459, 10803728460, 10803728461, 10803728463, 10803728464, 10803728465, 10803728466, 10803728467, 10803728468, 10803728469, 10803728470, 10803728471, 10803728472, 10803728474, 10803728475, 10803728476, 10803728477, 10803728478, 10803728479, 10803728480, 10803728481, 10803728482, 10803728483, 10803728485, 10803728486, 10803728487, 10803728488, 10803728489, 10803728490, 10803728491, 10803728492, 10803728501, 10803728493, 10803728504, 10803728505, 10803728511, 10803728509, 10803728510, 10803728512, 10803728513, 10803728517, 10803728515, 10803728518, 10803728516, 10803728506, 10803728507, 10803728508, 10803728383], 'projectCategory': 0, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/queuesdevaigeneratealbumdto/y1ntr0pkfeys_1hz-qp8v6x6.json', 'aiMetadata': {'photoIds': [], 'focus': [], 'personIds': [], 'subjects': [], 'density': 4}, 'conditionId': 'AAD_46245951_1a8a8233-b5b0-4547-9f6c-4da0ab99ae38.93.246', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}

    #_input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 412600, 'projectId': 46669751, 'userId': 605287669, 'userJobId': 1178402795, 'base_url': 'ptstorage_32://pictures/46/669/46669751/wojva6irswydalysu8', 'photos': [11201129286, 11201129287, 11201129288, 11201129289, 11201129290, 11201129291, 11201129292, 11201129293, 11201129294, 11201129295, 11201129296, 11201129297, 11201129298, 11201129299, 11201129300, 11201129301, 11201129302, 11201129303, 11201129304, 11201129305, 11201129306, 11201129307, 11201129308, 11201129309, 11201129310, 11201129311, 11201129312, 11201129313, 11201129314, 11201129315, 11201129316, 11201129317, 11201129318, 11201129319, 11201129320, 11201129321, 11201129322, 11201129323, 11201129324, 11201129325, 11201129326, 11201129327, 11201129328, 11201129329, 11201129330, 11201129331, 11201129332, 11201129333, 11201129334, 11201129335, 11201129336, 11201129337, 11201129338, 11201129339, 11201129340, 11201129341, 11201129342, 11201129343, 11201129344, 11201129345, 11201129346, 11201129347, 11201129348, 11201129349, 11201129350, 11201129351, 11201129352, 11201129353, 11201129354, 11201129355, 11201129356, 11201129357, 11201129358, 11201129359, 11201129360, 11201129361, 11201129362, 11201129363, 11201129364, 11201129365, 11201129366, 11201129367, 11201129368, 11201129369, 11201129370, 11201129371, 11201129372, 11201129373, 11201129374, 11201129375, 11201129376, 11201129377, 11201129378, 11201129379, 11201129380, 11201129381, 11201129382, 11201129383, 11201129384, 11201129385, 11201129386, 11201129387, 11201129388, 11201129389, 11201129390, 11201129391, 11201129392, 11201129393, 11201129394, 11201129395, 11201129396, 11201129397, 11201129398, 11201129399, 11201129400, 11201129401, 11201129402, 11201129403, 11201129404, 11201129405, 11201129406, 11201129407, 11201129408, 11201129409, 11201129410, 11201129411, 11201129412, 11201129413, 11201129414, 11201129415, 11201129416, 11201129417, 11201129418, 11201129419, 11201129420, 11201129421, 11201129422, 11201129423, 11201129424, 11201129425, 11201129426, 11201129427, 11201129428, 11201129429, 11201129430, 11201129431, 11201129432, 11201129433, 11201129434, 11201129435, 11201129436, 11201129437, 11201129438, 11201129439, 11201129440, 11201129441, 11201129442, 11201129443, 11201129444, 11201129445, 11201129446, 11201129447, 11201129448, 11201129449, 11201129450, 11201129451, 11201129452, 11201129453, 11201129454, 11201129455, 11201129456, 11201129457, 11201129458, 11201129459, 11201129460, 11201129461, 11201129462, 11201129463, 11201129464, 11201129465, 11201129466, 11201129467, 11201129468, 11201129469, 11201129470, 11201129471, 11201129472, 11201129473, 11201129474, 11201129475, 11201129476, 11201129477, 11201129478, 11201129479, 11201129480, 11201129481, 11201129482, 11201129483, 11201129484, 11201129485, 11201129486, 11201129487, 11201129488, 11201129489, 11201129490, 11201129491, 11201129492, 11201129493, 11201129494, 11201129495, 11201129496, 11201129497, 11201129498, 11201129499, 11201129500, 11201129501, 11201129502, 11201129503, 11201129504, 11201129505, 11201129506, 11201129507, 11201129508, 11201129509, 11201129510, 11201129511, 11201129512, 11201129513, 11201129514, 11201129515, 11201129516, 11201133516, 11201133517, 11201133518, 11201133519, 11201133520, 11201133521, 11201133522, 11201133523, 11201133524, 11201133525, 11201133526, 11201133527, 11201133528, 11201133529, 11201133530, 11201133531, 11201133532, 11201133533, 11201133534, 11201133535, 11201133536, 11201133537, 11201133538, 11201133539, 11201133540, 11201133541, 11201133542, 11201133543, 11201133544, 11201133545, 11201133546, 11201133547, 11201133548, 11201133549, 11201133550, 11201133551, 11201133552, 11201133553, 11201133554, 11201133555, 11201133556, 11201133557, 11201133558], 'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/queuesdevaigeneratealbumdto/xxslawu2kuqilz4urpn4z2im.json', 'aiMetadata': {'photoIds': [11201129286, 11201129287], 'focus': ['everyoneElse'], 'personIds': [12, 9], 'subjects': ['baby'], 'density': 3}, 'conditionId': 'AAD_46669751_8fd6b42e-f421-446c-befd-698b8cabf137.201.427', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}

    #_input_request = {'replyQueueName': 'testaigeneratealbumresponsedto', 'storeId': 4, 'accountId': 412346, 'projectId': 41341518, 'userId': 602668421, 'userJobId': 1173364617, 'base_url': 'ptstorage_15://pictures/41/341/41341518/27whkd9q2gbouyr0kf', 'photos': [10724494050, 10724494051, 10724494052, 10724494053, 10724494054, 10724494055, 10724494056, 10724494057, 10724494058, 10724494059, 10724494060, 10724494061, 10724494062, 10059449039, 10724287460, 10724287461, 10724287462, 10724287463, 10724287465, 9947080434, 10044486236, 10044486237, 10044486238, 10044486239, 10044486240, 10044486241, 10044486242, 10044486243, 10044486244, 10044486245, 10044486246, 10044486247, 10044486248, 10044486249, 10044486250, 10044486251, 10044486252, 10044486253, 10044486254, 10044486255, 10044486256, 10044486257, 10044486258, 10044486259, 10044486260, 10044486261, 10044486262, 10044486263, 10044486264, 10044486265, 10044486266, 10044486267, 10044486268, 10044486269, 10044486270, 10044486271, 10044486272, 10044486273, 10044486274, 10044486275, 10044486276, 10044486277, 10044486278, 10044486279, 10044486280, 10044486281, 10044486282, 10044486283, 10044486284, 10044486285, 10044486286, 10044486287, 10044486288, 10044486289, 10044486290, 10044486291, 10044518737, 10044518738, 10044518739, 10044518740, 10044518741, 10044518742, 10044518743, 10044518744, 10044518745, 10044518746, 10044518747, 10044518748, 10044518749, 10044518750, 10044518751, 10044518752, 10044518753, 10044518754, 10044518755, 10044518756, 10044518757, 10044518758, 10044518759, 10044518760, 10044518761, 10044518762, 10044518763, 10044518764, 10044518765, 10044518766, 10044518767, 10044518768, 10044518769, 10044518770, 10044518771, 10044518772, 10044518773, 10044518774, 10044518775, 10044518776, 10044518777, 10044518778, 10044518779, 10044518780, 10044518781, 10044518782, 10044518783, 10044518784, 10044518785, 10044518786, 10044518787, 10044518788, 10044518789, 10044518790, 10044518791, 10044518792, 10044518793, 10044518794, 10044518795, 10044518796, 10044518797, 10044518798, 10044518799, 10044518800, 10044518801, 10044518802, 10044518803, 10044518804, 10044518805, 10044518806, 10044518807, 10044518808, 10044518809, 10044518810, 10044518811, 10044518812, 10044518813, 10044518814, 10044518815, 10044518816, 10044518817, 10044518818, 10044518819, 10044518820, 10044518821, 10044518822, 10044518823, 10044518824, 10044518825, 10044518826, 10044518827, 10044518828, 10044518829, 10044518830, 10044518831, 10044518832, 10044518833, 10044518834, 10044518835, 10044518836, 10044518837, 10044518838, 10044518839, 10044518840, 10044518841, 10044518842, 10044518843, 10044518844, 10044518845, 10044518846, 10044518847, 10044518848, 10044518849, 10044518850, 10044518851, 10044518852, 10044518853, 10044518854, 10044518855, 10044518856, 10044518857, 10044518858, 10044518859, 10044518860, 10044518861, 10044518862, 10044518863, 10044518864, 10044518865, 10044518866, 10044518867, 10044518868, 10044518869, 10044518870, 10044518871, 10044518872, 10044518873, 10044518874, 10044518875, 10044518876, 10044518877, 10044518878, 10044518879, 10044518880, 10044518881, 10044518882, 10044518883, 10044518884, 10044518885, 10044518886, 10044518887, 10044518888, 10044518889, 10044518890, 10044518891, 10044518892, 10044518893, 10044518894, 10044518895, 10044518896, 10044518897, 10044518898, 10044518899, 10044518900, 10044518901, 10044518902, 10044518903, 10044518904, 10044518905, 10044518906, 10044518907, 10044518908, 10044518909, 10044518910, 10044518911, 10044518912, 10044518913, 10044518914, 10044518915, 10044518916, 10044518917, 10044518918, 10044518919, 10044518920, 10044518921, 10044518922, 10044518923, 10044518924, 10044518925, 10044518926, 10044518927, 10044518928, 10044518929, 10044518930, 10044518931, 10044518932, 10044518933, 10044518934, 10044518935, 10044518936, 10044518937, 10044518938, 10044518939, 10044518940, 10044518941, 10044518942, 10044518943, 10044518944, 10044518945, 10044518946, 10044518947, 10044518948, 10044518949, 10044518950, 10044518951, 10044518952, 10044518953, 10044518954, 10044518955, 10044518956, 10044518957, 10044518958, 10044518959, 10044518960, 10044518961, 10044518962, 10044518963, 10044518964, 10044518965, 10044518966, 10044518967, 10044518968, 10044518969, 10044518970, 10044518971, 10044518972, 10044518973, 10044518974, 10044518975, 10044518976, 10044518977, 10044518978, 10044518979, 10044518980, 10044518981, 10044518982, 10044518983, 10044518984, 10044518985, 10044518986, 10044518987, 10044518988, 10044518989, 10044518990, 10044518991, 10044518992, 10044518993, 10044518994, 10044518995, 10044518996, 10044518997, 10044518998, 10044518999, 10044519000, 10044519001, 10044519002, 10044519003, 10044519004, 10044519005, 10044519006, 10044519007, 10044519008, 10044519009, 10044519010, 10044519011, 10044519012, 10044519013, 10044519014, 10044519015, 10044519016, 10044519017, 10044519018, 10044519019, 10044519020, 10044519021, 10044519022, 10044519023, 10044519024, 10044519025, 10044519026, 10044519027, 10044519028, 10044519029, 10044519030, 10044519031, 10044519032, 10044519033, 10044519034, 10044519035, 10044519036, 10044519037, 10044519038, 10044519039, 10044519040, 10044519041, 10044519042, 10044519043, 10044519044, 10044519045, 10044519046, 10044519047, 10044519048, 10044519049, 10044519050, 10044519051, 10044519052, 10044519053, 10044519054, 10044519055, 10044519056, 10044519057, 10044519058, 10044519059, 10044519060, 10044519061, 10044519062, 10044519063, 10044519064, 10044519065, 10044519066, 10044519067, 10044519068, 10044519069, 10044519070, 10044519071, 10044519072, 10044519073, 10044519074, 10044519075, 10044519076, 10044519077, 10044519078, 10044519079, 10044519080, 10044519081, 10044519082, 10044519083, 10044519084, 10044519085, 10044519086, 10044519087, 10044519088, 10044519089, 10044519090, 10044519091, 10044519092, 10044519093, 10044519094, 10044519095, 10044519096, 10044519097, 10044519098, 10044519099, 10044519100, 10044519101, 10044519102, 10044519103, 10044519104, 10044519105, 10044519106, 10044519107, 10044519108, 10044519109, 10044519110, 10044519111, 10044519112, 10044519113, 10044519114, 10044519115, 10044519116, 10044519117, 10044519118, 10044519119, 10044519120, 10044519121, 10044519122, 10044519123, 10044519124, 10044519125, 10044519126, 10044519127, 10044519128, 10044519129, 10044519130, 10044519131, 10044519132, 10044519133, 10044519134, 10044519135, 10044519136, 10044520100, 10044520101, 10044520102, 10044520103, 10044520104, 10044520105, 10044520106, 10044520107, 10044520108, 10044520109, 10044520110, 10044520111, 10044520112, 10044520113, 10044520114, 10044520115, 10044520116, 10044520117, 10044520118, 10044520119, 10044520120, 10044520121, 10044520122, 10044520123, 10044520124, 10044520125, 10044520126, 10044520127, 10064416344, 10064416345, 10064416346, 10064416347, 10064416348], 'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/queuesdevaigeneratealbumdto/w1_8prb7n0ufbjh1rinschbw.json', 'aiMetadata': {'photoIds': [], 'focus': [], 'personIds': [], 'subjects': [], 'density': 3}, 'conditionId': 'AAD_41341518_36e710ff-75c3-446c-b7ca-940ac3bab874.223.95', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}

    settings_filename = os.environ.get('HostingSettingsPath',
                                       '/ptinternal/pictures/hosting/ai_settings_audiobeat.json.txt')
    intialize('AlbumDesigner', settings_filename)


    # WITHOUT DATEEEEEEEE
    _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 294318, 'projectId': 46559988,
     'userId': 304034801, 'userJobId': 1048365853,
     'base_url': 'ptstorage_32://pictures/46/559/46559988/9wsgwxz9tdqwuvf4sa',
     'photos': [10875544473, 10875544911, 10875544474, 10875544475, 10875544477, 10875544479, 10875544480, 10875544481,
                10875544482, 10875544483, 10875544484, 10875544485, 10875544486, 10875544487, 10875544488, 10875544489,
                10875544490, 10875544491, 10875544492, 10875544493, 10875544476, 10875544494, 10875544495, 10875544496,
                10875544497, 10875544498, 10875544499, 10875544500, 10875544501, 10875544502, 10875544503, 10875544504,
                10875544505, 10875544506, 10875544507, 10875544508, 10875544509, 10875544510, 10875544511, 10875544512,
                10875544513, 10875544514, 10875544516, 10875544517, 10875544518, 10875544519, 10875544520, 10875544521,
                10875544522, 10875544523, 10875544524, 10875544525, 10875544526, 10875544527, 10875544528, 10875544529,
                10875544530, 10875544531, 10875544532, 10875544533, 10875544534, 10875544535, 10875544536, 10875544515,
                10875544537, 10875544538, 10875544539, 10875544540, 10875544541, 10875544542, 10875544543, 10875544544,
                10875544545, 10875544546, 10875544547, 10875544548, 10875544549, 10875544550, 10875544551, 10875544552,
                10875544553, 10875544554, 10875544555, 10875544556, 10875544557, 10875544558, 10875544559, 10875544560,
                10875544561, 10875544562, 10875544563, 10875544564, 10875544565, 10875544566, 10875544567, 10875544568,
                10875544569, 10875544570, 10875544571, 10875544572, 10875544573, 10875544574, 10875544575, 10875544576,
                10875544577, 10875544578, 10875544579, 10875544580, 10875544581, 10875544582, 10875544583, 10875544584,
                10875544585, 10875544586, 10875544587, 10875544588, 10875544589, 10875544590, 10875544591, 10875544592,
                10875544593, 10875544594, 10875544595, 10875544596, 10875544597, 10875544598, 10875544599, 10875544600,
                10875544601, 10875544602, 10875544603, 10875544604, 10875544605, 10875544606, 10875544607, 10875544608,
                10875544609, 10875544610, 10875544611, 10875544612, 10875544613, 10875544614, 10875544615, 10875544616,
                10875544617, 10875544618, 10875544619, 10875544620, 10875544621, 10875544622, 10875544623, 10875544624,
                10875544625, 10875544626, 10875544627, 10875544628, 10875544629, 10875544630, 10875544631, 10875544632,
                10875544633, 10875544634, 10875544635, 10875544636, 10875544478, 10875544637, 10875544638, 10875544639,
                10875544640, 10875544641, 10875544642, 10875544643, 10875544644, 10875544645, 10875544646, 10875544647,
                10875544648, 10875544649, 10875544650, 10875544651, 10875544652, 10875544653, 10875544654, 10875544655,
                10875544656, 10875544657, 10875544658, 10875544659, 10875544660, 10875544661, 10875544662, 10875544663,
                10875544664, 10875544665, 10875544666, 10875544667, 10875544668, 10875544669, 10875544670, 10875544671,
                10875544672, 10875544673, 10875544674, 10875544675, 10875544676, 10875544678, 10875544679, 10875544680,
                10875544681, 10875544682, 10875544683, 10875544684, 10875544685, 10875544686, 10875544687, 10875544688,
                10875544689, 10875544690, 10875544691, 10875544692, 10875544693, 10875544694, 10875544695, 10875544696,
                10875544697, 10875544698, 10875544699, 10875544700, 10875544701, 10875544702, 10875544703, 10875544704,
                10875544705, 10875544706, 10875544707, 10875544708, 10875544709, 10875544710, 10875544711, 10875544712,
                10875544713, 10875544714, 10875544746, 10875544715, 10875544716, 10875544717, 10875544718, 10875544719,
                10875544720, 10875544721, 10875544722, 10875544723, 10875544724, 10875544725, 10875544726, 10875544727,
                10875544728, 10875544729, 10875544730, 10875544731, 10875544732, 10875544733, 10875544734, 10875544735,
                10875544736, 10875544737, 10875544738, 10875544739, 10875544740, 10875544741, 10875544742, 10875544743,
                10875544744, 10875544745, 10875544747, 10875544748, 10875544749, 10875544750, 10875544751, 10875544752,
                10875544753, 10875544754, 10875544755, 10875544756, 10875544757, 10875544758, 10875544759, 10875544760,
                10875544761, 10875544762, 10875544763, 10875544764, 10875544765, 10875544766, 10875544767, 10875544768,
                10875544769, 10875544770, 10875544771, 10875544772, 10875544773, 10875544774, 10875544775, 10875544776,
                10875544777, 10875544778, 10875544779, 10875544780, 10875544781, 10875544782, 10875544783, 10875544784,
                10875544785, 10875544786, 10875544787, 10875544788, 10875544789, 10875544790, 10875544791, 10875544792,
                10875544793, 10875544828, 10875544794, 10875544795, 10875544796, 10875544797, 10875544798, 10875544799,
                10875544800, 10875544801, 10875544802, 10875544803, 10875544804, 10875544805, 10875544806, 10875544807,
                10875544808, 10875544809, 10875544810, 10875544811, 10875544812, 10875544813, 10875544814, 10875544815,
                10875544816, 10875544817, 10875544818, 10875544819, 10875544820, 10875544821, 10875544822, 10875544823,
                10875544824, 10875544825, 10875544826, 10875544827, 10875544829, 10875544830, 10875544831, 10875544832,
                10875544833, 10875544834, 10875544835, 10875544836, 10875544837, 10875544838, 10875544839, 10875544840,
                10875544841, 10875544842, 10875544843, 10875544845, 10875544846, 10875544847, 10875544848, 10875544849,
                10875544850, 10875544851, 10875544852, 10875544853, 10875544854, 10875544855, 10875544856, 10875544857,
                10875544858, 10875544859, 10875544860, 10875544861, 10875544862, 10875544863, 10875544864, 10875544865,
                10875544866, 10875544867, 10875544868, 10875544869, 10875544870, 10875544871, 10875544872, 10875544882,
                10875544883, 10875544884, 10875544885, 10875544886, 10875544887, 10875544888, 10875544889, 10875544890,
                10875544891, 10875544892, 10875544893, 10875544894, 10875544895, 10875544896, 10875544908, 10875544897,
                10875544898, 10875544899, 10875544900, 10875544905, 10875544901, 10875544902, 10875544903, 10875544904,
                10875544906, 10875544907, 10875544909, 10875544910, 10875544912, 10875544913, 10875544914, 10875544915,
                10875544916, 10875544917, 10875544918, 10875544919, 10875544920, 10875544921, 10875544922, 10875544923,
                10875544924, 10875544925, 10875544926, 10875544927, 10875544928, 10875544929, 10875544930, 10875544931,
                10875544932, 10875544933, 10875544934, 10875544935, 10875544936, 10875544937, 10875544938, 10875544939,
                10875544940, 10875544941, 10875544942, 10875544943, 10875544944, 10875544945, 10875544946, 10875544947,
                10875544948, 10875544949, 10875544950, 10875544951, 10875544952, 10875544953, 10875544954, 10875544955,
                10875544956, 10875544957, 10875545125, 10875544959, 10875544960, 10875544961, 10875544962, 10875544963,
                10875544964, 10875544965, 10875544966, 10875544967, 10875544968, 10875544969, 10875544970, 10875544971,
                10875544972, 10875544973, 10875544974, 10875544975, 10875544976, 10875544977, 10875544978, 10875544979,
                10875544980, 10875544981, 10875544982, 10875544983, 10875544984, 10875544985, 10875544986, 10875544987,
                10875544988, 10875544989, 10875544990, 10875544991, 10875544992, 10875544958, 10875544993, 10875544994,
                10875544995, 10875544996, 10875544997, 10875544998, 10875544999, 10875545000, 10875545001, 10875545002,
                10875545003, 10875545004, 10875545005, 10875545006, 10875545007, 10875545008, 10875545009, 10875545010,
                10875545011, 10875545012, 10875545013, 10875545014, 10875545015, 10875545016, 10875545017, 10875545018,
                10875544844, 10875545019, 10875545020, 10875545021, 10875545022, 10875545023, 10875545024, 10875545025,
                10875545026, 10875545027, 10875545028, 10875545029, 10875545030, 10875545031, 10875545032, 10875545033,
                10875545034, 10875545035, 10875545036, 10875545037, 10875545038, 10875545039, 10875545040, 10875545041,
                10875545042, 10875545043, 10875545044, 10875545045, 10875545046, 10875545047, 10875545048, 10875545049,
                10875545050, 10875545051, 10875545052, 10875545053, 10875545054, 10875545055, 10875545056, 10875545057,
                10875545058, 10875545059, 10875545060, 10875545061, 10875545062, 10875545063, 10875545064, 10875545065,
                10875545066, 10875545067, 10875545068, 10875545069, 10875545070, 10875545071, 10875545072, 10875545073,
                10875545074, 10875545075, 10875545076, 10875545077, 10875545078, 10875545079, 10875545080, 10875545081,
                10875545082, 10875545083, 10875545084, 10875545085, 10875545086, 10875545087, 10875545088, 10875545089,
                10875545090, 10875545091, 10875545092, 10875545093, 10875545094, 10875545095, 10875545096, 10875545097,
                10875545098, 10875545099, 10875545100, 10875545101, 10875545102, 10875545103, 10875545104, 10875545105,
                10875545106, 10875545107, 10875545108, 10875545109, 10875545110, 10875545111, 10875545112, 10875545113,
                10875545114, 10875545115, 10875545116, 10875545117, 10875545118, 10875545119, 10875545120, 10875545121,
                10875545122, 10875545123, 10875545124, 10875545126, 10875545127, 10875545128, 10875545129, 10875545130,
                10875545131, 10875545132, 10875545133, 10875545134, 10875545135, 10875545136, 10875545137, 10875545138,
                10875545139, 10875545140, 10875545141, 10875545142, 10875545143, 10875545144, 10875545145, 10875545146,
                10875545147, 10875545148, 10875545149, 10875545150, 10875545151, 10875545152, 10875545153, 10875545154,
                10875545155, 10875545156, 10875544677, 10875545157, 10875545158, 10875545159, 10875545160, 10875545161,
                10875545162, 10875545163, 10875545164, 10875545165, 10875545166, 10875545167, 10875545168, 10875545169,
                10875545170, 10875545171, 10875545172, 10875545173, 10875545174, 10875545175, 10875545176, 10875545177,
                10875545178, 10875545179, 10875545180, 10875545181, 10875545182, 10875545183, 10875545184, 10875545185,
                10875545186, 10875545187, 10875545188, 10875545189, 10875545190, 10875545191, 10875545192, 10875545193,
                10875545194, 10875545195, 10875545196, 10875545197, 10875545198, 10875545199, 10875545200, 10875545201,
                10875545202, 10875545203, 10875545204, 10875545205, 10875545206, 10875545207, 10875545208, 10875545209,
                10875545210, 10875545211, 10875545212, 10875545213, 10875545214, 10875545215, 10875545216, 10875545217,
                10875545218, 10875545219, 10875545220, 10875545221, 10875545222, 10875545223, 10875545224, 10875545225,
                10875545226, 10875545227, 10875545228, 10875545229, 10875545230, 10875545231, 10875545232, 10875545233,
                10875545234, 10875545235, 10875545236, 10875545237, 10875545238, 10875545239, 10875545240, 10875545241,
                10875545242, 10875545243, 10875545244, 10875545245, 10875545246, 10875545247, 10875545248, 10875545249,
                10875545250, 10875545251, 10875545252, 10875545253, 10875545254, 10875545255, 10875545256, 10875545257,
                10875545258, 10875545259, 10875545260, 10875545261, 10875545262, 10875545263, 10875545264, 10875545265,
                10875545266, 10875545267, 10875545268, 10875545269, 10875545270, 10875545271, 10875545272, 10875545273,
                10875545274, 10875545275, 10875545276, 10875545277, 10875545278, 10875545279, 10875545280, 10875545281,
                10875545814, 10875545815, 10875545816, 10875545817, 10875545818, 10875545819, 10875545820, 10875545821,
                10875545822, 10875545823, 10875545824, 10875545825, 10875545826, 10875545827, 10875545828, 10875545829,
                10875545830, 10875545831, 10875545832, 10875545833, 10875545834, 10875545835, 10875545836, 10875545837,
                10875545838, 10875545839, 10875545840, 10875545841, 10875545842, 10875545843, 10875545977, 10875545844,
                10875545845, 10875545846, 10875545847, 10875545848, 10875545849, 10875545850, 10875545851, 10875545852,
                10875545853, 10875545854, 10875545855, 10875545856, 10875545857, 10875545858, 10875545859, 10875545860,
                10875545861, 10875545862, 10875545863, 10875545864, 10875545865, 10875545866, 10875545867, 10875545868,
                10875545869, 10875545870, 10875545871, 10875545872, 10875545873, 10875545874, 10875545875, 10875545876,
                10875545877, 10875545878, 10875545879, 10875545880, 10875545881, 10875545882, 10875545883, 10875545884,
                10875545885, 10875545886, 10875545887, 10875545888, 10875545889, 10875545890, 10875545891, 10875545892,
                10875545893, 10875545894, 10875545895, 10875545896, 10875545897, 10875545898, 10875545899, 10875545900,
                10875545901, 10875545902, 10875545903, 10875545904, 10875545905, 10875545906, 10875545907, 10875545908,
                10875545909, 10875545910, 10875545911, 10875545912, 10875545913, 10875545914, 10875545915, 10875545916,
                10875545917, 10875545918, 10875545919, 10875545920, 10875545921, 10875545922, 10875545923, 10875545924,
                10875545925, 10875545926, 10875545927, 10875545928, 10875545929, 10875545930, 10875545931, 10875545932,
                10875545933, 10875545934, 10875545935, 10875545936, 10875545937, 10875545938, 10875545939, 10875545940,
                10875545941, 10875545942, 10875545943, 10875545944, 10875545945, 10875545946, 10875545947, 10875545948,
                10875545949, 10875545950, 10875545951, 10875545952, 10875545953, 10875545954, 10875545955, 10875545956,
                10875545957, 10875545958, 10875545959, 10875545960, 10875545961, 10875545962, 10875545963, 10875545964,
                10875545965, 10875545966, 10875545967, 10875545968, 10875545969, 10875545970, 10875545971, 10875545972,
                10875545973, 10875545974, 10875545975, 10875545976, 10875545978, 10875545979, 10875545813, 10875545980,
                10875545981, 10875545982, 10875545983, 10875545984, 10875545985, 10875545986, 10875545987],
     'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None,
     'designInfoTempLocation': 'pictures/temp/queuesdevaigeneratealbumdto/qexs1fq2yeyvjlu-pcgtsn0m.json',
     'aiMetadata': {'photoIds': [], 'focus': [], 'personIds': [], 'subjects': [], 'density': 3},
     'conditionId': 'AAD_46559988_2bba34b1-fa6e-4646-9872-51c6fc690dfc.133.70', 'timedOut': False,
     'dependencyDeleted': False, 'retryCount': 0}


    #Without Time Number 2
    # _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 493990, 'projectId': 47981912,
    #  'userId': 588345648, 'userJobId': 1144911075,
    #  'base_url': 'ptstorage_1://pictures/47/981/47981912/5q7khaa47itlml28n8',
    #  'photos': [11192656630, 11192656631, 11192656632, 11192656633, 11192656634, 11192656635, 11192656636, 11192656637,
    #             11192656638, 11192656639, 11192656640, 11192656641, 11192656642, 11192656643, 11192656644, 11192656645,
    #             11192656646, 11192656647, 11192656648, 11192656649, 11192656650, 11192656651, 11192656652, 11192656653,
    #             11192656654, 11192656655, 11192656656, 11192656657, 11192656658, 11192656659, 11192656660, 11192656661,
    #             11192656662, 11192656663, 11192656664, 11192656665, 11192656666, 11192656667, 11192656668, 11192656669,
    #             11192656670, 11192656671, 11192656672, 11192656673, 11192656674, 11192656675, 11192656676, 11192656677,
    #             11192656678, 11192656679, 11192656680, 11192656681, 11192656682, 11192656683, 11192656684, 11192656685,
    #             11192656686, 11192656687, 11192656688, 11192656689, 11192656690, 11192656691, 11192656692, 11192656693,
    #             11192656694, 11192656695, 11192656696, 11192656697, 11192656698, 11192656699, 11192656700, 11192656701,
    #             11192656702, 11192656703, 11192656704, 11192656705, 11192656706, 11192656707, 11192656708, 11192656709,
    #             11192656710, 11192656711, 11192656712, 11192656713, 11192656714, 11192656715, 11192656716, 11192656717,
    #             11192656718, 11192656719, 11192656720, 11192656721, 11192656722, 11192656723, 11192656724, 11192656725,
    #             11192656726, 11192656727, 11192656728, 11192656729, 11192656730, 11192656731, 11192656732, 11192656733,
    #             11192656734, 11192656735, 11192656736, 11192656737, 11192656738, 11192656739, 11192656740, 11192656741,
    #             11192656742, 11192656743, 11192656744, 11192656745, 11192656746, 11192656747, 11192656748, 11192656749,
    #             11192656750, 11192656751, 11192656752, 11192656753, 11192656754, 11192656755, 11192656756, 11192656757,
    #             11192656758, 11192656759, 11192656760, 11192656761, 11192656762, 11192656763, 11192656764, 11192656765,
    #             11192656766, 11192656767, 11192656768, 11192656769, 11192656770, 11192656771, 11192656772, 11192656773,
    #             11192656774, 11192656775, 11192656776, 11192656777, 11192656778, 11192656779, 11192656780, 11192656781,
    #             11192656782, 11192656783, 11192656784, 11192656785, 11192656786, 11192656787, 11192656788, 11192656789,
    #             11192656790, 11192656791, 11192656792, 11192656793, 11192656794, 11192656795, 11192656796, 11192656797,
    #             11192656798, 11192656799, 11192656800, 11192656801, 11192656802, 11192656803, 11192656804, 11192656805,
    #             11192656806, 11192656807, 11192656808, 11192656809, 11192656810, 11192656811, 11192656812, 11192656813,
    #             11192656814, 11192656815, 11192656816, 11192656817, 11192656818, 11192656819, 11192656820, 11192656821,
    #             11192656822, 11192656823, 11192656824, 11192656825, 11192656826, 11192656827, 11192656828, 11192656829,
    #             11192656830, 11192656831, 11192656832, 11192656833, 11192656834, 11192656835, 11192656836, 11192656837,
    #             11192656838, 11192656839, 11192656840, 11192656841, 11192656842, 11192656843, 11192656844, 11192656845,
    #             11192656846, 11192656847, 11192656848, 11192656849, 11192656850, 11192656851, 11192656852, 11192656853,
    #             11192656854, 11192656855, 11192656856, 11192656857, 11192656858, 11192656859, 11192656860, 11192656861,
    #             11192656862, 11192656863, 11192656864, 11192656865, 11192656866, 11192656867, 11192656868, 11192656869,
    #             11192656870, 11192656871, 11192656872, 11192656873, 11192656874, 11192656875, 11192656876, 11192656877,
    #             11192656878, 11192656879, 11192656880, 11192656881, 11192656882, 11192656883, 11192656884, 11192656885,
    #             11192656886, 11192656887, 11192656888, 11192656889, 11192656890, 11192656891, 11192656892, 11192656893,
    #             11192656894, 11192656895, 11192656896, 11192656897, 11192656898, 11192656899, 11192656900, 11192656901,
    #             11192656902, 11192656903, 11192656904, 11192656905, 11192656906, 11192656907, 11192656908, 11192656909,
    #             11192656910, 11192656911, 11192656912, 11192656913, 11192656914, 11192656915, 11192656916, 11192656917,
    #             11192656918, 11192656919, 11192656920, 11192656921, 11192656922, 11192656923, 11192656924, 11192656925,
    #             11192656926, 11192656927, 11192656928, 11192656929, 11192656930, 11192656931, 11192656932, 11192656933,
    #             11192656934, 11192656935, 11192656936, 11192656937, 11192656938, 11192656939, 11192656940, 11192656941,
    #             11192656942, 11192656943, 11192656944, 11192656945, 11192656946, 11192656947, 11192656948, 11192656949,
    #             11192656950, 11192656951, 11192656952, 11192656953, 11192656954, 11192656955, 11192656956, 11192656957,
    #             11192656958, 11192656959, 11192656960, 11192656961, 11192656962, 11192656963, 11192656964, 11192656965,
    #             11192656966, 11192656967, 11192656968, 11192656969, 11192656970, 11192656971, 11192656972, 11192656973,
    #             11192656974, 11192656975, 11192656976, 11192656977, 11192656978, 11192656979, 11192656980, 11192656981,
    #             11192656982, 11192656983, 11192656984, 11192656985, 11192656986, 11192656987, 11192656988, 11192656989,
    #             11192656990, 11192656991, 11192656992, 11192656993, 11192656994, 11192656995, 11192656996, 11192656997,
    #             11192656998, 11192656999, 11192657000, 11192657001, 11192657002, 11192657003, 11192657004, 11192657005,
    #             11192657006, 11192657007, 11192657008, 11192657009, 11192657010, 11192657011, 11192657012, 11192657013,
    #             11192657014, 11192657015, 11192657016, 11192657017, 11192657018, 11192657019, 11192657020, 11192657021,
    #             11192657022, 11192657023, 11192657024, 11192657025, 11192657026, 11192657027, 11192657028, 11192657029,
    #             11192657053, 11192657054, 11192657055, 11192657056, 11192657057, 11192657058, 11192657059, 11192657060,
    #             11192657061, 11192657062, 11192657063, 11192657064, 11192657065, 11192657066, 11192657067, 11192657068,
    #             11192657069, 11192657070, 11192657071, 11192657072, 11192657073, 11192657074, 11192657075, 11192657076,
    #             11192657077, 11192657078, 11192657079, 11192657080, 11192657081, 11192657082, 11192657083, 11192657084,
    #             11192657085, 11192657086, 11192657087, 11192657088, 11192657089, 11192657090, 11192657091, 11192657092,
    #             11192657093, 11192657094, 11192657095, 11192657096, 11192657097, 11192657098, 11192657099, 11192657100,
    #             11192657101, 11192657102, 11192657103, 11192657104, 11192657105, 11192657106, 11192657107, 11192657108,
    #             11192657109, 11192657110, 11192657111, 11192657112, 11192657113, 11192657114, 11192657115, 11192657116,
    #             11192657117, 11192657118, 11192657119, 11192657120, 11192657121, 11192657122, 11192657123, 11192657124,
    #             11192657125, 11192657126, 11192657127, 11192657128, 11192657129, 11192657130, 11192657131, 11192657132,
    #             11192657133, 11192657134, 11192657135, 11192657136, 11192657137, 11192657138, 11192657139, 11192657140,
    #             11192657141, 11192657142, 11192657143, 11192657144, 11192657145, 11192657146, 11192657147, 11192657148,
    #             11192657149, 11192657150, 11192657151, 11192657152, 11192657153, 11192657154, 11192657155, 11192657156,
    #             11192657157, 11192657158, 11192657159, 11192657160, 11192657161, 11192657162, 11192657163, 11192657164,
    #             11192657165, 11192657166, 11192657167, 11192657168, 11192657169, 11192657170, 11192657171, 11192657172,
    #             11192657173, 11192657174, 11192657175, 11192657176, 11192657177, 11192657178, 11192657179, 11192657180,
    #             11192657181, 11192657182, 11192657183, 11192657184, 11192657185, 11192657186, 11192657187, 11192657188,
    #             11192657189, 11192657190, 11192657191, 11192657192, 11192657193, 11192657194, 11192657195, 11192657196,
    #             11192657197, 11192657198, 11192657199, 11192657200, 11192657201, 11192657202, 11192657203, 11192657204,
    #             11192657205, 11192657206, 11192657207, 11192657208, 11192657209, 11192657210, 11192657211, 11192657212,
    #             11192657213, 11192657214, 11192657215, 11192657216, 11192657217, 11192657218, 11192657219, 11192657220,
    #             11192657221, 11192657222, 11192657223, 11192657224, 11192657225, 11192657226, 11192657227, 11192657228,
    #             11192657229, 11192657230, 11192657231, 11192657232, 11192657233, 11192657234], 'projectCategory': 1,
    #  'compositionPackageId': -1, 'designInfo': None,
    #  'designInfoTempLocation': 'pictures/temp/queuesdevaigeneratealbumdto/unxzt7xrvuchyydgp_wmqx4g.json',
    #  'aiMetadata': {'photoIds': [], 'focus': [], 'personIds': [], 'subjects': [], 'density': 3},
    #  'conditionId': 'AAD_47981912_908cb0fb-2e39-4a8a-ace7-0a451dc25e85.160.10', 'timedOut': False,
    #  'dependencyDeleted': False, 'retryCount': 0}

    #_input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310, 'projectId': 46227780, 'userId': 570976104, 'userJobId': 1111385993, 'base_url': 'ptstorage_32://pictures/46/227/46227780/njev8ankt8x9b7ynth', 'photos': [10799337783, 10799337786, 10799337789, 10799337792, 10799337796, 10799337798, 10799337802, 10799337805, 10799337808, 10799337811, 10799337817, 10799337820, 10799337824, 10799337828, 10799337832, 10799337836, 10799337840, 10799337845, 10799337849, 10799337853, 10799337857, 10799337861, 10799337865, 10799337869, 10799337873, 10799337876, 10799337878, 10799337882, 10799337886, 10799337889, 10799337893, 10799337897, 10799337901, 10799337905, 10799337909, 10799337913, 10799337918, 10799337921, 10799337926, 10799337929, 10799337933, 10799337938, 10799337942, 10799337946, 10799337950, 10799337954, 10799337958, 10799337962, 10799337966, 10799337970, 10799337975, 10799337980, 10799337985, 10799337990, 10799337995, 10799338000, 10799338005, 10799338010, 10799338014, 10799338020, 10799338025, 10799338031, 10799338036, 10799338041, 10799338046, 10799338051, 10799338056, 10799338061, 10799338066, 10799338071, 10799338075, 10799338080, 10799338085, 10799338089, 10799338094, 10799338099, 10799338104, 10799338110, 10799338120, 10799338125, 10799338130, 10799338135, 10799338139, 10799338144, 10799338149, 10799338154, 10799338159, 10799338164, 10799338169, 10799338173, 10799338177, 10799338182, 10799338187, 10799338192, 10799338197, 10799338202, 10799338207, 10799338212, 10799338217, 10799338222, 10799338227, 10799338231, 10799338236, 10799338241, 10799338247, 10799338253, 10799338259, 10799338265, 10799338271, 10799338277, 10799338283, 10799338289, 10799338295, 10799338301, 10799338307, 10799338314, 10799338321, 10799338328, 10799338334, 10799338340, 10799338345, 10799338350, 10799338358, 10799338366, 10799338372, 10799338379, 10799338386, 10799338394, 10799338402, 10799338409, 10799338416, 10799338423, 10799338429, 10799338436, 10799338443, 10799338451, 10799338460, 10799338468, 10799338475, 10799338484, 10799338493, 10799338501, 10799338509, 10799338518, 10799338530, 10799338539, 10799338547, 10799338555, 10799338562, 10799338570, 10799338578, 10799338585, 10799338593, 10799338601, 10799338605, 10799338613, 10799338621, 10799338629, 10799338637, 10799338645, 10799338653, 10799338662, 10799338670, 10799338677, 10799338683, 10799338691, 10799338700, 10799338709, 10799338718, 10799338727, 10799338735, 10799338745, 10799338753, 10799338762, 10799338772, 10799338781, 10799338789, 10799338800, 10799338809, 10799338817, 10799338825, 10799338835, 10799338844, 10799338855, 10799338867, 10799338878, 10799338887, 10799338896, 10799338905, 10799338914, 10799338923, 10799338932, 10799338941, 10799338951, 10799338960, 10799338969, 10799338978, 10799338987, 10799338996, 10799339001, 10799339010, 10799339019, 10799339032, 10799339041, 10799339050, 10799339060, 10799339069, 10799339078, 10799339088, 10799339096, 10799339105, 10799339114, 10799339123, 10799339132, 10799339142, 10799339152, 10799339161, 10799339171, 10799339180, 10799339189, 10799339198, 10799339209, 10799339218, 10799339227, 10799339237, 10799339246, 10799339255, 10799339264, 10799339273, 10799339281, 10799339291, 10799339300, 10799339309, 10799339317, 10799339326, 10799339335, 10799339344, 10799339354, 10799339361, 10799339370, 10799339379, 10799339388, 10799339396, 10799339407, 10799339418, 10799346928, 10799346935, 10799346942, 10799346949, 10799346956, 10799346962, 10799346969, 10799346976, 10799346983, 10799346990, 10799346997, 10799347004, 10799347012, 10799347018, 10799347026, 10799347033, 10799347039, 10799347045, 10799347051, 10799347057, 10799347063, 10799347069, 10799347075, 10799347080, 10799347086, 10799347092, 10799347098, 10799347104, 10799347110, 10799347116, 10799347122, 10799347128, 10799347134, 10799347140, 10799347145, 10799347151, 10799347157, 10799347163, 10799339426, 10799339435, 10799339443, 10799339453, 10799339462, 10799339471, 10799339479, 10799339488, 10799339497, 10799339505, 10799339516, 10799339526, 10799339534, 10799339541, 10799339548, 10799339556, 10799339562, 10799339568, 10799339577, 10799339586, 10799339595, 10799339604, 10799339612, 10799339620, 10799339629, 10799339637, 10799339646, 10799339654, 10799339662, 10799339682, 10799339689, 10799339706, 10799339716, 10799339724, 10799339734, 10799339744, 10799339749, 10799339754, 10799339761, 10799339771, 10799339780, 10799339788, 10799339797, 10799339807, 10799339816, 10799339826, 10799339840, 10799339851, 10799339861, 10799339870, 10799339881, 10799339890, 10799339899, 10799339907, 10799339917, 10799339926, 10799339935, 10799339943, 10799339950, 10799339960, 10799339969, 10799339976, 10799339985, 10799339991, 10799340000, 10799340009, 10799340018, 10799340028, 10799340036, 10799340045, 10799340054, 10799340060, 10799340069, 10799340076, 10799340078, 10799340081, 10799340087, 10799340095, 10799340101, 10799340106, 10799340113, 10799340120, 10799340128, 10799340138, 10799340144, 10799340149, 10799340154, 10799340158, 10799340164, 10799340169, 10799340176, 10799340181, 10799340186, 10799340190, 10799340194, 10799340199, 10799340204, 10799340210, 10799340214, 10799340218, 10799340223, 10799340227, 10799340232, 10799340237, 10799340244, 10799340252, 10799340259, 10799340266, 10799340274, 10799340281, 10799340288, 10799340295, 10799340302, 10799340309, 10799340318, 10799340327, 10799340333, 10799340341, 10799340348, 10799340357, 10799340364, 10799340373, 10799340383, 10799340390, 10799340398, 10799340407, 10799340413, 10799340421, 10799340430, 10799340437, 10799340444, 10799340453, 10799340463, 10799340473, 10799340483, 10799340492, 10799340501, 10799340510, 10799340517, 10799340525, 10799340533, 10799340542, 10799340549, 10799340556, 10799340564, 10799340573, 10799340580, 10799340589, 10799340597, 10799340606, 10799340614, 10799340622, 10799340630, 10799340640, 10799340650, 10799340660, 10799340667, 10799340675, 10799340682, 10799340690, 10799340699, 10799340709, 10799340717, 10799340727, 10799340732, 10799340739, 10799340747, 10799340765, 10799340771, 10799340778, 10799340785, 10799340794, 10799340802, 10799340810, 10799340819, 10799340828, 10799340838, 10799340847, 10799340856, 10799340865, 10799340874, 10799340883, 10799340892, 10799340901, 10799340910, 10799340919, 10799340925, 10799340936, 10799340948, 10799340956, 10799340966, 10799340974, 10799340983, 10799340989, 10799340996, 10799341003, 10799341011, 10799341019, 10799341027, 10799341035, 10799341044, 10799341052, 10799341059, 10799341066, 10799341075, 10799341083, 10799341090, 10799341098, 10799341106, 10799341114, 10799341124, 10799341131, 10799341136, 10799341141, 10799341147, 10799341155, 10799341166, 10799341174, 10799341183, 10799341193, 10799341202, 10799341211, 10799341218, 10799341226, 10799341233, 10799341241, 10799341246, 10799341252, 10799341259, 10799341266, 10799341273, 10799341280, 10799341286, 10799341295, 10799341303, 10799341311, 10799341318, 10799341326, 10799341335, 10799341343, 10799341351, 10799341358, 10799341367, 10799341375, 10799341384, 10799341393, 10799341402, 10799341409, 10799341418, 10799341428, 10799341435, 10799341444, 10799341456, 10799341461, 10799341469, 10799341477, 10799341485, 10799341494, 10799341502, 10799341512, 10799341520, 10799341529, 10799341538, 10799341546, 10799341556, 10799341565, 10799341573, 10799341580, 10799341589, 10799341598, 10799341607, 10799341615, 10799341626, 10799341635, 10799341644, 10799341653, 10799341662, 10799341670, 10799341677, 10799341686, 10799341695, 10799341704, 10799341713, 10799341725, 10799341734, 10799341743, 10799341752, 10799341760, 10799341769, 10799341777, 10799341785, 10799341795, 10799341802, 10799341812, 10799341820, 10799341829, 10799341846, 10799341856, 10799341865, 10799341875, 10799341883, 10799341893, 10799341903, 10799341911, 10799341917, 10799341927, 10799341936, 10799341944, 10799341954, 10799341962, 10799341972, 10799341981, 10799341982, 10799341983, 10799341984, 10799341990, 10799341996, 10799342006, 10799342014, 10799342021, 10799342030, 10799342039, 10799342049, 10799342057, 10799342065, 10799342073, 10799342083, 10799342092, 10799342101, 10799342108, 10799342117, 10799342125, 10799342134, 10799342143, 10799342153, 10799342162, 10799342171, 10799342180, 10799342189, 10799342198, 10799342206, 10799342215, 10799342223, 10799342232, 10799342241, 10799342250, 10799342259, 10799342269, 10799342277, 10799342285, 10799342295, 10799342303, 10799342313, 10799342321, 10799342331, 10799342340, 10799342349, 10799342358, 10799342367, 10799342376, 10799342385, 10799342394, 10799342402, 10799342410, 10799342420, 10799342429, 10799342439, 10799342448, 10799342457, 10799342466, 10799342476, 10799342485, 10799342492, 10799342501, 10799342511, 10799342521, 10799342530, 10799342541, 10799342550, 10799342557, 10799342565, 10799342574, 10799342583, 10799342601, 10799342611, 10799342620, 10799342629, 10799342634, 10799342642, 10799342651, 10799342661, 10799342670, 10799342678, 10799342687, 10799342696, 10799342704, 10799342714, 10799342722, 10799342732, 10799342743, 10799342749, 10799342759, 10799342769, 10799342777, 10799342787, 10799342796, 10799342805, 10799342815, 10799342823, 10799342831, 10799342840, 10799342849, 10799342858, 10799342867, 10799342876, 10799342886, 10799342895, 10799342905, 10799342915, 10799342923, 10799342932, 10799342941, 10799342949, 10799342959, 10799342968, 10799342978, 10799342985, 10799342995, 10799343004, 10799343013, 10799343022, 10799343026, 10799343036, 10799343049, 10799343059, 10799343068, 10799343075, 10799343084, 10799343093, 10799343104, 10799343115, 10799343124, 10799343133, 10799343144, 10799343151, 10799343159, 10799343170, 10799343180, 10799343188, 10799343206, 10799343215, 10799343223, 10799343232, 10799343241, 10799343249, 10799343258, 10799343267, 10799343276, 10799343286, 10799343294, 10799343303, 10799343312, 10799343321, 10799343330, 10799343339, 10799343348, 10799343357, 10799343366, 10799343375, 10799343384, 10799343393, 10799343402, 10799343412, 10799343420, 10799343429, 10799343438, 10799343447, 10799343456, 10799343466, 10799343478, 10799343487, 10799343496, 10799343505, 10799343513, 10799343524, 10799343533, 10799343542, 10799343550, 10799343558, 10799343568, 10799343577, 10799343585, 10799343594, 10799343603, 10799343611, 10799343620, 10799343629, 10799343637, 10799343644, 10799343652, 10799343661, 10799343671, 10799343678, 10799343687, 10799343696, 10799343705, 10799343714, 10799343723, 10799343732, 10799343740, 10799343747, 10799343756, 10799343767, 10799343778, 10799343786, 10799343793, 10799343802, 10799343811, 10799343820, 10799343829, 10799343838, 10799343847, 10799343855, 10799343863, 10799343871, 10799343880, 10799343889, 10799343898, 10799343907, 10799343916, 10799343925, 10799343932, 10799343940, 10799343950, 10799343958, 10799343968, 10799343976, 10799343985, 10799343994, 10799344003, 10799344012, 10799344022, 10799344031, 10799344039, 10799344048, 10799344058, 10799344064, 10799344073, 10799344087, 10799344094, 10799344103, 10799344112, 10799344121, 10799344131, 10799344140, 10799344149, 10799344158, 10799344167, 10799344176, 10799344185, 10799344194, 10799344204, 10799344214, 10799344223, 10799344232, 10799344241, 10799344250, 10799344259, 10799344267, 10799344276, 10799344285, 10799344297, 10799344307, 10799344315, 10799344323, 10799344330, 10799344340, 10799344349, 10799344357, 10799344365, 10799344371, 10799344376, 10799344383, 10799344388, 10799344396, 10799344405, 10799344412, 10799344419, 10799344428, 10799344437, 10799344447, 10799344456, 10799344465, 10799344474, 10799344484, 10799344495, 10799344506, 10799344511, 10799344517, 10799344526, 10799344535, 10799344543, 10799344553, 10799344559, 10799344565, 10799344570, 10799344577, 10799344585, 10799344592, 10799344603, 10799344613, 10799344621, 10799344629, 10799344636, 10799344644, 10799344653, 10799344662, 10799344671, 10799344680, 10799344688, 10799344697, 10799344706, 10799344714, 10799344723, 10799344732, 10799344741, 10799344749, 10799344758, 10799344768, 10799344778, 10799344788, 10799344795, 10799344807, 10799344814, 10799344829, 10799344834, 10799344842, 10799344851, 10799344860, 10799344875, 10799344884, 10799344893, 10799344901, 10799344910, 10799344919, 10799344928, 10799344937, 10799344946, 10799344955, 10799344965, 10799344974, 10799344983, 10799344992, 10799345001, 10799345011, 10799345021, 10799345031, 10799345040, 10799345048, 10799345057, 10799345065, 10799345076, 10799345088, 10799345097, 10799345106, 10799345113, 10799345122, 10799345131, 10799345140, 10799345149, 10799345156, 10799345166, 10799345174, 10799345182, 10799345191, 10799345200, 10799345207, 10799345217, 10799345227, 10799345236, 10799345245, 10799345254, 10799345263, 10799345272, 10799345281, 10799345289, 10799345297, 10799345304, 10799345312, 10799345321, 10799345330, 10799345336, 10799345345, 10799345351, 10799345378, 10799345388, 10799345395, 10799345406, 10799345413, 10799345421, 10799345430, 10799345439, 10799345448, 10799345458, 10799345466, 10799345475, 10799345483, 10799345494, 10799345502, 10799345511, 10799345520, 10799345529, 10799345538, 10799345549, 10799345558, 10799345566, 10799345575, 10799345583, 10799345593, 10799345602, 10799345611, 10799345619, 10799345629, 10799345638, 10799345647, 10799345656, 10799345665, 10799345674, 10799345683, 10799345693, 10799345702, 10799345710, 10799345719, 10799345727, 10799345737, 10799345746, 10799345755, 10799345764, 10799345772, 10799345781, 10799345799, 10799345808, 10799345816, 10799345826, 10799345835, 10799345844, 10799345854, 10799345863, 10799345873, 10799345882, 10799345891, 10799345900, 10799345911, 10799345920, 10799345929, 10799345938, 10799345947, 10799345956, 10799345965, 10799345973, 10799345982, 10799345990, 10799346000, 10799346009, 10799346017, 10799346027, 10799346036, 10799346046, 10799346055, 10799346064, 10799346072, 10799346082, 10799346092, 10799346100, 10799346109, 10799346117, 10799346125, 10799346133, 10799346141, 10799346148, 10799346155, 10799346159, 10799346165, 10799346173, 10799346181, 10799346189, 10799346196, 10799346204, 10799346212, 10799346222, 10799346229, 10799346237, 10799346245, 10799346253, 10799346261, 10799346269, 10799346278, 10799346291, 10799346298, 10799346309, 10799346316, 10799346324, 10799346332, 10799346338, 10799346345, 10799346352, 10799346359, 10799346365, 10799346373, 10799346383, 10799346388, 10799346395, 10799346405, 10799346408, 10799345790, 10799346418, 10799346424, 10799346431, 10799346436, 10799346443, 10799346450, 10799346457, 10799346463, 10799346471, 10799346478, 10799346485, 10799346491, 10799346498, 10799346505, 10799346511, 10799346520, 10799346528, 10799346535, 10799346542, 10799346550, 10799346564, 10799346571, 10799346578, 10799346585, 10799346592, 10799346599, 10799346605, 10799346612, 10799346619, 10799346626, 10799346633, 10799346640, 10799346647, 10799346654, 10799346664, 10799346670, 10799346677, 10799346684, 10799346691, 10799346698, 10799346705, 10799346713, 10799346719, 10799346726, 10799346733, 10799346740, 10799346747, 10799346754, 10799346761, 10799346768, 10799346775, 10799346782, 10799346789, 10799346796, 10799346803, 10799346809, 10799346816, 10799346823, 10799346830, 10799346834, 10799346843, 10799346850, 10799346858, 10799346862, 10799346872, 10799346880, 10799346886, 10799346893, 10799346900, 10799346907, 10799346914, 10799346921], 'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/queuesdevaigeneratealbumdto/3bg3qsjsqu-f20oa2rorv9la.json', 'aiMetadata': {'photoIds': [], 'focus': [], 'personIds': [], 'subjects': [], 'density': 3}, 'conditionId': 'AAD_46227780_4b274e0f-de2f-4f24-a4b9-5c743a7f08a9.85.284', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}

    # _input_request =  {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310, 'projectId': 46229128,
    #  'userId': 576349956, 'userJobId': 1121483286,
    #  'base_url': 'ptstorage_32://pictures/46/229/46229128/hbltfcpcopx67ta3tc',
    #  'photos': [10799708360, 10799708362, 10799708364, 10799708366, 10799708368, 10799708370, 10799708372, 10799708374,
    #             10799708376, 10799708378, 10799708380, 10799708381, 10799708383, 10799708386, 10799708388, 10799708390,
    #             10799708393, 10799708396, 10799708398, 10799708400, 10799708402, 10799708404, 10799708406, 10799708408,
    #             10799708409, 10799708411, 10799708413, 10799708415, 10799708417, 10799708419, 10799708421, 10799708423,
    #             10799708425, 10799708427, 10799708429, 10799708431, 10799708433, 10799708435, 10799708437, 10799708439,
    #             10799708441, 10799708443, 10799708445, 10799708447, 10799708449, 10799708451, 10799708453, 10799708455,
    #             10799708457, 10799708459, 10799708461, 10799708463, 10799708465, 10799708467, 10799708469, 10799708471,
    #             10799708473, 10799708475, 10799708477, 10799708479, 10799708481, 10799708483, 10799708485, 10799708487,
    #             10799708489, 10799708491, 10799708493, 10799708495, 10799708497, 10799708499, 10799708500, 10799708502,
    #             10799708504, 10799708506, 10799708508, 10799708510, 10799708512, 10799708514, 10799708516, 10799708518,
    #             10799708520, 10799708522, 10799708524, 10799708526, 10799708528, 10799708530, 10799708532, 10799708534,
    #             10799708536, 10799708538, 10799708540, 10799708541, 10799708543, 10799708545, 10799708547, 10799708550,
    #             10799708552, 10799708554, 10799708556, 10799708558, 10799708560, 10799708562, 10799708564, 10799708566,
    #             10799708568, 10799708570, 10799708572, 10799708574, 10799708576, 10799708578, 10799708580, 10799708582,
    #             10799708584, 10799708586, 10799708588, 10799708590, 10799708592, 10799708594, 10799708596, 10799708598,
    #             10799708600, 10799708602, 10799708604, 10799708606, 10799708608, 10799708610, 10799708612, 10799708614,
    #             10799708616, 10799708618, 10799708620, 10799708622, 10799708624, 10799708626, 10799708628, 10799708630,
    #             10799708632, 10799708634, 10799708636, 10799708638, 10799708640, 10799708641, 10799708643, 10799708645,
    #             10799708647, 10799708649, 10799708651, 10799708652, 10799708654, 10799708656, 10799708658, 10799708660,
    #             10799708661, 10799708662, 10799708665, 10799708667, 10799708669, 10799708671, 10799708673, 10799708675,
    #             10799708677, 10799708678, 10799708680, 10799708683, 10799708685, 10799708687, 10799708689, 10799708690,
    #             10799708692, 10799708694, 10799708696, 10799708698, 10799708700, 10799708702, 10799708704, 10799708706,
    #             10799708708, 10799708710, 10799708712, 10799708714, 10799708716, 10799708718, 10799708720, 10799708722,
    #             10799708724, 10799708726, 10799708728, 10799708730, 10799708732, 10799708734, 10799708737, 10799708739,
    #             10799708741, 10799708743, 10799708745, 10799708747, 10799708749, 10799708751, 10799708753, 10799708755,
    #             10799708757, 10799708759, 10799708761, 10799708763, 10799708765, 10799708767, 10799708769, 10799708771,
    #             10799708773, 10799708775, 10799708777, 10799708779, 10799708781, 10799708783, 10799708785, 10799708787,
    #             10799708789, 10799708791, 10799708793, 10799708795, 10799708797, 10799708799, 10799708801, 10799708803,
    #             10799708805, 10799708807, 10799708809, 10799708811, 10799708813, 10799708815, 10799708817, 10799708819,
    #             10799708821, 10799708823, 10799708825, 10799708827, 10799708829, 10799708831, 10799708833, 10799708835,
    #             10799708837, 10799708839, 10799708841, 10799708843, 10799708845, 10799708847, 10799708849, 10799708851,
    #             10799708853, 10799708855, 10799708857, 10799708859, 10799708861, 10799708863, 10799708865, 10799708867,
    #             10799708869, 10799708871, 10799708872, 10799708873, 10799708875, 10799708877, 10799708879, 10799708881,
    #             10799708883, 10799708890, 10799708892, 10799708893, 10799708894, 10799708896, 10799708897, 10799708898,
    #             10799708901, 10799708903, 10799708905, 10799708907, 10799708909, 10799708911, 10799708913, 10799708914,
    #             10799708916, 10799708918, 10799708920, 10799708922, 10799708927, 10799708929, 10799708931, 10799708933,
    #             10799708935, 10799708937, 10799708939, 10799709142, 10799709144, 10799709146, 10799709147, 10799709149,
    #             10799709151, 10799709154, 10799709156, 10799709158, 10799709160, 10799709162, 10799709164, 10799709166,
    #             10799709168, 10799709170, 10799709172, 10799709174, 10799709176, 10799709177, 10799709178, 10799709179,
    #             10799709180, 10799709181, 10799709183, 10799709184, 10799709185, 10799709186, 10799709188, 10799709190,
    #             10799709192, 10799709194, 10799709196, 10799709198, 10799709200, 10799709202, 10799709204, 10799709206,
    #             10799709208, 10799709210, 10799709212, 10799709214, 10799709217, 10799709219, 10799709221, 10799709222,
    #             10799709224, 10799709226, 10799709227, 10799709228, 10799709230, 10799709232, 10799709237, 10799709239,
    #             10799709241, 10799709243, 10799709245, 10799709247, 10799709251, 10799709253, 10799709255, 10799709257,
    #             10799709259, 10799709261, 10799709263, 10799709265, 10799709267, 10799709269, 10799709272, 10799709275,
    #             10799709277, 10799709278, 10799709280, 10799709282, 10799709284, 10799709286, 10799709289, 10799709291,
    #             10799709293, 10799709296, 10799709299, 10799709301, 10799709303, 10799709305, 10799709307, 10799709309,
    #             10799709311, 10799709313, 10799709315, 10799709317, 10799709319, 10799709321, 10799709323, 10799709325,
    #             10799709327, 10799709331, 10799709333, 10799709335, 10799709337, 10799709339, 10799709341, 10799709343,
    #             10799709344, 10799709345, 10799709346, 10799709347, 10799709348, 10799709349, 10799709351, 10799709353,
    #             10799709355, 10799709357, 10799709359, 10799709361, 10799709363, 10799709365, 10799709367, 10799709372,
    #             10799709374, 10799709376, 10799709377, 10799709379, 10799709382, 10799709383, 10799709385, 10799709387,
    #             10799709389, 10799709391, 10799709393, 10799709395, 10799709397, 10799709399, 10799709402, 10799709404,
    #             10799709406, 10799709408, 10799709410, 10799709412, 10799709414, 10799709416, 10799709418, 10799709420,
    #             10799709422, 10799709424, 10799709426, 10799709428, 10799709430, 10799709432, 10799709434, 10799709436,
    #             10799709438, 10799709440, 10799709442, 10799709444, 10799709446, 10799709448, 10799709450, 10799709452,
    #             10799709454, 10799709456, 10799709458, 10799709460, 10799709462, 10799709464, 10799709466, 10799709468,
    #             10799709470, 10799709472, 10799709474, 10799709476, 10799709478, 10799709480, 10799709482, 10799709484,
    #             10799709486, 10799709488, 10799709490, 10799709492, 10799709494, 10799709496, 10799709498, 10799709500,
    #             10799709502, 10799709504, 10799709506, 10799709508, 10799709510, 10799709512, 10799709514, 10799709515,
    #             10799709516, 10799709518, 10799709520, 10799709522, 10799709524, 10799709526, 10799709528, 10799709530,
    #             10799709532, 10799709534, 10799709536, 10799709538, 10799709540, 10799709542, 10799709544, 10799709546,
    #             10799709548, 10799709550, 10799709552, 10799709554, 10799709556, 10799709558, 10799709560, 10799709562,
    #             10799709564, 10799709566, 10799709568, 10799709570, 10799709572, 10799709574, 10799709576, 10799709578,
    #             10799709580, 10799709582, 10799709584, 10799709586, 10799709588, 10799709590, 10799709592, 10799709594,
    #             10799709596, 10799709599, 10799709600, 10799709602, 10799709604, 10799709606, 10799709607, 10799709609,
    #             10799709611, 10799709613, 10799709615, 10799709617, 10799709619, 10799709621, 10799709623, 10799709625,
    #             10799709627, 10799709629, 10799708941, 10799708943, 10799708945, 10799708947, 10799708949, 10799708951,
    #             10799708953, 10799708955, 10799708957, 10799708960, 10799708962, 10799708964, 10799708966, 10799708968,
    #             10799708970, 10799708972, 10799708974, 10799708976, 10799708978, 10799708980, 10799708982, 10799708983,
    #             10799708984, 10799708986, 10799708988, 10799708990, 10799708992, 10799708994, 10799708996, 10799708998,
    #             10799709000, 10799709002, 10799709004, 10799709006, 10799709008, 10799709010, 10799709012, 10799709014,
    #             10799709022, 10799709025, 10799709026, 10799709028, 10799709030, 10799709032, 10799709034, 10799709036,
    #             10799709038, 10799709040, 10799709042, 10799709044, 10799709046, 10799709048, 10799709050, 10799709052,
    #             10799709054, 10799709056, 10799709058, 10799709063, 10799709065, 10799709067, 10799709069, 10799709071,
    #             10799709073, 10799709074, 10799709075, 10799709076, 10799709078, 10799709080, 10799709083, 10799709084,
    #             10799709085, 10799709087, 10799709089, 10799709091, 10799709093, 10799709095, 10799709097, 10799709099,
    #             10799709101, 10799709103, 10799709105, 10799709107, 10799709109, 10799709111, 10799709113, 10799709115,
    #             10799709117, 10799709120, 10799709121, 10799709124, 10799709125, 10799709127, 10799709129, 10799709131,
    #             10799709133, 10799709135, 10799709138, 10799709140, 10799709631, 10799709633, 10799709635, 10799709637,
    #             10799709639, 10799709641, 10799709643, 10799709645, 10799709647, 10799709649, 10799709651, 10799709653,
    #             10799709655, 10799709657, 10799709659, 10799709661, 10799709663, 10799709665, 10799709667, 10799709669,
    #             10799709671, 10799709673, 10799709675, 10799709677, 10799709679, 10799709681, 10799709683, 10799709685,
    #             10799709687, 10799709689, 10799709691, 10799709693, 10799709695, 10799709697, 10799709699, 10799709701,
    #             10799709703, 10799709705, 10799709707, 10799709709, 10799709711, 10799709713, 10799709715, 10799709717,
    #             10799709719, 10799709721, 10799709723, 10799709725, 10799709727, 10799709729, 10799709731, 10799709733,
    #             10799709735, 10799709737, 10799709739, 10799709741, 10799709743, 10799709745, 10799709747, 10799709748,
    #             10799709750, 10799709753, 10799709754, 10799709756, 10799709758, 10799709760, 10799709762, 10799709764,
    #             10799709766, 10799709768, 10799709770, 10799709772, 10799709774, 10799709776, 10799709778, 10799709780,
    #             10799709782, 10799709784, 10799709786, 10799709788, 10799709790, 10799709792, 10799709794, 10799709795,
    #             10799709797, 10799709799, 10799709801, 10799709803, 10799709805, 10799709807, 10799709809, 10799709811,
    #             10799709813, 10799709815, 10799709817, 10799709819, 10799709822, 10799709824, 10799709826, 10799709828,
    #             10799709830, 10799709832, 10799709834, 10799709836, 10799709838, 10799709840, 10799709842, 10799709844,
    #             10799709846, 10799709848, 10799709850, 10799709852, 10799709854, 10799709856, 10799709858, 10799709860,
    #             10799709862, 10799709864, 10799709866, 10799709868, 10799709870, 10799709872, 10799709874, 10799709875,
    #             10799709876, 10799709877, 10799709878, 10799709879, 10799709880, 10799709881, 10799709882, 10799709883,
    #             10799709884, 10799709885, 10799709886, 10799709887, 10799709888, 10799709889, 10799709890, 10799709891,
    #             10799709892, 10799709893, 10799709894, 10799709895, 10799709896, 10799709897, 10799709898, 10799709899,
    #             10799709900, 10799709901, 10799709902, 10799709903, 10799709904, 10799709905, 10799709906, 10799709907,
    #             10799709908, 10799709909, 10799709910, 10799709911, 10799709912, 10799709913, 10799709914, 10799709915,
    #             10799709916, 10799709917, 10799709918, 10799709919, 10799709920, 10799709921, 10799709922, 10799709923,
    #             10799709924, 10799709925, 10799709926, 10799709927, 10799709928, 10799709929, 10799709930, 10799709931,
    #             10799709932, 10799709933, 10799709934, 10799709935, 10799709936, 10799709937, 10799709938, 10799709939,
    #             10799709940, 10799709941, 10799709942, 10799709943, 10799709944, 10799709945, 10799709946, 10799709947,
    #             10799709948, 10799709949, 10799709950, 10799709951, 10799709952, 10799709953, 10799709954, 10799709955,
    #             10799709956, 10799709957, 10799709958, 10799709959, 10799709960, 10799709961, 10799709962, 10799709963,
    #             10799709964, 10799709965, 10799709966, 10799709967, 10799709968, 10799709969, 10799709970, 10799709971,
    #             10799709972, 10799709973, 10799709974, 10799709975, 10799709976, 10799709977, 10799709978, 10799709979,
    #             10799709980, 10799709981, 10799709982, 10799709983, 10799709984, 10799709985, 10799709986, 10799709987,
    #             10799709988, 10799709989, 10799709990, 10799709991, 10799709992, 10799709993, 10799709994, 10799709995,
    #             10799709996, 10799709997, 10799709998, 10799709999, 10799710000, 10799710001, 10799710002, 10799710003,
    #             10799710004, 10799710005, 10799710006, 10799710007, 10799710008, 10799710009, 10799710010, 10799710011,
    #             10799710012, 10799710013, 10799710014, 10799710015, 10799710016, 10799710017, 10799710018, 10799710019,
    #             10799710020, 10799710021, 10799710022, 10799710023, 10799710024, 10799710025, 10799710026, 10799710027,
    #             10799710028, 10799710029, 10799710030, 10799710031, 10799710032, 10799710033, 10799710034, 10799710035,
    #             10799710036, 10799710037, 10799710038, 10799710039, 10799710040, 10799710041, 10799710042, 10799710043,
    #             10799710044, 10799710045, 10799710046, 10799710047, 10799710048, 10799710049, 10799710050, 10799710051,
    #             10799710052, 10799710053, 10799710054, 10799710055, 10799710056, 10799710057, 10799710058, 10799710059,
    #             10799710060, 10799710061, 10799710062, 10799710063, 10799710064, 10799710065, 10799710066, 10799710067,
    #             10799710068, 10799710069, 10799710070, 10799710071, 10799710072, 10799710073, 10799710074, 10799710075,
    #             10799710076, 10799710077, 10799710078, 10799710079, 10799710080, 10799710081, 10799710082, 10799710083,
    #             10799710084, 10799710085, 10799710086, 10799710087, 10799710088, 10799710089, 10799710090, 10799710091,
    #             10799710092, 10799710093, 10799710094, 10799710095, 10799710096, 10799710097, 10799710098, 10799710099,
    #             10799710100, 10799710101, 10799710102, 10799710104, 10799710105, 10799710106, 10799710107, 10799710108,
    #             10799710109, 10799710110, 10799710111, 10799710112, 10799710113, 10799710114, 10799710115, 10799710116,
    #             10799710117, 10799710118, 10799710119, 10799710120, 10799710121, 10799710122, 10799710123, 10799710124,
    #             10799710125, 10799710126, 10799710127, 10799710128, 10799710129, 10799710130, 10799710131, 10799710132,
    #             10799710133, 10799710134, 10799710135, 10799710136, 10799710137, 10799710138, 10799710139, 10799710140,
    #             10799710141, 10799710142, 10799710143, 10799710144, 10799710145, 10799710146, 10799710147, 10799710148,
    #             10799710149, 10799710150, 10799710151, 10799710152, 10799710153, 10799710154, 10799710155, 10799710156,
    #             10799710157, 10799710158, 10799710159, 10799710160, 10799710161, 10799710162, 10799710163, 10799710164,
    #             10799710165, 10799710166, 10799710167, 10799710168, 10799710169, 10799710170, 10799710171, 10799710172,
    #             10799710173, 10799710174, 10799710175, 10799710176, 10799710177, 10799710178, 10799710179, 10799710180,
    #             10799710181, 10799710182, 10799710183, 10799710184, 10799710185, 10799710186, 10799710187, 10799710188,
    #             10799710189, 10799710190, 10799710191, 10799710192, 10799710193, 10799710194, 10799710195, 10799710196,
    #             10799710197, 10799710198, 10799710199, 10799710200, 10799710201, 10799710202, 10799710203, 10799710204,
    #             10799710205, 10799710206, 10799710207, 10799710208, 10799710209, 10799710210, 10799710211, 10799710212,
    #             10799710213, 10799710214, 10799710215, 10799710216, 10799710217, 10799710218, 10799710219, 10799710220,
    #             10799710221, 10799710222, 10799710223, 10799710224, 10799710225, 10799710226, 10799710227, 10799710228,
    #             10799710229, 10799710230, 10799710231, 10799710232, 10799710233, 10799710234, 10799710235, 10799710236,
    #             10799710237, 10799710238, 10799710239, 10799710240, 10799710241, 10799710242, 10799710243, 10799710244,
    #             10799710245, 10799710246, 10799710247, 10799710248, 10799710249, 10799710250, 10799710251, 10799710252,
    #             10799710253, 10799710254, 10799710255, 10799710256], 'projectCategory': 1, 'compositionPackageId': -1,
    #  'designInfo': None,
    #  'designInfoTempLocation': 'pictures/temp/queuesdevaigeneratealbumdto/flmotjte30kjlrtggsojhcq8.json',
    #  'aiMetadata': {'photoIds': [], 'focus': [], 'personIds': [], 'subjects': [], 'density': 3},
    #  'conditionId': 'AAD_46229128_1c2b05fd-be76-4664-9964-16e3a68095c9.137.111', 'timedOut': False,
    #  'dependencyDeleted': False, 'retryCount': 0}


    # _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310, 'projectId': 46229129,
    #  'userId': 576349956, 'userJobId': 1121483286,
    #  'base_url': 'ptstorage_32://pictures/46/229/46229129/66cam5evxasipdko4j',
    #  'photos': [10799708185, 10799708186, 10799708187, 10799708188, 10799708189, 10799708190, 10799708191, 10799708192,
    #             10799708193, 10799708194, 10799708195, 10799708196, 10799708197, 10799708198, 10799708199, 10799708200,
    #             10799708201, 10799708202, 10799708203, 10799708204, 10799708205, 10799708206, 10799708207, 10799708208,
    #             10799708209, 10799708210, 10799708211, 10799708212, 10799708213, 10799708214, 10799708215, 10799708216,
    #             10799708217, 10799708218, 10799708219, 10799708220, 10799708221, 10799708222, 10799708223, 10799708224,
    #             10799708225, 10799708226, 10799708227, 10799708228, 10799708229, 10799708230, 10799708231, 10799708232,
    #             10799708233, 10799708234, 10799708235, 10799708236, 10799708237, 10799708238, 10799708239, 10799708240,
    #             10799708241, 10799708242, 10799708243, 10799708244, 10799708245, 10799708246, 10799708247, 10799708248,
    #             10799708249, 10799708250, 10799708251, 10799708252, 10799708253, 10799708254, 10799708255, 10799708256,
    #             10799708257, 10799708258, 10799708259, 10799708260, 10799708261, 10799708262, 10799708263, 10799708264,
    #             10799708265, 10799708266, 10799708267, 10799708268, 10799708269, 10799708270, 10799708271, 10799708272,
    #             10799708273, 10799708274, 10799708275, 10799708276, 10799708277, 10799708278, 10799708279, 10799708280,
    #             10799708281, 10799708282, 10799708283, 10799708284, 10799708285, 10799708286, 10799708287, 10799708288,
    #             10799708289, 10799708290, 10799708291, 10799708292, 10799708293, 10799708294, 10799708295, 10799708296,
    #             10799708297, 10799708298, 10799708299, 10799708300, 10799708301, 10799708302, 10799708303, 10799708304,
    #             10799708305, 10799708306, 10799708307, 10799708308, 10799708309, 10799708310, 10799708311, 10799708312,
    #             10799708313, 10799708314, 10799708315, 10799708316, 10799708317, 10799708318, 10799708319, 10799708320,
    #             10799708321, 10799708322, 10799708323, 10799708324, 10799708325, 10799708326, 10799708327, 10799708328,
    #             10799708329, 10799708330, 10799708331, 10799708332, 10799708333, 10799708334, 10799708335, 10799708336,
    #             10799708337, 10799708338, 10799708339, 10799708340, 10799708341, 10799708342, 10799708343, 10799708344,
    #             10799708345, 10799708346, 10799708347, 10799708348, 10799708349, 10799708350, 10799708351, 10799708352,
    #             10799708353, 10799708354, 10799708355, 10799708356, 10799708357, 10799708358, 10799708359, 10799708361,
    #             10799708363, 10799708365, 10799708367, 10799708369, 10799708371, 10799708373, 10799708375, 10799708377,
    #             10799708379, 10799708382, 10799708384, 10799708385, 10799708387, 10799708389, 10799708391, 10799708392,
    #             10799708394, 10799708395, 10799708397, 10799708399, 10799708401, 10799708403, 10799708405, 10799708407,
    #             10799708410, 10799708412, 10799708414, 10799708416, 10799708418, 10799708420, 10799708422, 10799708424,
    #             10799708426, 10799708428, 10799708430, 10799708432, 10799708434, 10799708436, 10799708438, 10799708440,
    #             10799708442, 10799708444, 10799708446, 10799708448, 10799708450, 10799708452, 10799708454, 10799708456,
    #             10799708458, 10799708460, 10799708462, 10799708464, 10799708466, 10799708468, 10799708470, 10799708472,
    #             10799708474, 10799708476, 10799708478, 10799708480, 10799708482, 10799708484, 10799708486, 10799708488,
    #             10799708490, 10799708492, 10799708494, 10799708496, 10799708498, 10799708501, 10799708503, 10799708505,
    #             10799708507, 10799708509, 10799708511, 10799708513, 10799708515, 10799708517, 10799708519, 10799708521,
    #             10799708523, 10799708525, 10799708527, 10799708529, 10799708531, 10799708533, 10799708535, 10799708537,
    #             10799708539, 10799708542, 10799708544, 10799708546, 10799708548, 10799708551, 10799708553, 10799708555,
    #             10799708557, 10799708559, 10799708561, 10799708563, 10799708565, 10799708567, 10799708569, 10799708571,
    #             10799708573, 10799708575, 10799708577, 10799708579, 10799708581, 10799708583, 10799708585, 10799708587,
    #             10799708589, 10799708591, 10799708593, 10799708595, 10799708597, 10799708599, 10799708601, 10799708603,
    #             10799708605, 10799708607, 10799708609, 10799708611, 10799708613, 10799708615, 10799708617, 10799708619,
    #             10799708621, 10799708623, 10799708625, 10799708627, 10799708629, 10799708631, 10799708633, 10799708635,
    #             10799708637, 10799708639, 10799708642, 10799708644, 10799708646, 10799708648, 10799708650, 10799708653,
    #             10799708655, 10799708657, 10799708659, 10799708663, 10799708664, 10799708666, 10799708668, 10799708670,
    #             10799708672, 10799708674, 10799708676, 10799708679, 10799708681, 10799708682, 10799708684, 10799708686,
    #             10799708688, 10799708691, 10799708693, 10799708695, 10799708697, 10799708699, 10799708701, 10799708703,
    #             10799708705, 10799708707, 10799708709, 10799708711, 10799708713, 10799708715, 10799708717, 10799708719,
    #             10799708721, 10799708723, 10799708725, 10799708727, 10799708729, 10799708731, 10799708733, 10799708735,
    #             10799708736, 10799708738, 10799708740, 10799708742, 10799708744, 10799708746, 10799708748, 10799708750,
    #             10799708752, 10799708754, 10799708756, 10799708758, 10799708760, 10799708762, 10799708764, 10799708766,
    #             10799708768, 10799708770, 10799708772, 10799708774, 10799708776, 10799708778, 10799708780, 10799708782,
    #             10799708784, 10799708786, 10799708788, 10799708790, 10799708792, 10799708794, 10799708796, 10799708798,
    #             10799708800, 10799708802, 10799708804, 10799708806, 10799708808, 10799708810, 10799708812, 10799708814,
    #             10799708816, 10799708818, 10799708820, 10799708822, 10799708824, 10799708826, 10799708828, 10799708830,
    #             10799708832, 10799708834, 10799708836, 10799708838, 10799708840, 10799708842, 10799708844, 10799708846,
    #             10799708848, 10799708850, 10799708852, 10799708854, 10799708856, 10799708858, 10799708860, 10799708862,
    #             10799708864, 10799708866, 10799708868, 10799708870, 10799708874, 10799708876, 10799708878, 10799708880,
    #             10799708882, 10799708884, 10799708885, 10799708886, 10799708887, 10799708888, 10799708889, 10799708891,
    #             10799708895, 10799708899, 10799708900, 10799708902, 10799708904, 10799708906, 10799708908, 10799708910,
    #             10799708912, 10799708915, 10799708917, 10799708919, 10799708921, 10799708923, 10799708924, 10799708925,
    #             10799708926, 10799708928, 10799708930, 10799708932, 10799708934, 10799708936, 10799708938, 10799708940,
    #             10799708942, 10799708944, 10799708946, 10799708948, 10799708950, 10799708952, 10799708954, 10799708956,
    #             10799708958, 10799708959, 10799708961, 10799708963, 10799708965, 10799708967, 10799708969, 10799708971,
    #             10799708973, 10799708975, 10799708977, 10799708979, 10799708981, 10799708985, 10799708987, 10799708989,
    #             10799708991, 10799708993, 10799708995, 10799708997, 10799708999, 10799709001, 10799709003, 10799709005,
    #             10799709007, 10799709009, 10799709011, 10799709013, 10799709015, 10799709016, 10799709018, 10799709019,
    #             10799709020, 10799709021, 10799709023, 10799709024, 10799709027, 10799709029, 10799709031, 10799709033,
    #             10799709035, 10799709037, 10799709039, 10799709041, 10799709043, 10799709045, 10799709047, 10799709049,
    #             10799709051, 10799709053, 10799709055, 10799709057, 10799709059, 10799709060, 10799709061, 10799709062,
    #             10799709064, 10799709066, 10799709068, 10799709070, 10799709072, 10799709077, 10799709079, 10799709081,
    #             10799709082, 10799709086, 10799709088, 10799709090, 10799709092, 10799709094, 10799709096, 10799709098,
    #             10799709100, 10799709102, 10799709104, 10799709106, 10799709108, 10799709110, 10799709112, 10799709114,
    #             10799709116, 10799709118, 10799709119, 10799709122, 10799709123, 10799709126, 10799709128, 10799709130,
    #             10799709132, 10799709134, 10799709136, 10799709137, 10799709139, 10799709141, 10799709143, 10799709145,
    #             10799709148, 10799709150, 10799709152, 10799709153, 10799709155, 10799709157, 10799709159, 10799709161,
    #             10799709163, 10799709165, 10799709167, 10799709169, 10799709171, 10799709173, 10799709175, 10799709182,
    #             10799709187, 10799709189, 10799709191, 10799709193, 10799709195, 10799709197, 10799709199, 10799709201,
    #             10799709203, 10799709205, 10799709207, 10799709209, 10799709211, 10799709213, 10799709215, 10799709216,
    #             10799709218, 10799709220, 10799709223, 10799709225, 10799709229, 10799709231, 10799709233, 10799709234,
    #             10799709235, 10799709236, 10799709238, 10799709240, 10799709242, 10799709244, 10799709246, 10799709248,
    #             10799709249, 10799709250, 10799709252, 10799709254, 10799709256, 10799709258, 10799709260, 10799709262,
    #             10799709264, 10799709266, 10799709268, 10799709270, 10799709271, 10799709273, 10799709274, 10799709276,
    #             10799709279, 10799709281, 10799709283, 10799709285, 10799709287, 10799709288, 10799709290, 10799709292,
    #             10799709294, 10799709295, 10799709297, 10799709298, 10799709300, 10799709302, 10799709304, 10799709306,
    #             10799709308, 10799709310, 10799709312, 10799709314, 10799709316, 10799709318, 10799709320, 10799709322,
    #             10799709324, 10799709326, 10799709328, 10799709329, 10799709330, 10799709332, 10799709334, 10799709336,
    #             10799709338, 10799709340, 10799709342, 10799709350, 10799709352, 10799709354, 10799709356, 10799709358,
    #             10799709360, 10799709362, 10799709364, 10799709366, 10799709368, 10799709369, 10799709370, 10799709371,
    #             10799709373, 10799709375, 10799709378, 10799709380, 10799709381, 10799709384, 10799709386, 10799709388,
    #             10799709390, 10799709392, 10799709394, 10799709396, 10799709398, 10799709400, 10799709401, 10799709403,
    #             10799709405, 10799709407, 10799709409, 10799709411, 10799709413, 10799709415, 10799709417, 10799709419,
    #             10799709421, 10799709423, 10799709425, 10799709427, 10799709429, 10799709431, 10799709433, 10799709435,
    #             10799709437, 10799709439, 10799709441, 10799709443, 10799709445, 10799709447, 10799709449, 10799709451,
    #             10799709453, 10799709455, 10799709457, 10799709459, 10799709461, 10799709463, 10799709465, 10799709467,
    #             10799709469, 10799709471, 10799709473, 10799709475, 10799709477, 10799709479, 10799709481, 10799709483,
    #             10799709485, 10799709487, 10799709489, 10799709491, 10799709493, 10799709495, 10799709497, 10799709499,
    #             10799709501, 10799709503, 10799709505, 10799709507, 10799709509, 10799709511, 10799709513, 10799709517,
    #             10799709519, 10799709521, 10799709523, 10799709525, 10799709527, 10799709529, 10799709531, 10799709533,
    #             10799709535, 10799709537, 10799709539, 10799709541, 10799709543, 10799709545, 10799709547, 10799709549,
    #             10799709551, 10799709553, 10799709555, 10799709557, 10799709559, 10799709561, 10799709563, 10799709565,
    #             10799709567, 10799709569, 10799709571, 10799709573, 10799709575, 10799709577, 10799709579, 10799709581,
    #             10799709583, 10799709585, 10799709587, 10799709589, 10799709591, 10799709593, 10799709595, 10799709597,
    #             10799709598, 10799709601, 10799709603, 10799709605, 10799709608, 10799709610, 10799709612, 10799709614,
    #             10799709616, 10799709618, 10799709620, 10799709622, 10799709624, 10799709626, 10799709628, 10799709630,
    #             10799709632, 10799709634, 10799709636, 10799709638, 10799709640, 10799709642, 10799709644, 10799709646,
    #             10799709648, 10799709650, 10799709652, 10799709654, 10799709656, 10799709658, 10799709660, 10799709662,
    #             10799709664, 10799709666, 10799709668, 10799709670, 10799709672, 10799709674, 10799709676, 10799709678,
    #             10799709680, 10799709682, 10799709684, 10799709686, 10799709688, 10799709690, 10799709692, 10799709694,
    #             10799709696, 10799709698, 10799709700, 10799709702, 10799709704, 10799709706, 10799709708, 10799709710,
    #             10799709712, 10799709714, 10799709716, 10799709718, 10799709720, 10799709722, 10799709724, 10799709726,
    #             10799709728, 10799709730, 10799709732, 10799709734, 10799709736, 10799709738, 10799709740, 10799709742,
    #             10799709744, 10799709746, 10799709749, 10799709751, 10799709752, 10799709755, 10799709757, 10799709759,
    #             10799709761, 10799709763, 10799709765, 10799709767, 10799709769, 10799709771, 10799709773, 10799709775,
    #             10799709777, 10799709779, 10799709781, 10799709783, 10799709785, 10799709787, 10799709789, 10799709791,
    #             10799709793, 10799709796, 10799709798, 10799709800, 10799709802, 10799709804, 10799709806, 10799709808,
    #             10799709810, 10799709812, 10799709814, 10799709816, 10799709818, 10799709820, 10799709821, 10799709823,
    #             10799709825, 10799709827, 10799709829, 10799709831, 10799709833, 10799709835, 10799709837, 10799709839,
    #             10799709841, 10799709843, 10799709845, 10799709847, 10799709849, 10799709851, 10799709853, 10799709855,
    #             10799709857, 10799709859, 10799709861, 10799709863, 10799709865, 10799709867, 10799709869, 10799709871,
    #             10799709873], 'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None,
    #  'designInfoTempLocation': 'pictures/temp/queuesdevaigeneratealbumdto/2ezy_wtihuk7ymgwj7am-zm_.json',
    #  'aiMetadata': {'photoIds': [], 'focus': [], 'personIds': [], 'subjects': [], 'density': 3},
    #  'conditionId': 'AAD_46229129_1c2b05fd-be76-4664-9964-16e3a68095c9.220.220', 'timedOut': False,
    #  'dependencyDeleted': False, 'retryCount': 0}

    # _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 493990, 'projectId': 47981912,
    #  'userId': 588345648, 'userJobId': 1144911075,
    #  'base_url': 'ptstorage_1://pictures/47/981/47981912/5q7khaa47itlml28n8',
    #  'photos': [11192656630, 11192656631, 11192656632, 11192656633, 11192656634, 11192656635, 11192656636, 11192656637,
    #             11192656638, 11192656639, 11192656640, 11192656641, 11192656642, 11192656643, 11192656644, 11192656645,
    #             11192656646, 11192656647, 11192656648, 11192656649, 11192656650, 11192656651, 11192656652, 11192656653,
    #             11192656654, 11192656655, 11192656656, 11192656657, 11192656658, 11192656659, 11192656660, 11192656661,
    #             11192656662, 11192656663, 11192656664, 11192656665, 11192656666, 11192656667, 11192656668, 11192656669,
    #             11192656670, 11192656671, 11192656672, 11192656673, 11192656674, 11192656675, 11192656676, 11192656677,
    #             11192656678, 11192656679, 11192656680, 11192656681, 11192656682, 11192656683, 11192656684, 11192656685,
    #             11192656686, 11192656687, 11192656688, 11192656689, 11192656690, 11192656691, 11192656692, 11192656693,
    #             11192656694, 11192656695, 11192656696, 11192656697, 11192656698, 11192656699, 11192656700, 11192656701,
    #             11192656702, 11192656703, 11192656704, 11192656705, 11192656706, 11192656707, 11192656708, 11192656709,
    #             11192656710, 11192656711, 11192656712, 11192656713, 11192656714, 11192656715, 11192656716, 11192656717,
    #             11192656718, 11192656719, 11192656720, 11192656721, 11192656722, 11192656723, 11192656724, 11192656725,
    #             11192656726, 11192656727, 11192656728, 11192656729, 11192656730, 11192656731, 11192656732, 11192656733,
    #             11192656734, 11192656735, 11192656736, 11192656737, 11192656738, 11192656739, 11192656740, 11192656741,
    #             11192656742, 11192656743, 11192656744, 11192656745, 11192656746, 11192656747, 11192656748, 11192656749,
    #             11192656750, 11192656751, 11192656752, 11192656753, 11192656754, 11192656755, 11192656756, 11192656757,
    #             11192656758, 11192656759, 11192656760, 11192656761, 11192656762, 11192656763, 11192656764, 11192656765,
    #             11192656766, 11192656767, 11192656768, 11192656769, 11192656770, 11192656771, 11192656772, 11192656773,
    #             11192656774, 11192656775, 11192656776, 11192656777, 11192656778, 11192656779, 11192656780, 11192656781,
    #             11192656782, 11192656783, 11192656784, 11192656785, 11192656786, 11192656787, 11192656788, 11192656789,
    #             11192656790, 11192656791, 11192656792, 11192656793, 11192656794, 11192656795, 11192656796, 11192656797,
    #             11192656798, 11192656799, 11192656800, 11192656801, 11192656802, 11192656803, 11192656804, 11192656805,
    #             11192656806, 11192656807, 11192656808, 11192656809, 11192656810, 11192656811, 11192656812, 11192656813,
    #             11192656814, 11192656815, 11192656816, 11192656817, 11192656818, 11192656819, 11192656820, 11192656821,
    #             11192656822, 11192656823, 11192656824, 11192656825, 11192656826, 11192656827, 11192656828, 11192656829,
    #             11192656830, 11192656831, 11192656832, 11192656833, 11192656834, 11192656835, 11192656836, 11192656837,
    #             11192656838, 11192656839, 11192656840, 11192656841, 11192656842, 11192656843, 11192656844, 11192656845,
    #             11192656846, 11192656847, 11192656848, 11192656849, 11192656850, 11192656851, 11192656852, 11192656853,
    #             11192656854, 11192656855, 11192656856, 11192656857, 11192656858, 11192656859, 11192656860, 11192656861,
    #             11192656862, 11192656863, 11192656864, 11192656865, 11192656866, 11192656867, 11192656868, 11192656869,
    #             11192656870, 11192656871, 11192656872, 11192656873, 11192656874, 11192656875, 11192656876, 11192656877,
    #             11192656878, 11192656879, 11192656880, 11192656881, 11192656882, 11192656883, 11192656884, 11192656885,
    #             11192656886, 11192656887, 11192656888, 11192656889, 11192656890, 11192656891, 11192656892, 11192656893,
    #             11192656894, 11192656895, 11192656896, 11192656897, 11192656898, 11192656899, 11192656900, 11192656901,
    #             11192656902, 11192656903, 11192656904, 11192656905, 11192656906, 11192656907, 11192656908, 11192656909,
    #             11192656910, 11192656911, 11192656912, 11192656913, 11192656914, 11192656915, 11192656916, 11192656917,
    #             11192656918, 11192656919, 11192656920, 11192656921, 11192656922, 11192656923, 11192656924, 11192656925,
    #             11192656926, 11192656927, 11192656928, 11192656929, 11192656930, 11192656931, 11192656932, 11192656933,
    #             11192656934, 11192656935, 11192656936, 11192656937, 11192656938, 11192656939, 11192656940, 11192656941,
    #             11192656942, 11192656943, 11192656944, 11192656945, 11192656946, 11192656947, 11192656948, 11192656949,
    #             11192656950, 11192656951, 11192656952, 11192656953, 11192656954, 11192656955, 11192656956, 11192656957,
    #             11192656958, 11192656959, 11192656960, 11192656961, 11192656962, 11192656963, 11192656964, 11192656965,
    #             11192656966, 11192656967, 11192656968, 11192656969, 11192656970, 11192656971, 11192656972, 11192656973,
    #             11192656974, 11192656975, 11192656976, 11192656977, 11192656978, 11192656979, 11192656980, 11192656981,
    #             11192656982, 11192656983, 11192656984, 11192656985, 11192656986, 11192656987, 11192656988, 11192656989,
    #             11192656990, 11192656991, 11192656992, 11192656993, 11192656994, 11192656995, 11192656996, 11192656997,
    #             11192656998, 11192656999, 11192657000, 11192657001, 11192657002, 11192657003, 11192657004, 11192657005,
    #             11192657006, 11192657007, 11192657008, 11192657009, 11192657010, 11192657011, 11192657012, 11192657013,
    #             11192657014, 11192657015, 11192657016, 11192657017, 11192657018, 11192657019, 11192657020, 11192657021,
    #             11192657022, 11192657023, 11192657024, 11192657025, 11192657026, 11192657027, 11192657028, 11192657029,
    #             11192657053, 11192657054, 11192657055, 11192657056, 11192657057, 11192657058, 11192657059, 11192657060,
    #             11192657061, 11192657062, 11192657063, 11192657064, 11192657065, 11192657066, 11192657067, 11192657068,
    #             11192657069, 11192657070, 11192657071, 11192657072, 11192657073, 11192657074, 11192657075, 11192657076,
    #             11192657077, 11192657078, 11192657079, 11192657080, 11192657081, 11192657082, 11192657083, 11192657084,
    #             11192657085, 11192657086, 11192657087, 11192657088, 11192657089, 11192657090, 11192657091, 11192657092,
    #             11192657093, 11192657094, 11192657095, 11192657096, 11192657097, 11192657098, 11192657099, 11192657100,
    #             11192657101, 11192657102, 11192657103, 11192657104, 11192657105, 11192657106, 11192657107, 11192657108,
    #             11192657109, 11192657110, 11192657111, 11192657112, 11192657113, 11192657114, 11192657115, 11192657116,
    #             11192657117, 11192657118, 11192657119, 11192657120, 11192657121, 11192657122, 11192657123, 11192657124,
    #             11192657125, 11192657126, 11192657127, 11192657128, 11192657129, 11192657130, 11192657131, 11192657132,
    #             11192657133, 11192657134, 11192657135, 11192657136, 11192657137, 11192657138, 11192657139, 11192657140,
    #             11192657141, 11192657142, 11192657143, 11192657144, 11192657145, 11192657146, 11192657147, 11192657148,
    #             11192657149, 11192657150, 11192657151, 11192657152, 11192657153, 11192657154, 11192657155, 11192657156,
    #             11192657157, 11192657158, 11192657159, 11192657160, 11192657161, 11192657162, 11192657163, 11192657164,
    #             11192657165, 11192657166, 11192657167, 11192657168, 11192657169, 11192657170, 11192657171, 11192657172,
    #             11192657173, 11192657174, 11192657175, 11192657176, 11192657177, 11192657178, 11192657179, 11192657180,
    #             11192657181, 11192657182, 11192657183, 11192657184, 11192657185, 11192657186, 11192657187, 11192657188,
    #             11192657189, 11192657190, 11192657191, 11192657192, 11192657193, 11192657194, 11192657195, 11192657196,
    #             11192657197, 11192657198, 11192657199, 11192657200, 11192657201, 11192657202, 11192657203, 11192657204,
    #             11192657205, 11192657206, 11192657207, 11192657208, 11192657209, 11192657210, 11192657211, 11192657212,
    #             11192657213, 11192657214, 11192657215, 11192657216, 11192657217, 11192657218, 11192657219, 11192657220,
    #             11192657221, 11192657222, 11192657223, 11192657224, 11192657225, 11192657226, 11192657227, 11192657228,
    #             11192657229, 11192657230, 11192657231, 11192657232, 11192657233, 11192657234], 'projectCategory': 1,
    #  'compositionPackageId': -1, 'designInfo': None,
    #  'designInfoTempLocation': 'pictures/temp/queuesdevaigeneratealbumdto/unxzt7xrvuchyydgp_wmqx4g.json',
    #  'aiMetadata': {'photoIds': [], 'focus': [], 'personIds': [], 'subjects': [], 'density': 3},
    #  'conditionId': 'AAD_47981912_908cb0fb-2e39-4a8a-ace7-0a451dc25e85.160.10', 'timedOut': False,
    #  'dependencyDeleted': False, 'retryCount': 0}


    #_input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310, 'projectId': 46227780, 'userId': 570976104, 'userJobId': 1111385993, 'base_url': 'ptstorage_32://pictures/46/227/46227780/njev8ankt8x9b7ynth', 'photos': [10799337783, 10799337786, 10799337789, 10799337792, 10799337796, 10799337798, 10799337802, 10799337805, 10799337808, 10799337811, 10799337817, 10799337820, 10799337824, 10799337828, 10799337832, 10799337836, 10799337840, 10799337845, 10799337849, 10799337853, 10799337857, 10799337861, 10799337865, 10799337869, 10799337873, 10799337876, 10799337878, 10799337882, 10799337886, 10799337889, 10799337893, 10799337897, 10799337901, 10799337905, 10799337909, 10799337913, 10799337918, 10799337921, 10799337926, 10799337929, 10799337933, 10799337938, 10799337942, 10799337946, 10799337950, 10799337954, 10799337958, 10799337962, 10799337966, 10799337970, 10799337975, 10799337980, 10799337985, 10799337990, 10799337995, 10799338000, 10799338005, 10799338010, 10799338014, 10799338020, 10799338025, 10799338031, 10799338036, 10799338041, 10799338046, 10799338051, 10799338056, 10799338061, 10799338066, 10799338071, 10799338075, 10799338080, 10799338085, 10799338089, 10799338094, 10799338099, 10799338104, 10799338110, 10799338120, 10799338125, 10799338130, 10799338135, 10799338139, 10799338144, 10799338149, 10799338154, 10799338159, 10799338164, 10799338169, 10799338173, 10799338177, 10799338182, 10799338187, 10799338192, 10799338197, 10799338202, 10799338207, 10799338212, 10799338217, 10799338222, 10799338227, 10799338231, 10799338236, 10799338241, 10799338247, 10799338253, 10799338259, 10799338265, 10799338271, 10799338277, 10799338283, 10799338289, 10799338295, 10799338301, 10799338307, 10799338314, 10799338321, 10799338328, 10799338334, 10799338340, 10799338345, 10799338350, 10799338358, 10799338366, 10799338372, 10799338379, 10799338386, 10799338394, 10799338402, 10799338409, 10799338416, 10799338423, 10799338429, 10799338436, 10799338443, 10799338451, 10799338460, 10799338468, 10799338475, 10799338484, 10799338493, 10799338501, 10799338509, 10799338518, 10799338530, 10799338539, 10799338547, 10799338555, 10799338562, 10799338570, 10799338578, 10799338585, 10799338593, 10799338601, 10799338605, 10799338613, 10799338621, 10799338629, 10799338637, 10799338645, 10799338653, 10799338662, 10799338670, 10799338677, 10799338683, 10799338691, 10799338700, 10799338709, 10799338718, 10799338727, 10799338735, 10799338745, 10799338753, 10799338762, 10799338772, 10799338781, 10799338789, 10799338800, 10799338809, 10799338817, 10799338825, 10799338835, 10799338844, 10799338855, 10799338867, 10799338878, 10799338887, 10799338896, 10799338905, 10799338914, 10799338923, 10799338932, 10799338941, 10799338951, 10799338960, 10799338969, 10799338978, 10799338987, 10799338996, 10799339001, 10799339010, 10799339019, 10799339032, 10799339041, 10799339050, 10799339060, 10799339069, 10799339078, 10799339088, 10799339096, 10799339105, 10799339114, 10799339123, 10799339132, 10799339142, 10799339152, 10799339161, 10799339171, 10799339180, 10799339189, 10799339198, 10799339209, 10799339218, 10799339227, 10799339237, 10799339246, 10799339255, 10799339264, 10799339273, 10799339281, 10799339291, 10799339300, 10799339309, 10799339317, 10799339326, 10799339335, 10799339344, 10799339354, 10799339361, 10799339370, 10799339379, 10799339388, 10799339396, 10799339407, 10799339418, 10799346928, 10799346935, 10799346942, 10799346949, 10799346956, 10799346962, 10799346969, 10799346976, 10799346983, 10799346990, 10799346997, 10799347004, 10799347012, 10799347018, 10799347026, 10799347033, 10799347039, 10799347045, 10799347051, 10799347057, 10799347063, 10799347069, 10799347075, 10799347080, 10799347086, 10799347092, 10799347098, 10799347104, 10799347110, 10799347116, 10799347122, 10799347128, 10799347134, 10799347140, 10799347145, 10799347151, 10799347157, 10799347163, 10799339426, 10799339435, 10799339443, 10799339453, 10799339462, 10799339471, 10799339479, 10799339488, 10799339497, 10799339505, 10799339516, 10799339526, 10799339534, 10799339541, 10799339548, 10799339556, 10799339562, 10799339568, 10799339577, 10799339586, 10799339595, 10799339604, 10799339612, 10799339620, 10799339629, 10799339637, 10799339646, 10799339654, 10799339662, 10799339682, 10799339689, 10799339706, 10799339716, 10799339724, 10799339734, 10799339744, 10799339749, 10799339754, 10799339761, 10799339771, 10799339780, 10799339788, 10799339797, 10799339807, 10799339816, 10799339826, 10799339840, 10799339851, 10799339861, 10799339870, 10799339881, 10799339890, 10799339899, 10799339907, 10799339917, 10799339926, 10799339935, 10799339943, 10799339950, 10799339960, 10799339969, 10799339976, 10799339985, 10799339991, 10799340000, 10799340009, 10799340018, 10799340028, 10799340036, 10799340045, 10799340054, 10799340060, 10799340069, 10799340076, 10799340078, 10799340081, 10799340087, 10799340095, 10799340101, 10799340106, 10799340113, 10799340120, 10799340128, 10799340138, 10799340144, 10799340149, 10799340154, 10799340158, 10799340164, 10799340169, 10799340176, 10799340181, 10799340186, 10799340190, 10799340194, 10799340199, 10799340204, 10799340210, 10799340214, 10799340218, 10799340223, 10799340227, 10799340232, 10799340237, 10799340244, 10799340252, 10799340259, 10799340266, 10799340274, 10799340281, 10799340288, 10799340295, 10799340302, 10799340309, 10799340318, 10799340327, 10799340333, 10799340341, 10799340348, 10799340357, 10799340364, 10799340373, 10799340383, 10799340390, 10799340398, 10799340407, 10799340413, 10799340421, 10799340430, 10799340437, 10799340444, 10799340453, 10799340463, 10799340473, 10799340483, 10799340492, 10799340501, 10799340510, 10799340517, 10799340525, 10799340533, 10799340542, 10799340549, 10799340556, 10799340564, 10799340573, 10799340580, 10799340589, 10799340597, 10799340606, 10799340614, 10799340622, 10799340630, 10799340640, 10799340650, 10799340660, 10799340667, 10799340675, 10799340682, 10799340690, 10799340699, 10799340709, 10799340717, 10799340727, 10799340732, 10799340739, 10799340747, 10799340765, 10799340771, 10799340778, 10799340785, 10799340794, 10799340802, 10799340810, 10799340819, 10799340828, 10799340838, 10799340847, 10799340856, 10799340865, 10799340874, 10799340883, 10799340892, 10799340901, 10799340910, 10799340919, 10799340925, 10799340936, 10799340948, 10799340956, 10799340966, 10799340974, 10799340983, 10799340989, 10799340996, 10799341003, 10799341011, 10799341019, 10799341027, 10799341035, 10799341044, 10799341052, 10799341059, 10799341066, 10799341075, 10799341083, 10799341090, 10799341098, 10799341106, 10799341114, 10799341124, 10799341131, 10799341136, 10799341141, 10799341147, 10799341155, 10799341166, 10799341174, 10799341183, 10799341193, 10799341202, 10799341211, 10799341218, 10799341226, 10799341233, 10799341241, 10799341246, 10799341252, 10799341259, 10799341266, 10799341273, 10799341280, 10799341286, 10799341295, 10799341303, 10799341311, 10799341318, 10799341326, 10799341335, 10799341343, 10799341351, 10799341358, 10799341367, 10799341375, 10799341384, 10799341393, 10799341402, 10799341409, 10799341418, 10799341428, 10799341435, 10799341444, 10799341456, 10799341461, 10799341469, 10799341477, 10799341485, 10799341494, 10799341502, 10799341512, 10799341520, 10799341529, 10799341538, 10799341546, 10799341556, 10799341565, 10799341573, 10799341580, 10799341589, 10799341598, 10799341607, 10799341615, 10799341626, 10799341635, 10799341644, 10799341653, 10799341662, 10799341670, 10799341677, 10799341686, 10799341695, 10799341704, 10799341713, 10799341725, 10799341734, 10799341743, 10799341752, 10799341760, 10799341769, 10799341777, 10799341785, 10799341795, 10799341802, 10799341812, 10799341820, 10799341829, 10799341846, 10799341856, 10799341865, 10799341875, 10799341883, 10799341893, 10799341903, 10799341911, 10799341917, 10799341927, 10799341936, 10799341944, 10799341954, 10799341962, 10799341972, 10799341981, 10799341982, 10799341983, 10799341984, 10799341990, 10799341996, 10799342006, 10799342014, 10799342021, 10799342030, 10799342039, 10799342049, 10799342057, 10799342065, 10799342073, 10799342083, 10799342092, 10799342101, 10799342108, 10799342117, 10799342125, 10799342134, 10799342143, 10799342153, 10799342162, 10799342171, 10799342180, 10799342189, 10799342198, 10799342206, 10799342215, 10799342223, 10799342232, 10799342241, 10799342250, 10799342259, 10799342269, 10799342277, 10799342285, 10799342295, 10799342303, 10799342313, 10799342321, 10799342331, 10799342340, 10799342349, 10799342358, 10799342367, 10799342376, 10799342385, 10799342394, 10799342402, 10799342410, 10799342420, 10799342429, 10799342439, 10799342448, 10799342457, 10799342466, 10799342476, 10799342485, 10799342492, 10799342501, 10799342511, 10799342521, 10799342530, 10799342541, 10799342550, 10799342557, 10799342565, 10799342574, 10799342583, 10799342601, 10799342611, 10799342620, 10799342629, 10799342634, 10799342642, 10799342651, 10799342661, 10799342670, 10799342678, 10799342687, 10799342696, 10799342704, 10799342714, 10799342722, 10799342732, 10799342743, 10799342749, 10799342759, 10799342769, 10799342777, 10799342787, 10799342796, 10799342805, 10799342815, 10799342823, 10799342831, 10799342840, 10799342849, 10799342858, 10799342867, 10799342876, 10799342886, 10799342895, 10799342905, 10799342915, 10799342923, 10799342932, 10799342941, 10799342949, 10799342959, 10799342968, 10799342978, 10799342985, 10799342995, 10799343004, 10799343013, 10799343022, 10799343026, 10799343036, 10799343049, 10799343059, 10799343068, 10799343075, 10799343084, 10799343093, 10799343104, 10799343115, 10799343124, 10799343133, 10799343144, 10799343151, 10799343159, 10799343170, 10799343180, 10799343188, 10799343206, 10799343215, 10799343223, 10799343232, 10799343241, 10799343249, 10799343258, 10799343267, 10799343276, 10799343286, 10799343294, 10799343303, 10799343312, 10799343321, 10799343330, 10799343339, 10799343348, 10799343357, 10799343366, 10799343375, 10799343384, 10799343393, 10799343402, 10799343412, 10799343420, 10799343429, 10799343438, 10799343447, 10799343456, 10799343466, 10799343478, 10799343487, 10799343496, 10799343505, 10799343513, 10799343524, 10799343533, 10799343542, 10799343550, 10799343558, 10799343568, 10799343577, 10799343585, 10799343594, 10799343603, 10799343611, 10799343620, 10799343629, 10799343637, 10799343644, 10799343652, 10799343661, 10799343671, 10799343678, 10799343687, 10799343696, 10799343705, 10799343714, 10799343723, 10799343732, 10799343740, 10799343747, 10799343756, 10799343767, 10799343778, 10799343786, 10799343793, 10799343802, 10799343811, 10799343820, 10799343829, 10799343838, 10799343847, 10799343855, 10799343863, 10799343871, 10799343880, 10799343889, 10799343898, 10799343907, 10799343916, 10799343925, 10799343932, 10799343940, 10799343950, 10799343958, 10799343968, 10799343976, 10799343985, 10799343994, 10799344003, 10799344012, 10799344022, 10799344031, 10799344039, 10799344048, 10799344058, 10799344064, 10799344073, 10799344087, 10799344094, 10799344103, 10799344112, 10799344121, 10799344131, 10799344140, 10799344149, 10799344158, 10799344167, 10799344176, 10799344185, 10799344194, 10799344204, 10799344214, 10799344223, 10799344232, 10799344241, 10799344250, 10799344259, 10799344267, 10799344276, 10799344285, 10799344297, 10799344307, 10799344315, 10799344323, 10799344330, 10799344340, 10799344349, 10799344357, 10799344365, 10799344371, 10799344376, 10799344383, 10799344388, 10799344396, 10799344405, 10799344412, 10799344419, 10799344428, 10799344437, 10799344447, 10799344456, 10799344465, 10799344474, 10799344484, 10799344495, 10799344506, 10799344511, 10799344517, 10799344526, 10799344535, 10799344543, 10799344553, 10799344559, 10799344565, 10799344570, 10799344577, 10799344585, 10799344592, 10799344603, 10799344613, 10799344621, 10799344629, 10799344636, 10799344644, 10799344653, 10799344662, 10799344671, 10799344680, 10799344688, 10799344697, 10799344706, 10799344714, 10799344723, 10799344732, 10799344741, 10799344749, 10799344758, 10799344768, 10799344778, 10799344788, 10799344795, 10799344807, 10799344814, 10799344829, 10799344834, 10799344842, 10799344851, 10799344860, 10799344875, 10799344884, 10799344893, 10799344901, 10799344910, 10799344919, 10799344928, 10799344937, 10799344946, 10799344955, 10799344965, 10799344974, 10799344983, 10799344992, 10799345001, 10799345011, 10799345021, 10799345031, 10799345040, 10799345048, 10799345057, 10799345065, 10799345076, 10799345088, 10799345097, 10799345106, 10799345113, 10799345122, 10799345131, 10799345140, 10799345149, 10799345156, 10799345166, 10799345174, 10799345182, 10799345191, 10799345200, 10799345207, 10799345217, 10799345227, 10799345236, 10799345245, 10799345254, 10799345263, 10799345272, 10799345281, 10799345289, 10799345297, 10799345304, 10799345312, 10799345321, 10799345330, 10799345336, 10799345345, 10799345351, 10799345378, 10799345388, 10799345395, 10799345406, 10799345413, 10799345421, 10799345430, 10799345439, 10799345448, 10799345458, 10799345466, 10799345475, 10799345483, 10799345494, 10799345502, 10799345511, 10799345520, 10799345529, 10799345538, 10799345549, 10799345558, 10799345566, 10799345575, 10799345583, 10799345593, 10799345602, 10799345611, 10799345619, 10799345629, 10799345638, 10799345647, 10799345656, 10799345665, 10799345674, 10799345683, 10799345693, 10799345702, 10799345710, 10799345719, 10799345727, 10799345737, 10799345746, 10799345755, 10799345764, 10799345772, 10799345781, 10799345799, 10799345808, 10799345816, 10799345826, 10799345835, 10799345844, 10799345854, 10799345863, 10799345873, 10799345882, 10799345891, 10799345900, 10799345911, 10799345920, 10799345929, 10799345938, 10799345947, 10799345956, 10799345965, 10799345973, 10799345982, 10799345990, 10799346000, 10799346009, 10799346017, 10799346027, 10799346036, 10799346046, 10799346055, 10799346064, 10799346072, 10799346082, 10799346092, 10799346100, 10799346109, 10799346117, 10799346125, 10799346133, 10799346141, 10799346148, 10799346155, 10799346159, 10799346165, 10799346173, 10799346181, 10799346189, 10799346196, 10799346204, 10799346212, 10799346222, 10799346229, 10799346237, 10799346245, 10799346253, 10799346261, 10799346269, 10799346278, 10799346291, 10799346298, 10799346309, 10799346316, 10799346324, 10799346332, 10799346338, 10799346345, 10799346352, 10799346359, 10799346365, 10799346373, 10799346383, 10799346388, 10799346395, 10799346405, 10799346408, 10799345790, 10799346418, 10799346424, 10799346431, 10799346436, 10799346443, 10799346450, 10799346457, 10799346463, 10799346471, 10799346478, 10799346485, 10799346491, 10799346498, 10799346505, 10799346511, 10799346520, 10799346528, 10799346535, 10799346542, 10799346550, 10799346564, 10799346571, 10799346578, 10799346585, 10799346592, 10799346599, 10799346605, 10799346612, 10799346619, 10799346626, 10799346633, 10799346640, 10799346647, 10799346654, 10799346664, 10799346670, 10799346677, 10799346684, 10799346691, 10799346698, 10799346705, 10799346713, 10799346719, 10799346726, 10799346733, 10799346740, 10799346747, 10799346754, 10799346761, 10799346768, 10799346775, 10799346782, 10799346789, 10799346796, 10799346803, 10799346809, 10799346816, 10799346823, 10799346830, 10799346834, 10799346843, 10799346850, 10799346858, 10799346862, 10799346872, 10799346880, 10799346886, 10799346893, 10799346900, 10799346907, 10799346914, 10799346921], 'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/queuesdevaigeneratealbumdto/tmmtzxvkm0c39srqatyfjcu2.json', 'aiMetadata': {'photoIds': [], 'focus': [], 'personIds': [], 'subjects': [], 'density': 3}, 'conditionId': 'AAD_46227780_ff1528c7-0361-4be9-adde-2628957d933b.89.315', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}

    # Debug with Plotting
    _images_path_karmel = fr'C:\Users\ZivRotman\PycharmProjects\logAnalysis\galleries_pbs2\{_input_request["projectId"]}'
    _output_pdf_path_karmel = fr'c:\temp\albums\{_input_request["projectId"]}/'
    os.makedirs(_output_pdf_path_karmel, exist_ok=True)
    _output_pdf_path = os.path.join(_output_pdf_path_karmel, 'album1.pdf')

    _images_path = _images_path_karmel
    final_album, _message = process_gallery(_input_request)
    gallery_photos_info = _message.content['gallery_photos_info']

    box_id2data = _message.designsInfo['anyPagebox_id2data'] # if 'designsInfo' in _message and 'anyPagebox_id2data' in _message['designsInfo'] else {}
    visualize_album_to_pdf(final_album, _images_path, _output_pdf_path, box_id2data, gallery_photos_info)

    print(final_album)



