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
    _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310, 'projectId': 46227780, 'userId': 576349956, 'userJobId': 1121483286, 'base_url': 'ptstorage_32://pictures/46/227/46227780/njev8ankt8x9b7ynth', 'photos': [10799337783, 10799337786, 10799337789, 10799337792, 10799337796, 10799337798, 10799337802, 10799337805, 10799337808, 10799337811, 10799337817, 10799337820, 10799337824, 10799337828, 10799337832, 10799337836, 10799337840, 10799337845, 10799337849, 10799337853, 10799337857, 10799337861, 10799337865, 10799337869, 10799337873, 10799337876, 10799337878, 10799337882, 10799337886, 10799337889, 10799337893, 10799337897, 10799337901, 10799337905, 10799337909, 10799337913, 10799337918, 10799337921, 10799337926, 10799337929, 10799337933, 10799337938, 10799337942, 10799337946, 10799337950, 10799337954, 10799337958, 10799337962, 10799337966, 10799337970, 10799337975, 10799337980, 10799337985, 10799337990, 10799337995, 10799338000, 10799338005, 10799338010, 10799338014, 10799338020, 10799338025, 10799338031, 10799338036, 10799338041, 10799338046, 10799338051, 10799338056, 10799338061, 10799338066, 10799338071, 10799338075, 10799338080, 10799338085, 10799338089, 10799338094, 10799338099, 10799338104, 10799338110, 10799338120, 10799338125, 10799338130, 10799338135, 10799338139, 10799338144, 10799338149, 10799338154, 10799338159, 10799338164, 10799338169, 10799338173, 10799338177, 10799338182, 10799338187, 10799338192, 10799338197, 10799338202, 10799338207, 10799338212, 10799338217, 10799338222, 10799338227, 10799338231, 10799338236, 10799338241, 10799338247, 10799338253, 10799338259, 10799338265, 10799338271, 10799338277, 10799338283, 10799338289, 10799338295, 10799338301, 10799338307, 10799338314, 10799338321, 10799338328, 10799338334, 10799338340, 10799338345, 10799338350, 10799338358, 10799338366, 10799338372, 10799338379, 10799338386, 10799338394, 10799338402, 10799338409, 10799338416, 10799338423, 10799338429, 10799338436, 10799338443, 10799338451, 10799338460, 10799338468, 10799338475, 10799338484, 10799338493, 10799338501, 10799338509, 10799338518, 10799338530, 10799338539, 10799338547, 10799338555, 10799338562, 10799338570, 10799338578, 10799338585, 10799338593, 10799338601, 10799338605, 10799338613, 10799338621, 10799338629, 10799338637, 10799338645, 10799338653, 10799338662, 10799338670, 10799338677, 10799338683, 10799338691, 10799338700, 10799338709, 10799338718, 10799338727, 10799338735, 10799338745, 10799338753, 10799338762, 10799338772, 10799338781, 10799338789, 10799338800, 10799338809, 10799338817, 10799338825, 10799338835, 10799338844, 10799338855, 10799338867, 10799338878, 10799338887, 10799338896, 10799338905, 10799338914, 10799338923, 10799338932, 10799338941, 10799338951, 10799338960, 10799338969, 10799338978, 10799338987, 10799338996, 10799339001, 10799339010, 10799339019, 10799339032, 10799339041, 10799339050, 10799339060, 10799339069, 10799339078, 10799339088, 10799339096, 10799339105, 10799339114, 10799339123, 10799339132, 10799339142, 10799339152, 10799339161, 10799339171, 10799339180, 10799339189, 10799339198, 10799339209, 10799339218, 10799339227, 10799339237, 10799339246, 10799339255, 10799339264, 10799339273, 10799339281, 10799339291, 10799339300, 10799339309, 10799339317, 10799339326, 10799339335, 10799339344, 10799339354, 10799339361, 10799339370, 10799339379, 10799339388, 10799339396, 10799339407, 10799339418, 10799346928, 10799346935, 10799346942, 10799346949, 10799346956, 10799346962, 10799346969, 10799346976, 10799346983, 10799346990, 10799346997, 10799347004, 10799347012, 10799347018, 10799347026, 10799347033, 10799347039, 10799347045, 10799347051, 10799347057, 10799347063, 10799347069, 10799347075, 10799347080, 10799347086, 10799347092, 10799347098, 10799347104, 10799347110, 10799347116, 10799347122, 10799347128, 10799347134, 10799347140, 10799347145, 10799347151, 10799347157, 10799347163, 10799339426, 10799339435, 10799339443, 10799339453, 10799339462, 10799339471, 10799339479, 10799339488, 10799339497, 10799339505, 10799339516, 10799339526, 10799339534, 10799339541, 10799339548, 10799339556, 10799339562, 10799339568, 10799339577, 10799339586, 10799339595, 10799339604, 10799339612, 10799339620, 10799339629, 10799339637, 10799339646, 10799339654, 10799339662, 10799339682, 10799339689, 10799339706, 10799339716, 10799339724, 10799339734, 10799339744, 10799339749, 10799339754, 10799339761, 10799339771, 10799339780, 10799339788, 10799339797, 10799339807, 10799339816, 10799339826, 10799339840, 10799339851, 10799339861, 10799339870, 10799339881, 10799339890, 10799339899, 10799339907, 10799339917, 10799339926, 10799339935, 10799339943, 10799339950, 10799339960, 10799339969, 10799339976, 10799339985, 10799339991, 10799340000, 10799340009, 10799340018, 10799340028, 10799340036, 10799340045, 10799340054, 10799340060, 10799340069, 10799340076, 10799340078, 10799340081, 10799340087, 10799340095, 10799340101, 10799340106, 10799340113, 10799340120, 10799340128, 10799340138, 10799340144, 10799340149, 10799340154, 10799340158, 10799340164, 10799340169, 10799340176, 10799340181, 10799340186, 10799340190, 10799340194, 10799340199, 10799340204, 10799340210, 10799340214, 10799340218, 10799340223, 10799340227, 10799340232, 10799340237, 10799340244, 10799340252, 10799340259, 10799340266, 10799340274, 10799340281, 10799340288, 10799340295, 10799340302, 10799340309, 10799340318, 10799340327, 10799340333, 10799340341, 10799340348, 10799340357, 10799340364, 10799340373, 10799340383, 10799340390, 10799340398, 10799340407, 10799340413, 10799340421, 10799340430, 10799340437, 10799340444, 10799340453, 10799340463, 10799340473, 10799340483, 10799340492, 10799340501, 10799340510, 10799340517, 10799340525, 10799340533, 10799340542, 10799340549, 10799340556, 10799340564, 10799340573, 10799340580, 10799340589, 10799340597, 10799340606, 10799340614, 10799340622, 10799340630, 10799340640, 10799340650, 10799340660, 10799340667, 10799340675, 10799340682, 10799340690, 10799340699, 10799340709, 10799340717, 10799340727, 10799340732, 10799340739, 10799340747, 10799340765, 10799340771, 10799340778, 10799340785, 10799340794, 10799340802, 10799340810, 10799340819, 10799340828, 10799340838, 10799340847, 10799340856, 10799340865, 10799340874, 10799340883, 10799340892, 10799340901, 10799340910, 10799340919, 10799340925, 10799340936, 10799340948, 10799340956, 10799340966, 10799340974, 10799340983, 10799340989, 10799340996, 10799341003, 10799341011, 10799341019, 10799341027, 10799341035, 10799341044, 10799341052, 10799341059, 10799341066, 10799341075, 10799341083, 10799341090, 10799341098, 10799341106, 10799341114, 10799341124, 10799341131, 10799341136, 10799341141, 10799341147, 10799341155, 10799341166, 10799341174, 10799341183, 10799341193, 10799341202, 10799341211, 10799341218, 10799341226, 10799341233, 10799341241, 10799341246, 10799341252, 10799341259, 10799341266, 10799341273, 10799341280, 10799341286, 10799341295, 10799341303, 10799341311, 10799341318, 10799341326, 10799341335, 10799341343, 10799341351, 10799341358, 10799341367, 10799341375, 10799341384, 10799341393, 10799341402, 10799341409, 10799341418, 10799341428, 10799341435, 10799341444, 10799341456, 10799341461, 10799341469, 10799341477, 10799341485, 10799341494, 10799341502, 10799341512, 10799341520, 10799341529, 10799341538, 10799341546, 10799341556, 10799341565, 10799341573, 10799341580, 10799341589, 10799341598, 10799341607, 10799341615, 10799341626, 10799341635, 10799341644, 10799341653, 10799341662, 10799341670, 10799341677, 10799341686, 10799341695, 10799341704, 10799341713, 10799341725, 10799341734, 10799341743, 10799341752, 10799341760, 10799341769, 10799341777, 10799341785, 10799341795, 10799341802, 10799341812, 10799341820, 10799341829, 10799341846, 10799341856, 10799341865, 10799341875, 10799341883, 10799341893, 10799341903, 10799341911, 10799341917, 10799341927, 10799341936, 10799341944, 10799341954, 10799341962, 10799341972, 10799341981, 10799341982, 10799341983, 10799341984, 10799341990, 10799341996, 10799342006, 10799342014, 10799342021, 10799342030, 10799342039, 10799342049, 10799342057, 10799342065, 10799342073, 10799342083, 10799342092, 10799342101, 10799342108, 10799342117, 10799342125, 10799342134, 10799342143, 10799342153, 10799342162, 10799342171, 10799342180, 10799342189, 10799342198, 10799342206, 10799342215, 10799342223, 10799342232, 10799342241, 10799342250, 10799342259, 10799342269, 10799342277, 10799342285, 10799342295, 10799342303, 10799342313, 10799342321, 10799342331, 10799342340, 10799342349, 10799342358, 10799342367, 10799342376, 10799342385, 10799342394, 10799342402, 10799342410, 10799342420, 10799342429, 10799342439, 10799342448, 10799342457, 10799342466, 10799342476, 10799342485, 10799342492, 10799342501, 10799342511, 10799342521, 10799342530, 10799342541, 10799342550, 10799342557, 10799342565, 10799342574, 10799342583, 10799342601, 10799342611, 10799342620, 10799342629, 10799342634, 10799342642, 10799342651, 10799342661, 10799342670, 10799342678, 10799342687, 10799342696, 10799342704, 10799342714, 10799342722, 10799342732, 10799342743, 10799342749, 10799342759, 10799342769, 10799342777, 10799342787, 10799342796, 10799342805, 10799342815, 10799342823, 10799342831, 10799342840, 10799342849, 10799342858, 10799342867, 10799342876, 10799342886, 10799342895, 10799342905, 10799342915, 10799342923, 10799342932, 10799342941, 10799342949, 10799342959, 10799342968, 10799342978, 10799342985, 10799342995, 10799343004, 10799343013, 10799343022, 10799343026, 10799343036, 10799343049, 10799343059, 10799343068, 10799343075, 10799343084, 10799343093, 10799343104, 10799343115, 10799343124, 10799343133, 10799343144, 10799343151, 10799343159, 10799343170, 10799343180, 10799343188, 10799343206, 10799343215, 10799343223, 10799343232, 10799343241, 10799343249, 10799343258, 10799343267, 10799343276, 10799343286, 10799343294, 10799343303, 10799343312, 10799343321, 10799343330, 10799343339, 10799343348, 10799343357, 10799343366, 10799343375, 10799343384, 10799343393, 10799343402, 10799343412, 10799343420, 10799343429, 10799343438, 10799343447, 10799343456, 10799343466, 10799343478, 10799343487, 10799343496, 10799343505, 10799343513, 10799343524, 10799343533, 10799343542, 10799343550, 10799343558, 10799343568, 10799343577, 10799343585, 10799343594, 10799343603, 10799343611, 10799343620, 10799343629, 10799343637, 10799343644, 10799343652, 10799343661, 10799343671, 10799343678, 10799343687, 10799343696, 10799343705, 10799343714, 10799343723, 10799343732, 10799343740, 10799343747, 10799343756, 10799343767, 10799343778, 10799343786, 10799343793, 10799343802, 10799343811, 10799343820, 10799343829, 10799343838, 10799343847, 10799343855, 10799343863, 10799343871, 10799343880, 10799343889, 10799343898, 10799343907, 10799343916, 10799343925, 10799343932, 10799343940, 10799343950, 10799343958, 10799343968, 10799343976, 10799343985, 10799343994, 10799344003, 10799344012, 10799344022, 10799344031, 10799344039, 10799344048, 10799344058, 10799344064, 10799344073, 10799344087, 10799344094, 10799344103, 10799344112, 10799344121, 10799344131, 10799344140, 10799344149, 10799344158, 10799344167, 10799344176, 10799344185, 10799344194, 10799344204, 10799344214, 10799344223, 10799344232, 10799344241, 10799344250, 10799344259, 10799344267, 10799344276, 10799344285, 10799344297, 10799344307, 10799344315, 10799344323, 10799344330, 10799344340, 10799344349, 10799344357, 10799344365, 10799344371, 10799344376, 10799344383, 10799344388, 10799344396, 10799344405, 10799344412, 10799344419, 10799344428, 10799344437, 10799344447, 10799344456, 10799344465, 10799344474, 10799344484, 10799344495, 10799344506, 10799344511, 10799344517, 10799344526, 10799344535, 10799344543, 10799344553, 10799344559, 10799344565, 10799344570, 10799344577, 10799344585, 10799344592, 10799344603, 10799344613, 10799344621, 10799344629, 10799344636, 10799344644, 10799344653, 10799344662, 10799344671, 10799344680, 10799344688, 10799344697, 10799344706, 10799344714, 10799344723, 10799344732, 10799344741, 10799344749, 10799344758, 10799344768, 10799344778, 10799344788, 10799344795, 10799344807, 10799344814, 10799344829, 10799344834, 10799344842, 10799344851, 10799344860, 10799344875, 10799344884, 10799344893, 10799344901, 10799344910, 10799344919, 10799344928, 10799344937, 10799344946, 10799344955, 10799344965, 10799344974, 10799344983, 10799344992, 10799345001, 10799345011, 10799345021, 10799345031, 10799345040, 10799345048, 10799345057, 10799345065, 10799345076, 10799345088, 10799345097, 10799345106, 10799345113, 10799345122, 10799345131, 10799345140, 10799345149, 10799345156, 10799345166, 10799345174, 10799345182, 10799345191, 10799345200, 10799345207, 10799345217, 10799345227, 10799345236, 10799345245, 10799345254, 10799345263, 10799345272, 10799345281, 10799345289, 10799345297, 10799345304, 10799345312, 10799345321, 10799345330, 10799345336, 10799345345, 10799345351, 10799345378, 10799345388, 10799345395, 10799345406, 10799345413, 10799345421, 10799345430, 10799345439, 10799345448, 10799345458, 10799345466, 10799345475, 10799345483, 10799345494, 10799345502, 10799345511, 10799345520, 10799345529, 10799345538, 10799345549, 10799345558, 10799345566, 10799345575, 10799345583, 10799345593, 10799345602, 10799345611, 10799345619, 10799345629, 10799345638, 10799345647, 10799345656, 10799345665, 10799345674, 10799345683, 10799345693, 10799345702, 10799345710, 10799345719, 10799345727, 10799345737, 10799345746, 10799345755, 10799345764, 10799345772, 10799345781, 10799345799, 10799345808, 10799345816, 10799345826, 10799345835, 10799345844, 10799345854, 10799345863, 10799345873, 10799345882, 10799345891, 10799345900, 10799345911, 10799345920, 10799345929, 10799345938, 10799345947, 10799345956, 10799345965, 10799345973, 10799345982, 10799345990, 10799346000, 10799346009, 10799346017, 10799346027, 10799346036, 10799346046, 10799346055, 10799346064, 10799346072, 10799346082, 10799346092, 10799346100, 10799346109, 10799346117, 10799346125, 10799346133, 10799346141, 10799346148, 10799346155, 10799346159, 10799346165, 10799346173, 10799346181, 10799346189, 10799346196, 10799346204, 10799346212, 10799346222, 10799346229, 10799346237, 10799346245, 10799346253, 10799346261, 10799346269, 10799346278, 10799346291, 10799346298, 10799346309, 10799346316, 10799346324, 10799346332, 10799346338, 10799346345, 10799346352, 10799346359, 10799346365, 10799346373, 10799346383, 10799346388, 10799346395, 10799346405, 10799346408, 10799345790, 10799346418, 10799346424, 10799346431, 10799346436, 10799346443, 10799346450, 10799346457, 10799346463, 10799346471, 10799346478, 10799346485, 10799346491, 10799346498, 10799346505, 10799346511, 10799346520, 10799346528, 10799346535, 10799346542, 10799346550, 10799346564, 10799346571, 10799346578, 10799346585, 10799346592, 10799346599, 10799346605, 10799346612, 10799346619, 10799346626, 10799346633, 10799346640, 10799346647, 10799346654, 10799346664, 10799346670, 10799346677, 10799346684, 10799346691, 10799346698, 10799346705, 10799346713, 10799346719, 10799346726, 10799346733, 10799346740, 10799346747, 10799346754, 10799346761, 10799346768, 10799346775, 10799346782, 10799346789, 10799346796, 10799346803, 10799346809, 10799346816, 10799346823, 10799346830, 10799346834, 10799346843, 10799346850, 10799346858, 10799346862, 10799346872, 10799346880, 10799346886, 10799346893, 10799346900, 10799346907, 10799346914, 10799346921], 'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/queuesdevaigeneratealbumdto/_ztnamy550gmbil7zebntalo.json', 'aiMetadata': {'photoIds': [], 'focus': [], 'personIds': [], 'subjects': [], 'density': 3}, 'conditionId': 'AAD_46227780_251117-141214.72ea06ca-0613-49db-a811-dcec370b8b27.277.2', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}
    # Debug with Plotting
    _images_path_karmel = fr'C:\Users\user\Desktop\PicTime\AlbumDesigner\dataset\{_input_request["projectId"]}'
    _output_pdf_path_karmel = fr'C:\Users\user\Desktop\PicTime\AlbumDesigner\output\{_input_request["projectId"]}/'
    os.makedirs(_output_pdf_path_karmel, exist_ok=True)
    _output_pdf_path = os.path.join(_output_pdf_path_karmel, 'album1.pdf')

    _images_path = _images_path_karmel
    final_album, _message = process_gallery(_input_request)
    gallery_photos_info = _message.content['gallery_photos_info']

    box_id2data = _message.designsInfo['anyPagebox_id2data'] # if 'designsInfo' in _message and 'anyPagebox_id2data' in _message['designsInfo'] else {}
    visualize_album_to_pdf(final_album, _images_path, _output_pdf_path, box_id2data, gallery_photos_info)

    print(final_album)



