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

        manual_selection = message.content.get('manual_selection', False)
        album_result = album_processing(df, message.designsInfo, message.content['is_wedding'], modified_lut, params, logger=logger, density=density, manual_selection=manual_selection)

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

    settings_filename = os.environ.get('HostingSettingsPath',
                                       '/ptinternal/pictures/hosting/ai_settings_audiobeat.json.txt')
    intialize('AlbumDesigner', settings_filename)

    _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 4, 'accountId': 378922, 'projectId': 49807801, 'userId': 497144430, 'userJobId': 1119418516, 'base_url': 'ptstorage_11://pictures/49/807/49807801/smczjy8ieo81dimy1a', 'photos': [11518717291, 11518717304, 11518717307, 11518717314, 11518717322, 11518717347, 11518717351, 11518717359, 11518717379, 11518717388, 11518717394, 11518717401, 11518717434, 11518717437, 11518717440, 11518717443, 11518717448, 11518717449, 11518717460, 11518717478, 11518717488, 11518716722, 11518716726, 11518716732, 11518716734, 11518716830, 11518716846, 11518716914, 11518716942, 11518716955, 11518716986, 11518717058, 11518717061, 11518717087, 11518717090, 11518717166, 11518717182, 11518717211, 11518717213, 11518717219, 11518717242, 11518717257, 11518717258, 11523560050, 11523560249, 11523560276, 11523560302, 11523560497, 11523560518, 11523560533, 11523560534, 11523560539, 11523560546, 11523560564, 11523560566, 11523560595, 11523560597, 11523560598, 11523560606, 11523560965, 11523561005, 11523563552, 11523563562, 11523564076, 11523564082, 11523564091, 11523564157, 11523561060, 11523561072, 11523561098, 11523561099, 11523561107, 11523561110, 11523561141, 11523561172, 11523561186, 11523561194, 11523561196, 11523561217, 11523561221, 11523561225, 11523561229, 11523561266, 11523561270, 11523561297, 11523563195, 11523563217, 11523563230, 11523563285, 11523563299, 11523563331, 11523563339, 11523563377, 11523563385, 11523563386, 11523563394, 11523563397, 11523563406, 11523563419, 11523563420, 11523564169, 11523564178, 11523564237, 11523564263, 11523564329, 11523564373, 11523564378, 11523565282, 11523565301, 11523565312, 11523565313, 11523565316, 11523565320, 11523565352, 11523565397, 11523565464, 11523565557, 11523599403, 11523599405, 11523599415], 'projectCategory': 0, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/queues/devaigeneratealbumdto/vlh4awg1lk2icofy6jkzpbvv.json', 'aiMetadata': {'photoIds': None, 'focus': [], 'personIds': [], 'subjects': [], 'density': 3}, 'conditionId': 'AAD_49807801_D.251229-121829.8df713fc-8798-4aab-8e52-be06df244d62.157.6', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}
    _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 297542, 'projectId': 49294538, 'userId': 632122945, 'userJobId': 1234174399, 'base_url': 'ptstorage_32://pictures/49/294/49294538/a5hnfbjktt9t199p3m', 'photos': [11436509747, 11436509782, 11436509786, 11436509789, 11436509799, 11436509901, 11436509918, 11436509929, 11436509420, 11436509433, 11436509482, 11436509481, 11436509835, 11436509846, 11436509855, 11436509854, 11436509881, 11436509873, 11436509626, 11436509642, 11436509673, 11436509672, 11436509701, 11436509693, 11436509683, 11436509546, 11436509550, 11436509543, 11436509527, 11436509902, 11436509903, 11436509904, 11436509905, 11436509906, 11436509907, 11436509908, 11436509909, 11436509910, 11436509911, 11436509912, 11436509913, 11436509914, 11436509915, 11436509916, 11436509917, 11436509919, 11436509920, 11436509921, 11436509922, 11436509923, 11436509924, 11436509925, 11436509926, 11436509927, 11436509928, 11436509930, 11436509931, 11436509932, 11436509933, 11436509934, 11436509935, 11436509936, 11436509937, 11436509938, 11436509939, 11436509940, 11436509941, 11436509942, 11436509943, 11436509944, 11436509945, 11436509946, 11436509947, 11436509948, 11436509949, 11436509950, 11436509951, 11436509952, 11436509417, 11436509418, 11436509419, 11436509421, 11436509422, 11436509423, 11436509424, 11436509425, 11436509426, 11436509427, 11436509428, 11436509429, 11436509430, 11436509431, 11436509432, 11436509434, 11436509435, 11436509436, 11436509437, 11436509438, 11436509439, 11436509440, 11436509441, 11436509442, 11436509443, 11436509444, 11436509445, 11436509446, 11436509447, 11436509448, 11436509449, 11436509450, 11436509451, 11436509452, 11436509453, 11436509454, 11436509455, 11436509456, 11436509457, 11436509458, 11436509459, 11436509460, 11436509461, 11436509462, 11436509463, 11436509464, 11436509465, 11436509466, 11436509467, 11436509468, 11436509469, 11436509470, 11436509471, 11436509472, 11436509473, 11436509474, 11436509475, 11436509476, 11436509477, 11436509478, 11436509479, 11436509480, 11436509483, 11436509484, 11436509485, 11436509486, 11436509487, 11436509488, 11436509489, 11436509490, 11436509491, 11436509492, 11436509493, 11436509494, 11436509495, 11436509496, 11436509497, 11436509498, 11436509499, 11436509500, 11436509501, 11436509502, 11436509503, 11436509504, 11436509505, 11436509506, 11436509507, 11436509508, 11436509509, 11436509828, 11436509829, 11436509830, 11436509831, 11436509832, 11436509833, 11436509834, 11436509836, 11436509837, 11436509838, 11436509839, 11436509840, 11436509841, 11436509842, 11436509843, 11436509844, 11436509845, 11436509847, 11436509848, 11436509849, 11436509850, 11436509851, 11436509852, 11436509853, 11436509856, 11436509857, 11436509858, 11436509859, 11436509860, 11436509861, 11436509862, 11436509863, 11436509864, 11436509865, 11436509866, 11436509867, 11436509868, 11436509869, 11436509870, 11436509871, 11436509872, 11436509874, 11436509875, 11436509876, 11436509877, 11436509878, 11436509879, 11436509880, 11436509882, 11436509883, 11436509884, 11436509885, 11436509886, 11436509887, 11436509888, 11436509889, 11436509890, 11436509891, 11436509892, 11436509893, 11436509894, 11436509895, 11436509896, 11436509897, 11436509898, 11436509899, 11436509615, 11436509616, 11436509617, 11436509618, 11436509619, 11436509620, 11436509621, 11436509622, 11436509623, 11436509624, 11436509625, 11436509627, 11436509628, 11436509629, 11436509630, 11436509631, 11436509632, 11436509633, 11436509634, 11436509635, 11436509636, 11436509637, 11436509638, 11436509639, 11436509640, 11436509641, 11436509643, 11436509644, 11436509645, 11436509646, 11436509647, 11436509648, 11436509649, 11436509650, 11436509651, 11436509652, 11436509653, 11436509654, 11436509655, 11436509656, 11436509657, 11436509658, 11436509659, 11436509660, 11436509661, 11436509662, 11436509663, 11436509664, 11436509665, 11436509666, 11436509667, 11436509668, 11436509669, 11436509670, 11436509671, 11436509674, 11436509675, 11436509676, 11436509679, 11436509680, 11436509681, 11436509682, 11436509684, 11436509685, 11436509686, 11436509687, 11436509688, 11436509689, 11436509690, 11436509691, 11436509692, 11436509694, 11436509695, 11436509696, 11436509697, 11436509698, 11436509699, 11436509700, 11436509702, 11436509703, 11436509704, 11436509705, 11436509706, 11436509707, 11436509708, 11436509709, 11436509710, 11436509711, 11436509712, 11436509713, 11436509714, 11436509715, 11436509716, 11436509717, 11436509718, 11436509719, 11436509720, 11436509721, 11436509722, 11436509723, 11436509724, 11436509725, 11436509726, 11436509727, 11436509728, 11436509729, 11436509730, 11436509731, 11436509732, 11436509733], 'projectCategory': 0, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/queues/devaigeneratealbumdto/u6srvnj8-uu4-twxnq0flsh6.json', 'aiMetadata': {'photoIds': None, 'focus': [], 'personIds': [], 'subjects': [], 'density': 3}, 'conditionId': 'AAD_49294538_D.251229-125401.04d4a92d-2559-4288-babc-c65d589ed102.9.50', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}
    _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 493990, 'projectId': 47526108, 'userId': 646164715, 'userJobId': 1263181082, 'base_url': 'ptstorage_200://pictures/47/526/47526108/hzmishyaik3s0uy3ny', 'photos': [11097847007, 11097846688, 11097846734, 11097846713, 11097846925, 11097845859, 11097845877, 11097846283, 11097846193, 11097846270, 11097846295, 11097846207, 11097846252, 11097845929, 11097846242, 11097845931, 11097846329, 11097846214, 11097845592, 11097845556, 11097845543, 11097845569, 11097845574, 11097845563, 11097845607, 11097845617, 11097845611, 11097845529, 11097845530, 11097845525], 'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/queues/devaigeneratealbumdto/itfpciz4q0uzh6pjlmprevwv.json', 'aiMetadata': {'photoIds': None, 'focus': [], 'personIds': [], 'subjects': [], 'density': 3}, 'conditionId': 'AAD_47526108_D.251231-093347.7e514092-2bc8-4ee0-a0d8-949478ec5602.174.70', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}
    _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 493990, 'projectId': 47526108, 'userId': 646164715, 'userJobId': 1263181082, 'base_url': 'ptstorage_200://pictures/47/526/47526108/hzmishyaik3s0uy3ny', 'photos': [11097845606, 11097845595, 11097845535, 11097845619, 11097845552, 11097845596, 11097845591, 11097845567, 11097845735, 11097845682, 11097845628, 11097845641, 11097845637, 11097845675, 11097845671, 11097845674, 11097845670, 11097845644, 11097845661, 11097845656, 11097845711, 11097845689, 11097845686, 11097845693, 11097845714, 11097845712, 11097845694, 11097845683, 11097845672, 11097845649, 11097845678, 11097845667, 11097845679, 11097845650, 11097845692, 11097845699, 11097845710, 11097845709, 11097845700, 11097845804, 11097845776, 11097845766, 11097845759, 11097845833, 11097845821, 11097845824, 11097845823, 11097845822, 11097845748, 11097845825, 11097845811, 11097845747, 11097845819, 11097845826, 11097845745, 11097845827, 11097845925, 11097846217, 11097846244, 11097846268, 11097846269, 11097846220, 11097846245, 11097846235, 11097845924, 11097846283, 11097846259, 11097846253, 11097846281, 11097846280, 11097846202, 11097846278, 11097846287, 11097846266, 11097846270, 11097846262, 11097846252, 11097846216, 11097846229, 11097845930, 11097846206, 11097846240, 11097846241, 11097846221, 11097846256, 11097846257, 11097846239, 11097846223, 11097846222, 11097845931, 11097846205, 11097846291, 11097846276, 11097846254, 11097846264, 11097846247, 11097846234, 11097846261, 11097846218, 11097846191, 11097846198, 11097846297, 11097845923, 11097846236, 11097846255, 11097845909, 11097845867, 11097845879, 11097845851, 11097846330, 11097845868, 11097845876, 11097846334, 11097846346, 11097846343, 11097845875, 11097845874, 11097845871, 11097846344, 11097845873, 11097845872, 11097845910, 11097846329, 11097846336, 11097846350, 11097846233, 11097846231, 11097846250, 11097846285, 11097846298, 11097846387, 11097846380, 11097846396, 11097846379, 11097846374, 11097846388, 11097846397, 11097846408, 11097846413, 11097846398, 11097846362, 11097846375, 11097846378, 11097846358, 11097846400, 11097846405, 11097846376, 11097846377, 11097846395, 11097846390, 11097846404, 11097846401, 11097846411, 11097846391, 11097846394, 11097846383, 11097846370, 11097846392, 11097846393, 11097846385, 11097846402, 11097846409, 11097846412, 11097846450, 11097846451, 11097846745, 11097847032, 11097847006, 11097846981, 11097847033, 11097846671, 11097846691, 11097846755, 11097846735, 11097846960, 11097846833, 11097846836, 11097846758, 11097846811, 11097846891, 11097846925, 11097846955, 11097846907, 11097846861, 11097846926, 11097846911, 11097846793, 11097846806, 11097846936, 11097846762, 11097846853, 11097846818, 11097846881, 11097846771, 11097846798, 11097846844, 11097846851, 11097846843, 11097846852, 11097846826, 11097846773, 11097846769, 11097846800, 11097846945, 11097846874, 11097846849, 11097847026, 11097847016, 11097846698, 11097847017, 11097847041, 11097846970, 11097846704, 11097846680, 11097846989, 11097846742, 11097847039, 11097846986, 11097846694, 11097847052, 11097846761, 11097846952, 11097846893, 11097846855, 11097846781, 11097846892, 11097846910, 11097846930, 11097846909, 11097846956, 11097846905, 11097846987, 11097846700, 11097847020, 11097846972, 11097846729, 11097846718, 11097846746, 11097846992, 11097846725, 11097846705, 11097846684, 11097847015, 11097847025, 11097846774, 11097846796, 11097846822, 11097846871, 11097846850, 11097846823, 11097846845, 11097846824, 11097846775, 11097846802, 11097846947, 11097846895, 11097846942, 11097846917, 11097846868], 'projectCategory': 1, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/queues/devaigeneratealbumdto/yv8ijtll3ukhi-vmqybw_2da.json', 'aiMetadata': {'photoIds': None, 'focus': [], 'personIds': [], 'subjects': [], 'density': 3}, 'conditionId': 'AAD_47526108_D.251231-094344.7e514092-2bc8-4ee0-a0d8-949478ec5602.200.152', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}

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



