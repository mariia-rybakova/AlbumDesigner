import os
import pandas as pd

from typing import Dict
from datetime import datetime
import multiprocessing as mp
from collections import defaultdict

from ptinfra.azure.pt_file import PTFile
from ptinfra import get_logger

from utils.request_processing import read_messages

from src.smart_cropping import process_crop_images
from utils.cover_image import process_non_wedding_cover_image, process_wedding_first_last_image, get_first_last_design_ids
from utils.time_processing import process_image_time, get_time_clusters
from src.album_processing import album_processing
from utils.request_processing import assembly_output
from utils.clusters_labels import map_cluster_label
from utils.auto_selection import ai_selection

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from PIL import Image
import io


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
            img_path = os.path.join(images_path, f"{photo_id}.jpg")
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
                text = f"{general_time}{cluster_context}"
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
        if 'photosIds' not in message.content:
            logger("the 10 photos not selected!")
            ten_photos = []
        else:
            ten_photos = message.content.get('photosIds', [])

        if 'people_ids' not in message.content:
            people_ids = []
        else:
            people_ids = message.content.get('people_ids', [])

        df = message.content.get('gallery_photos_info', pd.DataFrame())
        if df.empty:
            logger.error(f"Gallery photos info DataFrame is empty for message {message}")
            message.content['error'] = f"Gallery photos info DataFrame is empty for message {message}"
            return message

        ai_photos_selected,errors = ai_selection(df, ten_photos, people_ids, message.content['focus'], message.content['tags'], message.content['is_wedding'], message.content['density'],
                                                 logger)

        if errors:
            logger.error(f"Error for Selection images for this message {message}")
            message.content['error'] = f"Error for Selection images for this message {message}"
            return message

        filtered_df = df[df['image_id'].isin(ai_photos_selected)]
        message.content['gallery_photos_info'] = filtered_df
        message.content['photos'] = ai_photos_selected
        return message

    except Exception as e:
        # self.logger.error(f"Error reading messages: {e}")
        raise(e)
        # return []

def process_message(message, logger):
    # check if its single message or list
    whole_messages_start = datetime.now()

    params = [0.01, 100, 1000, 100, 300, 12]
    logger.debug("Params for this Gallery are: {}".format(params))

    q = mp.Queue()
    p = mp.Process(target=process_crop_images, args=(q, message.content.get('gallery_photos_info')))
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

        if message.content.get('is_wedding', True):
            df, first_last_images_ids, first_last_imgs_df = process_wedding_first_last_image(sorted_df, logger)
        else:
            df, first_last_images_ids, first_last_imgs_df = process_non_wedding_cover_image(sorted_df, logger)

        first_last_design_ids = get_first_last_design_ids(message.designsInfo['anyPagelayouts_df'], logger)

        # Handle the processing time logging
        start = datetime.now()
        message.content['gallery_photos_info'] = df
        album_result = album_processing(df, message.designsInfo, message.content['is_wedding'], params, logger=logger)

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
        if first_last_imgs_df is not None:
            first_last_imgs_df = first_last_imgs_df.merge(cropped_df, how='inner', on='image_id')

        logger.debug('waited for cropping process: {}'.format(datetime.now() - wait_start))

        final_response = assembly_output(album_result, message, df,
                                         first_last_images_ids, first_last_imgs_df, first_last_design_ids, logger)

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
    msgs = read_messages(msgs, logger)
    message = get_selection(msgs[0], logger)

    final_album_result, message = process_message(message, logger)
    return final_album_result, message


if __name__ == '__main__':
    # _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310, 'projectId': 46105850, 'userId': 548864249, 'userJobId': 1069943370, 'base_url': 'ptstorage_32://pictures/46/105/46105850/g59f42f8oml2n45x4s', 'photos': [10772592874, 10772592890, 10772592931, 10772592939, 10772592913, 10772593214, 10772593221, 10772593257, 10772593189, 10772593224, 10772593206, 10772593216, 10772593260, 10772593311, 10772593308, 10772593261, 10772593263, 10772593688, 10772593314, 10772593695, 10772593633, 10772593612, 10772593754, 10772593445, 10772593467, 10772594359, 10772594274, 10772594268, 10772594277, 10772593563, 10772593585, 10772594384, 10772594387, 10772594380, 10772594272, 10772594381, 10772594392], 'projectCategory': 0, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/91qecwibrecpki6p_wcwffdo.json', 'conditionId': 'AAD_46105850_86e53b01-758e-4d5c-8bcd-d43b4e02ec07.326.101', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}
    # _images_path = '/home/a.hyryla/data/pic_time/imagesets/testing/46105850/'
    # _output_pdf_path = '/home/a.hyryla/data/pic_time/results/processed_imagesets/testing/46105850/album2.pdf'
    # _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310, 'projectId': 46245951, 'userId': 548224517, 'userJobId': 1069781153, 'base_url': 'ptstorage_32://pictures/46/245/46245951/ii52fnki40jq0i3xvu', 'photos': [10803725905, 10803725910, 10803725924, 10803725963, 10803725967, 10803725969, 10803725978, 10803725994, 10803725996, 10803725997, 10803726027, 10803726045, 10803726043, 10803726137, 10803726140, 10803726128, 10803726109, 10803726085, 10803726190, 10803726150, 10803726149, 10803726182, 10803726055, 10803726068, 10803726056, 10803726220, 10803726223, 10803726103, 10803726104, 10803726078, 10803726077, 10803726596, 10803726605, 10803726314, 10803726615, 10803726626, 10803726336, 10803726638, 10803726533, 10803726530, 10803726499, 10803726552, 10803726431, 10803726433, 10803726492], 'projectCategory': 0, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/henypeix2kyn2jrspfcd6wae.json', 'conditionId': 'AAD_46245951_bb226506-e333-4b98-bcee-e0ab624b64c9.208.429', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}
    _input_request = {'replyQueueName': 'devaigeneratealbumresponsedto', 'storeId': 32, 'accountId': 475310, 'projectId': 46245951, 'userId': 547128248, 'userJobId': 1068714614, 'base_url': 'ptstorage_32://pictures/46/245/46245951/ii52fnki40jq0i3xvu', 'photos': [10803728384, 10803728502, 10803728538, 10803728560, 10803728527, 10803728549, 10803728582, 10803728547, 10803728593, 10803728385, 10803728396, 10803728407, 10803728418, 10803728429, 10803728440, 10803728571, 10803728451, 10803728462, 10803728473, 10803728484, 10803728503, 10803728514, 10803728519, 10803728520, 10803728521, 10803728522, 10803728523, 10803728524, 10803728525, 10803728526, 10803728528, 10803728529, 10803728530, 10803728531, 10803728532, 10803728533, 10803728534, 10803728550, 10803728535, 10803728536, 10803728537, 10803728548, 10803728539, 10803728540, 10803728541, 10803728542, 10803728543, 10803728551, 10803728544, 10803728545, 10803728546, 10803728552, 10803728555, 10803728553, 10803728554, 10803728556, 10803728558, 10803728419, 10803728557, 10803728420, 10803728559, 10803728561, 10803728562, 10803728563, 10803728564, 10803728565, 10803728566, 10803728567, 10803728568, 10803728569, 10803728570, 10803728572, 10803728573, 10803728577, 10803728578, 10803728579, 10803728580, 10803728574, 10803728581, 10803728576, 10803728583, 10803728584, 10803728585, 10803728586, 10803728587, 10803728588, 10803728589, 10803728575, 10803728590, 10803728591, 10803728592, 10803728594, 10803728595, 10803728596, 10803728597, 10803728598, 10803728599, 10803728600, 10803728601, 10803728386, 10803728602, 10803728603, 10803728387, 10803728388, 10803728389, 10803728390, 10803728391, 10803728392, 10803728393, 10803728394, 10803728397, 10803728398, 10803728399, 10803728400, 10803728401, 10803728402, 10803728403, 10803728404, 10803728405, 10803728395, 10803728406, 10803728408, 10803728409, 10803728410, 10803728411, 10803728412, 10803728413, 10803728414, 10803728415, 10803728416, 10803728417, 10803728421, 10803728422, 10803728423, 10803728424, 10803728425, 10803728426, 10803728439, 10803728441, 10803728442, 10803728428, 10803728427, 10803728432, 10803728436, 10803728430, 10803728435, 10803728433, 10803728434, 10803728431, 10803728437, 10803728438, 10803728443, 10803728444, 10803728445, 10803728446, 10803728447, 10803728448, 10803728449, 10803728450, 10803728452, 10803728453, 10803728454, 10803728455, 10803728456, 10803728457, 10803728458, 10803728459, 10803728460, 10803728461, 10803728463, 10803728464, 10803728465, 10803728466, 10803728467, 10803728468, 10803728469, 10803728470, 10803728471, 10803728472, 10803728474, 10803728475, 10803728476, 10803728477, 10803728478, 10803728479, 10803728480, 10803728481, 10803728482, 10803728483, 10803728485, 10803728486, 10803728487, 10803728488, 10803728489, 10803728490, 10803728491, 10803728492, 10803728501, 10803728493, 10803728504, 10803728505, 10803728511, 10803728509, 10803728510, 10803728512, 10803728513, 10803728517, 10803728515, 10803728518, 10803728516, 10803728506, 10803728507, 10803728508, 10803728383], 'projectCategory': 0, 'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': 'pictures/temp/devaigeneratealbumdto/xzycustiue6zobqvnora4fpa.json', 'aiMetadata': None, 'conditionId': 'AAD_46245951_4038cbc7-0743-464c-af1d-cc8bc280e6bb.171.255', 'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}
    _images_path = '/home/a.hyryla/data/pic_time/imagesets/testing/46245951/'
    _output_pdf_path = '/home/a.hyryla/data/pic_time/results/processed_imagesets/testing/46245951/album_2.1.pdf'
    final_album, _message = process_gallery(_input_request)
    gallery_photos_info = _message.content['gallery_photos_info']

    box_id2data = _message.designsInfo['anyPagebox_id2data'] # if 'designsInfo' in _message and 'anyPagebox_id2data' in _message['designsInfo'] else {}
    visualize_album_to_pdf(final_album, _images_path, _output_pdf_path, box_id2data, gallery_photos_info)

    print(final_album)

