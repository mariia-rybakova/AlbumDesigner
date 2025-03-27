import pandas as pd
import numpy as np
from bson.int64 import Int64
from datetime import datetime

from concurrent.futures import ThreadPoolExecutor

from utils import get_layouts_data
from utils.parser import CONFIGS
from utils.protobufs_processing import get_info_protobufs
from utils.layouts_file import generate_layouts_df,generate_layouts_fromDesigns_df
from src.smart_cropping import process_cropping

cached_design_ids = None
cached_layouts_df = None

def process_cropping_for_row(row):
    cropped_x, cropped_y, cropped_w, cropped_h = process_cropping(
        float(row['image_as']),
        row['faces_info'],
        row['background_centroid'],
        float(row['diameter']),
        1
    )
    # Store the results in a dictionary to update the DataFrame later
    return {
        'image_id': row['image_id'],
        'cropped_x': cropped_x,
        'cropped_y': cropped_y,
        'cropped_w': cropped_w,
        'cropped_h': cropped_h
    }

def read_messages(messages,queries_file, logger):
    enriched_messages = []

    global cached_design_ids, cached_layouts_df

    for _msg in messages:
        reading_message_time = datetime.now()
        json_content = _msg.content
        if not (type(json_content) is dict or type(json_content) is list):
            logger.warning('Incorrect message format: {}.'.format(json_content))

        if 'photos' not in json_content or \
                'base_url' not in json_content or 'designInfo' not in json_content or 'projectId' not in json_content:
            logger.warning('Incorrect input request: {}. Skipping.'.format(json_content))
            _msg.image = None
            _msg.status = 0
            _msg.error = 'Incorrect message structure: {}. Skipping.'.format(json_content)
            continue
        try:
            images = json_content['photos']
            project_url = json_content['base_url']



            cached_layouts_df = generate_layouts_fromDesigns_df(json_content['designInfo']['designs'])


            # design_ids = json_content.get('designs', [])

            # if cached_design_ids is None or cached_design_ids != design_ids:
            #     cached_design_ids = design_ids  # Update cache
            #     cached_layouts_df = generate_layouts_df(CONFIGS["designs_json_file_path"], design_ids)

            df = pd.DataFrame(images, columns=['image_id'])
            proto_start = datetime.now()
            # check if its wedding here! and added to the message
            gallery_info_df, is_wedding = get_info_protobufs(project_base_url=project_url, df=df, logger=logger)

            logger.info(f"Reading Files protos for  {len(gallery_info_df)} images is: {datetime.now() - proto_start} secs.")

            # cropping_start = datetime.now()
            # results=[]
            # for _, row in gallery_info_df.iterrows():
            #     results.append(process_cropping_for_row(row))
            #
            #
            # # with ThreadPoolExecutor(max_workers=CONFIGS['cropping_workers']) as executor:
            # #     results = list(executor.map(process_cropping_for_row, [row for _, row in gallery_info_df.iterrows()]))
            #
            # cropped_df = pd.DataFrame(results)
            # # Merge the cropped data back into the original DataFrame
            # gallery_info_df = gallery_info_df.merge(cropped_df, how='inner', on='image_id')
            #
            #
            # logger.info(
            #     f"Cropping time for  {len(gallery_info_df)} images is: {datetime.now() - cropping_start} secs.")

            is_wedding = True
            if not gallery_info_df.empty and not cached_layouts_df.empty:
                _msg.content['gallery_photos_info'] = gallery_info_df
                _msg.content['is_wedding'] = is_wedding
                _msg.content['layouts_df'] = cached_layouts_df
                _msg.content['layout_id2data'] = get_layouts_data(cached_layouts_df)
                enriched_messages.append(_msg)
            else:
                logger.error(f"Failed to enrich image data for message: {_msg.content}")
                _msg.error = 'Failed to enrich image data for message: {}. Skipping.'.format(json_content)
                continue

            logger.info(
                f"Reading Time Stage for one Gallery  {len(gallery_info_df)} images is: {datetime.now() - reading_message_time} secs.")

        except Exception as e:
            logger.error(f"Error reading messages at reading stage: {e}")

    return enriched_messages


def organize_one_message_results(msg, model_version, logger):
    try:
        bg_data = msg.bg_data
    except AttributeError:
        logger.error('No background data found for this message: {}.'.format(msg.content['photoId']))
        bg_data = None

    result_dict = {
            "storeId": msg.content['storeId'],
            "accountId": msg.content['accountId'],
            "projectId": Int64(msg.project_id),  # convert projectId to int64
            "modelVersion": model_version,
            'status': msg.status,
            "processedTime": datetime.utcnow(),
            "aspectRatio" : msg.ar,
            "colorEnum" : msg.color,
        }
    if bg_data is not None:
        result_dict['backgroundMask'] = np.array(bg_data['bg_mask'], dtype=np.uint8).tobytes()
        result_dict['compositionScore'] = bg_data['comp_score']
        result_dict['flatnessScore'] = bg_data['flatness_score']
        result_dict['blobDiameter'] = bg_data['blob_diameter']
        result_dict['blobCentroid'] = [bg_data['centroid'][0], bg_data['centroid'][1]]

    return result_dict