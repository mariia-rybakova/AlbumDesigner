import pandas as pd
import numpy as np
from bson.int64 import Int64
from datetime import datetime

from utils.protobufs_processing import get_info_protobufs

def read_messages(messages,queries_file, logger):
    enriched_messages = []
    for _msg in messages:
        json_content = _msg.content
        if not (type(json_content) is dict or type(json_content) is list):
            logger.warning('Incorrect message format: {}.'.format(json_content))

        if 'photosIds' not in json_content or \
                'projectURL' not in json_content or \
                'storeId' not in json_content or\
                'layoutsCSV' not in json_content or\
                'sendTime' not in json_content:
            logger.warning('Incorrect input request: {}. Skipping.'.format(json_content))
            _msg.image = None
            _msg.status = 0
            _msg.error = 'Incorrect message structure: {}. Skipping.'.format(json_content)
            continue
        try:
            images = json_content['photosIds']
            project_url = json_content.get('projectURL', '')

            if not project_url or not images:
                logger.warning(f"Incomplete message content: {_msg.content}")
                _msg.error = 'Incomplete message content Project URL: {}. Skipping.'.format(json_content)
                continue

            df = pd.DataFrame(images, columns=['image_id'])
            # check if its wedding here! and added to the message
            gallery_info_df, is_wedding = get_info_protobufs(project_base_url=project_url, df=df,
                                                             queries_file=queries_file, logger=logger)

            if not gallery_info_df.empty and is_wedding:
                _msg.content['gallery_photos_info'] = gallery_info_df
                _msg.content['is_wedding'] = is_wedding
                enriched_messages.append(_msg)
            else:
                logger.error(f"Failed to enrich image data for message: {_msg.content}")
                _msg.error = 'Failed to enrich image data for message: {}. Skipping.'.format(json_content)
                continue

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