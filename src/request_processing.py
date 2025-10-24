import pandas as pd
import numpy as np
import traceback

from datetime import datetime

from utils.configs import CONFIGS
from utils.layouts_tools import generate_layouts_df, get_layouts_data
from utils.read_protos_files import get_info_protobufs
from utils.time_processing import process_gallery_time
from ptinfra.azure.pt_file import PTFile
from ptinfra.utils.gallery import Gallery
import json


def read_layouts_data(message, json_content):
    if 'designInfo' in json_content and json_content['designInfo'] is None:
        if 'designInfoTempLocation' in json_content:
            try:
                fb = PTFile(json_content['designInfoTempLocation'])
                fileBytes = fb.read_blob()
                designInfo = json.loads(fileBytes.decode('utf-8'))
                # logger.info('Read designInfo from blob location: {}'.format(designInfo))
                json_content['designInfo'] = designInfo
                message.content['designInfo'] = designInfo
                album_ar = {'anyPage': 2}
                for part in ['firstPage', 'lastPage', 'anyPage']:
                    if part in designInfo['parts']:
                        album_ar[part] = designInfo['parts'][part]['varient']['productWidth'] / \
                                         designInfo['parts'][part]['varient']['productHeight']
                message.content['album_ar'] = album_ar
            except Exception as e:
                return None, 'Error reading designInfo from blob location {}, error: {}'.format(
                    json_content['designInfoTempLocation'], e)
        else:
            return None, 'Incorrect message structure: {}. Skipping.'.format(json_content)

    message.pagesInfo = dict()
    message.designsInfo = dict()
    message.designsInfo['defaultPackageStyleId'] = json_content['designInfo']['defaultPackageStyleId']

    if 'anyPage' in json_content['designInfo']['parts'] and len(
            json_content['designInfo']['parts']['anyPage']['designIds']) > 0:
        message.designsInfo['anyPageIds'] = json_content['designInfo']['parts']['anyPage']['designIds']
    else:
        message.error = 'no anyPage in designInfo. Skipping.'
        return None, 'No anyPage in designInfo. Skipping message..'

    first_page_layouts_df = None
    last_page_layouts_df = None
    if 'firstPage' in json_content['designInfo']['parts']:
        if len(json_content['designInfo']['parts']['firstPage']['designIds']) > 0:
            message.designsInfo['firstPageDesignIds'] = json_content['designInfo']['parts']['firstPage']['designIds']
            message.pagesInfo['firstPage'] = True
            first_page_layouts_df = generate_layouts_df(json_content['designInfo']['designs'],
                                                       message.designsInfo['firstPageDesignIds'],
                                                       album_ar=message.content.get('album_ar', {'anyPage': 2})['anyPage'])
            message.designsInfo['firstPage_layouts_df'] = first_page_layouts_df

    if 'lastPage' in json_content['designInfo']['parts']:
        if len(json_content['designInfo']['parts']['lastPage']['designIds']) > 0:
            message.designsInfo['lastPageDesignIds'] = json_content['designInfo']['parts']['lastPage']['designIds']
            message.pagesInfo['lastPage'] = True
            last_page_layouts_df = generate_layouts_df(json_content['designInfo']['designs'],
                                                      message.designsInfo['lastPageDesignIds'],
                                                      album_ar=message.content.get('album_ar', {'anyPage': 2})['anyPage'])
            message.designsInfo['lastPage_layouts_df'] = last_page_layouts_df

    if 'cover' in json_content['designInfo']['parts']:
        if len(json_content['designInfo']['parts']['cover']['designIds']) > 0:
            message.designsInfo['coverDesignIds'] = json_content['designInfo']['parts']['cover']['designIds']
            message.pagesInfo['cover'] = True

            # coverPage_layouts_df = generate_layouts_df(json_content['designInfo']['designs'], _msg.designsInfo['coverDesignIds'])
            # _msg.designsInfo['coverPage_layouts_df'] = coverPage_layouts_df

    message.designsInfo['minPages'] = json_content['designInfo']['minPages'] if 'minPages' in json_content[
        'designInfo'] else 1
    message.designsInfo['maxPages'] = json_content['designInfo']['minPages'] if 'maxPages' in json_content[
        'designInfo'] else CONFIGS['max_total_spreads']

    any_page_layouts_df = generate_layouts_df(json_content['designInfo']['designs'], message.designsInfo['anyPageIds'],
                                             album_ar=message.content.get('album_ar', {'anyPage': 2})['anyPage'],
                                             do_mirror=True)

    if not any_page_layouts_df.empty:
        message.designsInfo['anyPagelayouts_df'] = any_page_layouts_df
        layout_id2data, box_id2data = get_layouts_data(any_page_layouts_df, first_page_layouts_df, last_page_layouts_df)
        message.designsInfo['anyPagelayout_id2data'] = layout_id2data
        message.designsInfo['anyPagebox_id2data'] = box_id2data

    return message


def add_scenes_info(gallery_info_df, project_base_url, logger):
    try:
        photos_metadata = Gallery(project_base_url)

        image_id2scene_image_order = dict()
        image_iter = 0
        for scene_idx, scene in enumerate(photos_metadata.scenes):
            for photo in scene.photos:
                try:
                    filename = photo.get_filename()
                    image_id2scene_image_order[np.int64(filename.split('.')[0])] = (scene_idx, image_iter)
                    image_iter += 1
                except Exception as e:
                    pass
    except Exception as e:
        logger.error(f"Error reading scenes info from gallery: {e}")
        return gallery_info_df

    mapped = gallery_info_df['image_id'].map(image_id2scene_image_order)
    mapped_df = pd.DataFrame(mapped.tolist(), columns=['scene_order', 'image_order'])
    for col in ['scene_order', 'image_order']:
        gallery_info_df[col] = mapped_df[col].combine_first(gallery_info_df[col])

    gallery_info_df[['scene_order', 'image_order']] = gallery_info_df[['scene_order', 'image_order']].astype('Int64')

    return gallery_info_df


def read_messages(messages, logger):
    enriched_messages = []

    for _msg in messages:
        reading_message_time = datetime.now()

        json_content = _msg.content
        if not (type(json_content) is dict or type(json_content) is list):
            logger.warning('Incorrect message format: {}.'.format(json_content))
        logger.info('Received message: {}/{}'.format(json_content, _msg))

        if 'photos' not in json_content or 'base_url' not in json_content:
            return None, 'There are missing fields in input request: {}. Skipping.'.format(json_content)

        try:
            message = read_layouts_data(_msg, json_content)
            _msg = message
            json_content = _msg.content

            proto_start = datetime.now()
            project_url = json_content['base_url']
            gallery_info_df, is_wedding, pt_error = get_info_protobufs(project_base_url=project_url, logger=logger)
            if pt_error is not None:
                return None, pt_error
            logger.info(f"Reading Files protos for  {len(gallery_info_df)} images is: {datetime.now() - proto_start} secs.")

            # add scenes info to gallery_info_df
            gallery_info_df = add_scenes_info(gallery_info_df, project_url, logger)

            # add time data
            gallery_info_df = process_gallery_time(_msg, gallery_info_df, logger)

            if not gallery_info_df.empty:
                _msg.content['gallery_photos_info'] = gallery_info_df
                _msg.content['is_wedding'] = is_wedding
                enriched_messages.append(_msg)
            else:
                return None, 'Failed to enrich image data for message: {}. Skipping.'.format(json_content)

            logger.info(
                f"Reading Time Stage for one Gallery  {len(gallery_info_df)} images is: {datetime.now() - reading_message_time} secs. message id: {_msg.source.id}")

        except Exception as ex:
            tb = traceback.extract_tb(ex.__traceback__)
            filename, lineno, func, text = tb[-1]
            return None, f'Error reading messages at reading stage: {ex}. Exception in function: {func}, line {lineno}, file {filename}.'

    return enriched_messages, None


def convert_int64_to_int(obj):
    if isinstance(obj, dict):
        return {key: convert_int64_to_int(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_int64_to_int(item) for item in obj]
    elif isinstance(obj, np.int64):
        return int(obj)
    else:
        return obj


def customize_box(image_info, box_info, album_ar=2):
    target_ar = box_info['width'] / box_info['height'] * album_ar
    if box_info['orientation'] == 'square':
        crop_x = image_info['cropped_x']
        crop_y = image_info['cropped_y']
        crop_w = image_info['cropped_w']
        crop_h = image_info['cropped_h']

        return crop_x, crop_y, crop_w, crop_h
    else:
        image_ar = float(image_info['image_as'])
        if image_ar > target_ar:
            # Image is too wide, crop horizontally
            new_width_ratio = target_ar / image_ar
            x = (1 - new_width_ratio) / 2
            y = 0.0
            w = new_width_ratio
            h = 1.0
        else:
            # Image is too tall, crop vertically
            new_height_ratio = image_ar / target_ar
            x = 0.0
            y = (1 - new_height_ratio) / 2
            w = 1.0
            h = new_height_ratio

        return x, y, w, h


def sort_boxes(boxes):
    xs = [box['x'] for box in boxes]
    ys = [box['y'] for box in boxes]
    page_flag = [0 if box['x'] < 0.5 else 1 for box in boxes]

    sorted_indices = np.lexsort((xs, ys, page_flag))

    sorted_boxes = [boxes[ind] for ind in sorted_indices]
    return sorted_boxes


def get_mirrored_boxes(boxes):
    mirrored_boxes = [box.copy() for box in boxes]
    for mirrored_box in mirrored_boxes:
        if mirrored_box is not None and 'x' in mirrored_box and 'width' in mirrored_box:
            mirrored_box['x'] = 1 - mirrored_box['x'] - mirrored_box['width']

    return sort_boxes(mirrored_boxes)


def assembly_output(output_list, message, images_df, first_last_pages_data_dict, album_ar = 2,logger=None):
    result_dict = dict()
    result_dict['compositions'] = list()
    result_dict['placementsTxt'] = list()
    result_dict['placementsImg'] = list()
    result_dict['userJobId'] = message.content['userJobId']
    result_dict['compositionPackageId'] = message.content['compositionPackageId']
    result_dict['productId'] = message.content['designInfo']['productId']
    result_dict['packageDesignId'] = None
    result_dict['projectId'] = message.content['projectId']
    result_dict['storeId'] = message.content['storeId']
    result_dict['accountId'] = message.content['accountId']
    result_dict['userId'] = message.content['userId']
    counter_comp_id = 0
    counter_image_id = 0

    original_designs_data = message.content['designInfo']['designs']
    layouts_df = message.designsInfo['anyPagelayouts_df']
    box_id2data = message.designsInfo['anyPagebox_id2data']
    # adding the Album Cover
    if 'cover' in message.pagesInfo.keys():
        result_dict['compositions'].append({"compositionId": counter_comp_id,
                                       "compositionPackageId": message.content['compositionPackageId'],
                                       "designId":  message.designsInfo['coverDesignIds'][0] ,
                                       "styleId": message.designsInfo['defaultPackageStyleId'],
                                       "revisionCounter": 0,
                                       "copies": 1,
                                       "boxes": None,
                                       "logicalSelectionsState": None})
        counter_comp_id += 1

    # adding the first spread image
    if 'firstPage' in first_last_pages_data_dict.keys() and first_last_pages_data_dict['firstPage']['first_images_df'] is not None:
        first_page_data = first_last_pages_data_dict['firstPage']
        first_page_layouts_df = message.designsInfo['firstPage_layouts_df']
        design_id = first_page_layouts_df.loc[first_page_data['design_id']]['id']
        if design_id > 0:
            design_boxes = sort_boxes(original_designs_data[str(design_id)]['boxes'])
        else:
            design_boxes = get_mirrored_boxes(original_designs_data[str(-1*design_id)]['boxes'])
            design_id = -1 * design_id
        left_box_ids = first_page_layouts_df.loc[first_page_data['design_id']]['left_box_ids']
        right_box_ids = first_page_layouts_df.loc[first_page_data['design_id']]['right_box_ids']
        all_box_ids = left_box_ids + right_box_ids
        result_dict['compositions'].append({"compositionId": counter_comp_id,
                                       "compositionPackageId": message.content['compositionPackageId'],
                                       "designId": design_id,
                                       "styleId": message.designsInfo['defaultPackageStyleId'],
                                       "revisionCounter": 0,
                                       "copies": 1,
                                       "boxes": design_boxes,
                                       "logicalSelectionsState": None})

        for idx, box_id in enumerate(all_box_ids):
            x, y, w, h = customize_box(first_page_data['first_images_df'].iloc[idx], box_id2data[box_id],album_ar)
            result_dict['placementsImg'].append({"placementImgId": counter_image_id,
                                            "compositionId": counter_comp_id,
                                            "compositionPackageId": message.content['compositionPackageId'],
                                            "boxId": box_id,
                                            "photoId": first_page_data['first_images_ids'][idx],
                                            "cropX": x,
                                            "cropY": y,
                                            "cropWidth": w,
                                            "cropHeight": h,
                                            "rotate": 0,
                                            "projectId": message.content['projectId'],
                                            "photoFilter": 0,
                                            "photo": None})
        counter_comp_id += 1
        counter_image_id += 1


    # Add images
    for number_groups,group_dict in enumerate(output_list):
        for group_name in group_dict.keys():
            group_result = group_dict[group_name]
            # logger.debug('Group name/result: {}: {}'.format(group_name, group_result))
            total_spreads = len(group_result)
            for i in range(total_spreads):
                group_data = group_result[i]
                if isinstance(group_data, float):
                    continue
                if isinstance(group_data, list):
                    number_of_spreads = len(group_data)

                    for spread_index in range(number_of_spreads):
                        layout_id = group_data[spread_index][0]

                        design_id = layouts_df.loc[layout_id]['id']
                        if design_id > 0:
                            design_boxes = sort_boxes(original_designs_data[str(design_id)]['boxes'])
                        else:
                            design_boxes = get_mirrored_boxes(original_designs_data[str(-1 * design_id)]['boxes'])
                            design_id = -1 * design_id

                        result_dict['compositions'].append({"compositionId": counter_comp_id,
                                                       "compositionPackageId": message.content['compositionPackageId'],
                                                       "designId": design_id,
                                                       "styleId": message.designsInfo['defaultPackageStyleId'],
                                                       "revisionCounter": 0,
                                                       "copies": 1,
                                                       "boxes": design_boxes,
                                                       "logicalSelectionsState": None})

                        cur_layout_info = layouts_df.loc[layout_id]['boxes_info']
                        left_box_ids = layouts_df.loc[layout_id]['left_box_ids']
                        right_box_ids = layouts_df.loc[layout_id]['right_box_ids']

                        left_page_photos = list(group_data[spread_index][1])
                        right_page_photos = list(group_data[spread_index][2])

                        all_box_ids = left_box_ids + right_box_ids
                        all_photos = left_page_photos + right_page_photos

                        # Loop over boxes and plot images
                        for j, box in enumerate(cur_layout_info):
                            box_id = box['id']
                            if box_id not in all_box_ids:
                                print('Some error, cant find box with id: {}'.format(box_id))

                            element_index = all_box_ids.index(box_id)
                            cur_photo = all_photos[element_index]
                            image_id = cur_photo.id

                            image_info = images_df[images_df["image_id"] == image_id]
                            if image_info is None or image_info.empty:
                                continue
                            else:
                                x, y, w, h = customize_box(image_info.iloc[0], box_id2data[box_id],album_ar)
                            result_dict['placementsImg'].append({"placementImgId" : counter_image_id,
                                                            "compositionId" : counter_comp_id,
                                                            "compositionPackageId": message.content['compositionPackageId'],
                                                            "boxId" : box_id,
                                                            "photoId" : image_id,
                                                            "cropX" : x,
                                                            "cropY" : y,
                                                            "cropWidth" : w,
                                                            "cropHeight" : h,
                                                            "rotate" : 0,
                                                            "projectId" : message.content['projectId'],
                                                            "photoFilter" : 0,
                                                            "photo" : None})
                            counter_image_id += 1
                        counter_comp_id += 1


    # adding the last page
    if 'lastPage' in first_last_pages_data_dict.keys() and first_last_pages_data_dict['lastPage'][
        'last_images_df'] is not None:
        last_page_data = first_last_pages_data_dict['lastPage']
        last_page_layouts_df = message.designsInfo['lastPage_layouts_df']
        design_id = last_page_layouts_df.loc[last_page_data['design_id']]['id']
        if design_id > 0:
            design_boxes = sort_boxes(original_designs_data[str(design_id)]['boxes'])
        else:
            design_boxes = get_mirrored_boxes(original_designs_data[str(-1*design_id)]['boxes'])
            design_id = -1 * design_id
        left_box_ids = last_page_layouts_df.loc[last_page_data['design_id']]['left_box_ids']
        right_box_ids = last_page_layouts_df.loc[last_page_data['design_id']]['right_box_ids']
        all_box_ids = left_box_ids + right_box_ids
        result_dict['compositions'].append({"compositionId": counter_comp_id,
                                            "compositionPackageId": message.content['compositionPackageId'],
                                            "designId": design_id,
                                            "styleId": message.designsInfo['defaultPackageStyleId'],
                                            "revisionCounter": 0,
                                            "copies": 1,
                                            "boxes": design_boxes,
                                            "logicalSelectionsState": None})

        for idx, box_id in enumerate(all_box_ids):
            x, y, w, h = customize_box(last_page_data['last_images_df'].iloc[idx], box_id2data[box_id],album_ar)
            result_dict['placementsImg'].append({"placementImgId": counter_image_id,
                                                 "compositionId": counter_comp_id,
                                                 "compositionPackageId": message.content['compositionPackageId'],
                                                 "boxId": box_id,
                                                 "photoId": last_page_data['last_images_ids'][idx],
                                                 "cropX": x,
                                                 "cropY": y,
                                                 "cropWidth": w,
                                                 "cropHeight": h,
                                                 "rotate": 0,
                                                 "projectId": message.content['projectId'],
                                                 "photoFilter": 0,
                                                 "photo": None})

    result_dict = convert_int64_to_int(result_dict)

    final_result = {
        'requestId': message.content['conditionId'],
        'error': message.error,
        'composition': result_dict
    }
    return final_result
