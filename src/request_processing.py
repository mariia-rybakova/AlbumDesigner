import os
import copy
import pandas as pd
import numpy as np

from functools import partial
from datetime import datetime

from utils.configs import CONFIGS,label_list
from utils.layouts_tools import generate_layouts_df, get_layouts_data
from utils.read_protos_files import get_image_embeddings,get_faces_info,get_persons_ids,get_clusters_info,get_photo_meta,get_person_vectors
from utils.image_queries import generate_query
from ptinfra.azure.pt_file import PTFile
import json
from math import isnan

def generate_dict_key(numbers, n_bodies):
    if (numbers == 0 and n_bodies == 0) or (not numbers):
        return 'No PEOPLE'
    if isinstance(numbers, float):
        if isnan(numbers):
            return 'No PEOPLE'

    # Convert the string of numbers into a list
    try:
        id_list = eval(numbers) if isinstance(numbers, str) else numbers
    except:
        return "Invalid_numbers"

    # Calculate the count based on the list length or n_bodies
    count = max(len(id_list), n_bodies) if isinstance(id_list, list) else n_bodies

    # Determine the suffix
    suffix = "person" if count == 1 else "pple"

    # Combine count, suffix, and the numbers joined by underscores
    key = f"{count}_{suffix}_" + "_".join(map(str, id_list))
    return key

def check_gallery_type(df):
    count = 0
    for idx, row in df.iterrows():  # Unpack the tuple into idx (index) and row (data)
        content_class = row['image_class']
        if pd.isna(content_class):
            continue
        if content_class == -1:
            count += 1

    number_images = len(df)

    if number_images > 0 and count / number_images > 0.6:  # Ensure no division by zero
        return False
    else:
        return True


def map_cluster_label(cluster_label):
    if cluster_label == -1:
        return "None"
    elif cluster_label >= 0 and cluster_label < len(label_list):
        return label_list[cluster_label]
    else:
        return "Unknown"

def process_content(row_dict):
    row_dict = copy.deepcopy(row_dict)
    cluster_class = row_dict.get('cluster_class')
    cluster_class_label = map_cluster_label(cluster_class)
    row_dict['cluster_context'] = cluster_class_label
    return row_dict


def _flatten(iterables):
    for x in iterables:
        if isinstance(x, (list, tuple, set)):
            for y in x:
                yield y
        elif pd.notna(x):
            yield x

def pick_from_set(candidates, allowed_set):
    if not isinstance(candidates, (list, tuple, set)):
        return np.nan
    for c in candidates:
        if c in allowed_set:
            return c
    return np.nan

def get_info_protobufs(project_base_url, logger):
    try:
        start = datetime.now()
        image_file = os.path.join(project_base_url, 'ai_search_matrix.pai')
        faces_file = os.path.join(project_base_url, 'ai_face_vectors.pb')
        persons_file = os.path.join(project_base_url, 'persons_info.pb')
        cluster_file = os.path.join(project_base_url, 'content_cluster.pb')
        segmentation_file = os.path.join(project_base_url, 'bg_segmentation.pb')
        person_vector_file = os.path.join(project_base_url, 'ai_person_vectors.pb')
        files = [image_file, faces_file, persons_file, cluster_file, segmentation_file, person_vector_file]

        # List of functions to run in parallel
        functions = [
            partial(get_image_embeddings, image_file),
            partial(get_faces_info, faces_file),
            partial(get_persons_ids, persons_file),
            partial(get_clusters_info, cluster_file),
            partial(get_photo_meta, segmentation_file),
            partial(get_person_vectors, person_vector_file)
        ]

        results = []
        for idx, func in enumerate(functions):
            result = func(logger)
            if result is None:
                return None, None, 'Error in reading data from protobuf file: {}'.format(files[idx])
            elif result.empty or result.shape[0] == 0:
                return None, None, 'There are no required data in protobuf file: {}'.format(files[idx])
            results.append(result)

        gallery_info_df = results[0]
        for res in results[1:]:
            gallery_info_df = pd.merge(gallery_info_df, res, on="image_id", how="outer")

        # Convert only the specified columns to 'Int64' (nullable integer type)
        columns_to_convert = ["image_class", "cluster_label", "cluster_class", "image_order", "scene_order"]
        gallery_info_df[columns_to_convert] = gallery_info_df[columns_to_convert].astype('Int64')

        is_wedding = check_gallery_type(gallery_info_df)

        if is_wedding:
            # make Cluster column
            gallery_info_df = gallery_info_df.apply(process_content, axis=1)
            # gallery_info_df = gallery_info_df.merge(processed_df[['image_id', 'cluster_context']],
            #                                         how='left', on='image_id')
            bride_id, groom_id = np.nan, np.nan

            from collections import Counter
            bride_set = Counter(
                _flatten(gallery_info_df.loc[gallery_info_df["cluster_context"] == "bride", "persons_ids"]))
            groom_set = Counter(
                _flatten(gallery_info_df.loc[gallery_info_df["cluster_context"] == "groom", "persons_ids"]))

            main_row = gallery_info_df["main_persons"].dropna().iloc[0]

            # bride_id = bride_set.most_common(1)[0][0] if bride_set else np.nan
            # groom_id = groom_set.most_common(1)[0][0] if groom_set else np.nan

            if bride_set:
                bride_candidates = [id for id, count in bride_set.most_common() if
                                    count == bride_set.most_common(1)[0][1]]
                bride_id = next((id for id in bride_candidates if id in main_row),
                                bride_candidates[0]) if bride_candidates else np.nan
            else:
                bride_id = np.nan


            if groom_set:
                groom_candidates = [id for id, count in groom_set.most_common() if
                                    count == groom_set.most_common(1)[0][1]]
                groom_id = next((id for id in groom_candidates if id in main_row),
                                groom_candidates[0]) if groom_candidates else np.nan
            else:
                groom_id = np.nan

            if np.isnan(bride_id) and not np.isnan(groom_id):
                for person_id in main_row:
                    if person_id != groom_id:
                        bride_id = person_id
                        break
            elif np.isnan(groom_id) and not np.isnan(bride_id):
                for person_id in main_row:
                    if person_id != bride_id:
                        groom_id = person_id
                        break
            elif np.isnan(bride_id) and np.isnan(groom_id):
                if len(main_row) >= 2:
                    bride_id = main_row[0]
                    groom_id = main_row[1]

            if groom_id not in main_row or bride_id not in main_row:
                logger.warning(f"Main persons {main_row} do not contain bride {bride_id} or groom {groom_id}")

            gallery_info_df["bride_id"] = bride_id
            gallery_info_df["groom_id"] = groom_id

            gallery_info_df["main_persons"] = gallery_info_df["main_persons"].apply(
                lambda x: x if isinstance(x, (list, tuple)) else []
            )

            gallery_info_df["persons_ids"] = gallery_info_df["persons_ids"].apply(
                lambda x: x if isinstance(x, (list, tuple)) else []
            )



        # Get Query Content of each image
        if gallery_info_df is not None:
            model_version = gallery_info_df.iloc[0]['model_version']
            if model_version == 1:
                gallery_info_df = generate_query(CONFIGS["queries_file_v2"], gallery_info_df, num_workers=8)
            else:
                #gallery_info_df = generate_query(CONFIGS["queries_file_v2"], gallery_info_df, num_workers=8)
                gallery_info_df = generate_query(CONFIGS["queries_file_v3"], gallery_info_df, num_workers=8)

        logger.debug("Number of images before cleaning the nan values: {}".format(len(gallery_info_df.index)))
        columns_to_check = ["ranking", "image_order", "image_class", "cluster_label", "cluster_class"]
        gallery_info_df = gallery_info_df.dropna(subset=columns_to_check)
        logger.debug("Number of images after cleaning the nan values: {}".format(len(gallery_info_df.index)))
        # make sure it has list values not float nan

        # Cluster people by number of people inside the image
        gallery_info_df['people_cluster'] = gallery_info_df.apply(lambda row: generate_dict_key(row['persons_ids'], row['number_bodies']), axis=1)
        logger.debug("Time for reading files: {}".format(datetime.now() - start))
        return gallery_info_df, is_wedding, None

    except Exception as e:
        return None, None, f'Error in reading protobufs: {e}'


def read_messages(messages, logger):
    enriched_messages = []

    for _msg in messages:
        reading_message_time = datetime.now()

        json_content = _msg.content
        if not (type(json_content) is dict or type(json_content) is list):
            logger.warning('Incorrect message format: {}.'.format(json_content))
        logger.info('Received message: {}/{}'.format(json_content, _msg))
        if 'designInfo' in json_content and json_content['designInfo'] is None:
            if 'designInfoTempLocation' in json_content:
                try:
                    fb = PTFile(json_content['designInfoTempLocation'])
                    fileBytes = fb.read_blob()
                    designInfo = json.loads(fileBytes.decode('utf-8'))
                    # logger.info('Read designInfo from blob location: {}'.format(designInfo))
                    json_content['designInfo'] = designInfo
                    _msg.content['designInfo'] = designInfo
                    album_ar = {'anyPage': 2}
                    for part in ['firstPage', 'lastPage', 'anyPage']:
                        if part in designInfo['parts']:
                            album_ar[part] = designInfo['parts'][part]['varient']['productWidth']/designInfo['parts'][part]['varient']['productHeight']
                    _msg.content['album_ar'] = album_ar
                except Exception as e:
                    return None, 'Error reading designInfo from blob location {}, error: {}'.format(json_content['designInfoTempLocation'], e)
            else:
                return None, 'Incorrect message structure: {}. Skipping.'.format(json_content)

        if 'photos' not in json_content or 'base_url' not in json_content or 'designInfo' not in json_content:
            return None, 'There are missing fields in input request: {}. Skipping.'.format(json_content)

        try:
            project_url = json_content['base_url']

            _msg.pagesInfo = dict()
            _msg.designsInfo = dict()
            _msg.designsInfo['defaultPackageStyleId'] = json_content['designInfo']['defaultPackageStyleId']

            if 'anyPage' in json_content['designInfo']['parts'] and len(json_content['designInfo']['parts']['anyPage']['designIds'])>0:
                _msg.designsInfo['anyPageIds'] = json_content['designInfo']['parts']['anyPage']['designIds']
            else:
                _msg.error = 'no anyPage in designInfo. Skipping.'
                return None, 'No anyPage in designInfo. Skipping message..'
            firstPage_layouts_df = None
            lastPage_layouts_df = None
            if 'firstPage' in json_content['designInfo']['parts']:
                if len(json_content['designInfo']['parts']['firstPage']['designIds']) > 0:
                    _msg.designsInfo['firstPageDesignIds'] = json_content['designInfo']['parts']['firstPage']['designIds']
                    _msg.pagesInfo['firstPage'] = True
                    firstPage_layouts_df = generate_layouts_df(json_content['designInfo']['designs'], _msg.designsInfo['firstPageDesignIds'], album_ar=_msg.content.get('album_ar', {'anyPage':2})['anyPage'])
                    _msg.designsInfo['firstPage_layouts_df'] = firstPage_layouts_df

            if 'lastPage' in json_content['designInfo']['parts']:
                if len(json_content['designInfo']['parts']['lastPage']['designIds']) > 0:
                    _msg.designsInfo['lastPageDesignIds'] = json_content['designInfo']['parts']['lastPage']['designIds']
                    _msg.pagesInfo['lastPage'] = True
                    lastPage_layouts_df = generate_layouts_df(json_content['designInfo']['designs'], _msg.designsInfo['lastPageDesignIds'], album_ar=_msg.content.get('album_ar', {'anyPage':2})['anyPage'])
                    _msg.designsInfo['lastPage_layouts_df'] = lastPage_layouts_df

            if 'cover' in json_content['designInfo']['parts']:
                if len(json_content['designInfo']['parts']['cover']['designIds']) > 0:
                    _msg.designsInfo['coverDesignIds'] = json_content['designInfo']['parts']['cover']['designIds']
                    _msg.pagesInfo['cover'] = True

                    # coverPage_layouts_df = generate_layouts_df(json_content['designInfo']['designs'], _msg.designsInfo['coverDesignIds'])
                    # _msg.designsInfo['coverPage_layouts_df'] = coverPage_layouts_df

            _msg.designsInfo['minPages'] = json_content['designInfo']['minPages'] if 'minPages' in json_content['designInfo'] else 1
            _msg.designsInfo['maxPages'] = json_content['designInfo']['minPages'] if 'maxPages' in json_content['designInfo'] else CONFIGS['max_total_spreads']

            anyPage_layouts_df = generate_layouts_df(json_content['designInfo']['designs'], _msg.designsInfo['anyPageIds'], album_ar=_msg.content.get('album_ar', {'anyPage':2})['anyPage'],do_mirror=True)

            proto_start = datetime.now()

            # check if its wedding here! and added to the message
            gallery_info_df, is_wedding, pt_error = get_info_protobufs(project_base_url=project_url, logger=logger)
            if pt_error is not None:
                return None, pt_error


            logger.info(f"Reading Files protos for  {len(gallery_info_df)} images is: {datetime.now() - proto_start} secs.")

            if not gallery_info_df.empty and not anyPage_layouts_df.empty:
                _msg.content['gallery_photos_info'] = gallery_info_df
                _msg.content['is_wedding'] = is_wedding
                _msg.designsInfo['anyPagelayouts_df'] = anyPage_layouts_df
                layout_id2data, box_id2data = get_layouts_data(anyPage_layouts_df, firstPage_layouts_df, lastPage_layouts_df)
                _msg.designsInfo['anyPagelayout_id2data'] = layout_id2data
                _msg.designsInfo['anyPagebox_id2data'] = box_id2data
                enriched_messages.append(_msg)
            else:
                return None, 'Failed to enrich image data for message: {}. Skipping.'.format(json_content)

            logger.info(
                f"Reading Time Stage for one Gallery  {len(gallery_info_df)} images is: {datetime.now() - reading_message_time} secs. message id: {_msg.source.id}")

        except Exception as e:
            return None, f'Error reading messages at reading stage: {e}'

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
    sorted_boxes = sorted(boxes, key=lambda x: (x['y'], x['x']))
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
            design_boxes = original_designs_data[str(design_id)]['boxes']
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
                            design_boxes = original_designs_data[str(design_id)]['boxes']
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
            design_boxes = original_designs_data[str(design_id)]['boxes']
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
