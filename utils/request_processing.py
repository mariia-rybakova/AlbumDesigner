import os
import pandas as pd
import numpy as np

from functools import partial
from datetime import datetime

from utils import get_layouts_data
from utils.parser import CONFIGS
from utils.layouts_file import generate_layouts_df
from utils.read_protos_files import get_image_embeddings,get_faces_info,get_persons_ids,get_clusters_info,get_photo_meta,get_person_vectors
from utils.image_queries import generate_query
from ptinfra.azure.pt_file import PTFile
import json


def generate_dict_key(numbers, n_bodies):
    if numbers == 0 and n_bodies == 0 or not numbers:
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
        if content_class == -1:
            count += 1

    number_images = len(df)

    if number_images > 0 and count / number_images > 0.6:  # Ensure no division by zero
        return False
    else:
        return True

def get_info_protobufs(project_base_url, df, logger):
    try:
        start = datetime.now()
        faces_file = os.path.join(project_base_url, 'ai_face_vectors.pb')
        cluster_file = os.path.join(project_base_url, 'content_cluster.pb')
        persons_file = os.path.join(project_base_url, 'persons_info.pb')
        image_file = os.path.join(project_base_url, 'ai_search_matrix.pai')
        segmentation_file = os.path.join(project_base_url, 'bg_segmentation.pb')
        person_vector_file = os.path.join(project_base_url, 'ai_person_vectors.pb')
        files = [faces_file, cluster_file, persons_file, image_file, segmentation_file, person_vector_file]

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
            result = func(df, logger)
            if result is None:
                logger.error('Error in reading data from protobuf file: {}'.format(files[idx]))
                raise Exception('Error in reading data from protobuf file: {}'.format(files[idx]))
            results.append(result)

        # if None in results:
        #     logger.error('Error in reading files.')
        #     return None, None

        logger.debug("Time for getting from files: {}".format(datetime.now() - start))

        gallery_info_df = results[0]
        for res in results[1:]:
            gallery_info_df = gallery_info_df.combine_first(res)  # Merge dataframes

        # Convert only the specified columns to 'Int64' (nullable integer type)
        columns_to_convert = ["image_class", "cluster_label", "cluster_class", "image_order", "scene_order"]
        gallery_info_df[columns_to_convert] = gallery_info_df[columns_to_convert].astype('Int64')
        print("Number of images before cleaning the nan values", len(gallery_info_df.index))

        # Get Query Content of each image
        if gallery_info_df is not None:
            model_version = gallery_info_df.iloc[0]['model_version']
            if model_version == 1:
                gallery_info_df = generate_query(CONFIGS["queries_file"], gallery_info_df, num_workers=8)
            else:
                gallery_info_df = generate_query(CONFIGS["queries_file_v2"], gallery_info_df, num_workers=8)

        columns_to_check = ["ranking", "image_order", "image_class", "cluster_label", "cluster_class"]
        gallery_info_df = gallery_info_df.dropna(subset=columns_to_check)
        print("Number of images after cleaning the nan values", len(gallery_info_df.index))

        # make sure it has list values not float nan
        gallery_info_df['persons_ids'] = gallery_info_df['persons_ids'].apply(lambda x: x if isinstance(x, list) else [])

        # Cluster people by number of people inside the image
        gallery_info_df['people_cluster'] = gallery_info_df.apply(lambda row: generate_dict_key(row['persons_ids'], row['number_bodies']), axis=1)
        is_wedding = check_gallery_type(gallery_info_df)

        logger.debug("Time for reading files: {}".format(datetime.now() - start))
        return gallery_info_df, is_wedding

    except Exception as e:
        logger.error("Error in reading protobufs: %s", e)
        raise Exception(f'Error in reading protobufs: {e}')


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
                    logger.info('Read designInfo from blob location: {}'.format(designInfo))
                    json_content['designInfo'] = designInfo
                    _msg.content['designInfo'] = designInfo
                except Exception as e:
                    logger.error('Error reading designInfo from blob location {}, error: {}'.format(json_content['designInfoTempLocation'],e))
                    _msg.image = None
                    _msg.status = 0
                    _msg.error = 'Error reading designInfo from blob: {}'.format(e)
                    raise(e)
                    continue
            else:
                logger.error('Incorrect input request: {}. Skipping.'.format(json_content))
                _msg.image = None
                _msg.status = 0
                _msg.error = 'Incorrect message structure: {}. Skipping.'.format(json_content)
                raise(Exception('Incorrect message structure: {}. Skipping.'.format(json_content)))
                # continue

        if 'photos' not in json_content or 'base_url' not in json_content or 'designInfo' not in json_content:
            logger.warning('Incorrect input request: {}. Skipping.'.format(json_content))
            _msg.image = None
            _msg.status = 0
            _msg.error = 'Incorrect message structure: {}. Skipping.'.format(json_content)
            raise (Exception('Incorrect message structure: {}. Skipping.'.format(json_content)))
            # continue

        if len(json_content['photos'])<10:
            logger.warning('Not enough photos: {}. Skipping.'.format(json_content))
            _msg.image = None
            _msg.status = 0
            _msg.error = 'Not enough photos: {}. Skipping.'.format(json_content)
            raise(Exception('Not enough photos: {}. Skipping.'.format(json_content)))
            # continue

        try:
            images = json_content['photos']
            project_url = json_content['base_url']

            _msg.pagesInfo = dict()
            _msg.designsInfo = dict()
            _msg.designsInfo['defaultPackageStyleId'] = json_content['designInfo']['defaultPackageStyleId']

            if 'anyPage' in json_content['designInfo']['parts']:
                _msg.designsInfo['anyPageIds'] = json_content['designInfo']['parts']['anyPage']['designIds']
            else:
                _msg.error = 'no anyPage in designInfo. Skipping.'
                continue
            if 'firstPage' in json_content['designInfo']['parts']:
                _msg.designsInfo['firstPageDesignIds'] = json_content['designInfo']['parts']['firstPage']['designIds']
                _msg.pagesInfo['firstPage'] = True

            if 'lastPage' in json_content['designInfo']['parts']:
                _msg.designsInfo['lastPageDesignIds'] = json_content['designInfo']['parts']['lastPage']['designIds']
                _msg.pagesInfo['lastPage'] = True

            if 'cover' in json_content['designInfo']['parts']:
                _msg.designsInfo['coverDesignIds'] = json_content['designInfo']['parts']['cover']['designIds']
                _msg.pagesInfo['cover'] = True

                coverPage_layouts_df = generate_layouts_df(json_content['designInfo']['designs'], _msg.designsInfo['coverDesignIds'])
                _msg.designsInfo['coverPage_layouts_df'] = coverPage_layouts_df


            anyPage_layouts_df = generate_layouts_df(json_content['designInfo']['designs'], _msg.designsInfo['anyPageIds'])

            df = pd.DataFrame(images, columns=['image_id'])
            proto_start = datetime.now()

            # check if its wedding here! and added to the message
            gallery_info_df, is_wedding = get_info_protobufs(project_base_url=project_url, df=df, logger=logger)

            logger.info(f"Reading Files protos for  {len(gallery_info_df)} images is: {datetime.now() - proto_start} secs.")

            is_wedding = True
            if not gallery_info_df.empty and not anyPage_layouts_df.empty:
                _msg.content['gallery_photos_info'] = gallery_info_df
                _msg.content['is_wedding'] = is_wedding
                _msg.designsInfo['anyPagelayouts_df'] = anyPage_layouts_df
                layout_id2data, box_id2data = get_layouts_data(anyPage_layouts_df)
                _msg.designsInfo['anyPagelayout_id2data'] = layout_id2data
                _msg.designsInfo['anyPagebox_id2data'] = box_id2data
                enriched_messages.append(_msg)
            else:
                logger.error(f"Failed to enrich image data for message: {_msg.content}")
                _msg.error = 'Failed to enrich image data for message: {}. Skipping.'.format(json_content)
                raise (Exception('Failed to enrich image data for message: {}. Skipping.'.format(json_content)))
                continue

            logger.info(
                f"Reading Time Stage for one Gallery  {len(gallery_info_df)} images is: {datetime.now() - reading_message_time} secs. message id: {_msg.source.id}")

        except Exception as e:
            logger.error(f"Error reading messages at reading stage: {e}")
            return None

    return enriched_messages


def convert_int64_to_int(obj):
    if isinstance(obj, dict):
        return {key: convert_int64_to_int(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_int64_to_int(item) for item in obj]
    elif isinstance(obj, np.int64):
        return int(obj)
    else:
        return obj


def customize_box(image_info, box_info):
    target_ar = box_info['width'] / box_info['height'] * 2
    if box_info['orientation'] == 'square':
        crop_x = image_info['cropped_x']
        crop_y = image_info['cropped_y']
        crop_w = image_info['cropped_w']
        crop_h = image_info['cropped_h']
        image_ar = crop_w / crop_h

        if image_ar > target_ar:
            # Crop is too wide → reduce width
            new_crop_w = crop_h * target_ar
            dx = (crop_w - new_crop_w) / 2
            adj_x = crop_x + dx
            adj_y = crop_y
            adj_w = new_crop_w
            adj_h = crop_h
        else:
            # Crop is too tall → reduce height
            new_crop_h = crop_w / target_ar
            dy = (crop_h - new_crop_h) / 2
            adj_x = crop_x
            adj_y = crop_y + dy
            adj_w = crop_w
            adj_h = new_crop_h

        return adj_x, adj_y, adj_w, adj_h
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


def assembly_output(output_list, message, images_df, first_last_images_ids, first_last_images_df, first_last_design_ids):
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
    if 'firstPage' in message.pagesInfo.keys() and first_last_images_df is not None:
        design_id = layouts_df.loc[first_last_design_ids[0]]['id']
        left_box_ids = layouts_df.loc[first_last_design_ids[0]]['left_box_ids']
        right_box_ids = layouts_df.loc[first_last_design_ids[0]]['right_box_ids']
        all_box_ids = left_box_ids + right_box_ids
        result_dict['compositions'].append({"compositionId": counter_comp_id,
                                       "compositionPackageId": message.content['compositionPackageId'],
                                       "designId": design_id,
                                       "styleId": message.designsInfo['defaultPackageStyleId'],
                                       "revisionCounter": 0,
                                       "copies": 1,
                                       "boxes": None,
                                       "logicalSelectionsState": None})

        x, y, w, h = customize_box(first_last_images_df.iloc[0], box_id2data[all_box_ids[0]])
        result_dict['placementsImg'].append({"placementImgId": counter_image_id,
                                        "compositionId": 2,
                                        "compositionPackageId": message.content['compositionPackageId'],
                                        "boxId": all_box_ids[0],
                                        "photoId": first_last_images_ids[0],
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
            total_spreads = len(group_result)
            for i in range(total_spreads):
                group_data = group_result[i]
                if isinstance(group_data, float):
                    continue
                if isinstance(group_data, list):
                    number_of_spreads = len(group_data)

                    for spread_index in range(number_of_spreads):
                        layout_id = group_data[spread_index][0]

                        result_dict['compositions'].append({"compositionId": counter_comp_id,
                                                       "compositionPackageId": message.content['compositionPackageId'],
                                                       "designId": layouts_df.loc[layout_id]['id'],
                                                       "styleId": message.designsInfo['defaultPackageStyleId'],
                                                       "revisionCounter": 0,
                                                       "copies": 1,
                                                       "boxes": None,
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
                            x, y, w, h = customize_box(image_info.iloc[0], box_id2data[box_id])
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
    if 'lastPage' in message.pagesInfo.keys() and first_last_images_df is not None:
        design_id = layouts_df.loc[first_last_design_ids[1]]['id']
        left_box_ids = layouts_df.loc[first_last_design_ids[1]]['left_box_ids']
        right_box_ids = layouts_df.loc[first_last_design_ids[1]]['right_box_ids']
        all_box_ids = left_box_ids + right_box_ids
        result_dict['compositions'].append({"compositionId": counter_comp_id,
                                       "compositionPackageId": message.content['compositionPackageId'],
                                       "designId": design_id,
                                       "styleId": message.designsInfo['defaultPackageStyleId'],
                                       "revisionCounter": 0,
                                       "copies": 1,
                                       "boxes": None,
                                       "logicalSelectionsState": None})

        x, y, w, h = customize_box(first_last_images_df.iloc[1], box_id2data[all_box_ids[1]])
        result_dict['placementsImg'].append({"placementImgId":  counter_image_id,
                                        "compositionId": counter_comp_id,
                                        "compositionPackageId": message.content['compositionPackageId'],
                                        "boxId": all_box_ids[1],
                                        "photoId": first_last_images_ids[1],
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
