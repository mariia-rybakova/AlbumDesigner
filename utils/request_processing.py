import os
import pandas as pd

from functools import partial
from datetime import datetime

from utils import get_layouts_data
from utils.parser import CONFIGS
from utils.layouts_file import generate_layouts_df
from utils.read_protos_files import get_image_embeddings,get_faces_info,get_persons_ids,get_clusters_info,get_photo_meta,get_person_vectors
from utils.image_queries import generate_query


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
        for func in functions:
            result = func(df, logger)
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
        gallery_info_df = generate_query(CONFIGS["queries_file"], gallery_info_df, num_workers=8)

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
        logger.error("Error in merging results: %s", e)
        return None, None


def read_messages(messages, logger):
    enriched_messages = []

    for _msg in messages:
        reading_message_time = datetime.now()

        json_content = _msg.content
        if not (type(json_content) is dict or type(json_content) is list):
            logger.warning('Incorrect message format: {}.'.format(json_content))

        if 'photos' not in json_content or 'base_url' not in json_content or 'designInfo' not in json_content:
            logger.warning('Incorrect input request: {}. Skipping.'.format(json_content))
            _msg.image = None
            _msg.status = 0
            _msg.error = 'Incorrect message structure: {}. Skipping.'.format(json_content)
            continue
        try:

            images = json_content['photos']
            project_url = json_content['base_url']


            if 'anyPage' in json_content['designInfo']['parts']:
                _msg.anyPageIds = json_content['designInfo']['parts']['anyPage']['designIds']
            else:
                _msg.error = 'no anyPage in designInfo. Skipping.'
                continue
            if 'firstPage' in json_content['designInfo']['parts']:
                _msg.firstPageIds = json_content['designInfo']['parts']['firstPage']['designIds']
            if 'lastPage' in json_content['designInfo']['parts']:
                _msg.lastPageIds = json_content['designInfo']['parts']['lastPage']['designIds']
            if 'cover' in json_content['designInfo']['parts']:
                _msg.coverPageIds = json_content['designInfo']['parts']['cover']['designIds']
                coverPage_layouts_df = generate_layouts_df(json_content['designInfo']['designs'], _msg.anyPageIds)
                _msg.content['coverPage_layouts_df'] = coverPage_layouts_df
            anyPage_layouts_df = generate_layouts_df(json_content['designInfo']['designs'], _msg.anyPageIds)

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
            if not gallery_info_df.empty and not anyPage_layouts_df.empty:
                _msg.content['gallery_photos_info'] = gallery_info_df
                _msg.content['is_wedding'] = is_wedding
                _msg.content['anyPagelayouts_df'] = anyPage_layouts_df
                _msg.content['anyPagelayout_id2data'] = get_layouts_data(anyPage_layouts_df)
                enriched_messages.append(_msg)
            else:
                logger.error(f"Failed to enrich image data for message: {_msg.content}")
                _msg.error = 'Failed to enrich image data for message: {}. Skipping.'.format(json_content)
                continue

            logger.info(
                f"Reading Time Stage for one Gallery  {len(gallery_info_df)} images is: {datetime.now() - reading_message_time} secs.")

        except Exception as e:
            logger.error(f"Error reading messages at reading stage: {e}")
            return None

    return enriched_messages
