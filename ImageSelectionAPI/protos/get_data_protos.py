import os
import concurrent.futures
import pandas as pd
from functools import partial

from faces import get_faces_info
from photo_meta import get_photo_meta
from embedding import get_image_embeddings
from clustering import get_clusters_info
from persons import get_persons_ids,get_person_vectors
from ImageSelectionAPI.utils.parser import CONFIGS


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

def generate_people_clustering(df):
    # Assuming generate_dict_key is a function that can be applied element-wise
    df['people_cluster'] = df.apply(lambda row: generate_dict_key(row['persons_ids'], row['number_bodies']), axis=1)
    return df


def get_info_protobufs(project_base_url, logger):
    # Initialize an empty DataFrame
    df = pd.DataFrame()

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
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIGS['max_reading_workers']) as executor:
        future_to_function = {executor.submit(func, df, logger): func for func in functions}

        for future in concurrent.futures.as_completed(future_to_function):
            func = future_to_function[future]
            try:
                result = future.result()
                if result is None:
                    logger.error("Error in function: %s", func)
                    return None
                results.append(result)
            except Exception as e:
                logger.error("Exception in function %s: %s", func, e)
                return None

    # Merge results (assuming they return modified df)
    gallery_info_df = results[0]
    for res in results[1:]:
        gallery_info_df = gallery_info_df.combine_first(res)  # Merge dataframes

    # Cluster people by number of people inside the image
    gallery_info_df = generate_people_clustering(gallery_info_df)
    is_wedding = check_gallery_type(gallery_info_df)

    logger.info("Reading from protobuf files has been finished successfully!")

    return gallery_info_df, is_wedding
