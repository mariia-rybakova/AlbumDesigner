import os
from utils import image_meta, image_faces, image_persons, image_embeddings, image_clustering
from utils.image_queries import generate_query


def generate_dict_key(numbers, n_bodies):
    if not numbers:
        return "No_numbers"

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


def get_info_protobufs(project_base_url,df,queries_file, logger):
    faces_file = os.path.join(project_base_url, 'ai_face_vectors.pb')
    cluster_file = os.path.join(project_base_url, 'content_cluster.pb')
    persons_file = os.path.join(project_base_url, 'persons_info.pb')
    image_file = os.path.join(project_base_url, 'ai_search_matrix.pai')
    segmentation_file = os.path.join(project_base_url, 'bg_segmentation.pb')

    # Get info from protobuf files server
    gallery_info_df = image_embeddings.get_image_embeddings(image_file,df,logger)
    if gallery_info_df is None:
         logger.error('Embeddings for images NOT FOUND in file %s', image_file)
         return None

    gallery_info_df = image_faces.get_faces_info(faces_file, gallery_info_df,logger)
    if gallery_info_df is None:
        logger.error('Faces info NOT FOUND for file %s', faces_file)
        return None

    gallery_info_df = image_persons.get_persons_ids(persons_file, gallery_info_df,logger)
    if gallery_info_df is None:
        logger.error('Persons info NOT FOUND for file %s', persons_file)
        return None

    gallery_info_df = image_clustering.get_clusters_info(cluster_file, gallery_info_df,logger)
    if gallery_info_df is None:
        logger.error('Clusters info Not Found for file %s', cluster_file)
        return None

    gallery_info_df = image_meta.get_photo_meta(segmentation_file, gallery_info_df,logger)
    if gallery_info_df is None:
        logger.error('Photo meta NOT FOUND for file %s', segmentation_file)
        return None

    # Get Query Content of each image
    gallery_info_df = generate_query(queries_file, gallery_info_df,logger)
    # cluster people by number of people inside the image
    gallery_info_df = generate_people_clustering(gallery_info_df)

    return gallery_info_df
