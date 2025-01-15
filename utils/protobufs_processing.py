import os
from utils import image_meta, image_faces, image_persons, image_embeddings, image_clustering
from utils.image_queries import generate_query

def get_info_protobufs(project_base_url, df, logger):
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
    queries_file = ''
    gallery_info_df = generate_query(queries_file, gallery_info_df,logger)

    return gallery_info_df
