import pandas as pd
from ptinfra.azure.pt_file import PTFile
from .files  import ContentCluster_pb2 as cluster_vector

def get_clusters_info(cluster_file,logger=None):
    try:
        cluster_info_bytes = PTFile(cluster_file)  # load file
        if not cluster_info_bytes.exists():
            return None
        cluster_info_bytes = cluster_info_bytes.read_blob()
        cluster_descriptor = cluster_vector.ContentClusterMessageWrapper()
        cluster_descriptor.ParseFromString(cluster_info_bytes)

    except Exception as e:
        logger.warning('Cant load cluster data from server: {}. Loading from local directory.'.format(e))
        return None

    if cluster_descriptor.WhichOneof("versions") == 'v1':
        message_data = cluster_descriptor.v1
    else:
        logger.error('There is no appropriate version of cluster vector message.')
        raise ValueError('There is no appropriate version of cluster vector message.')

    images_photos = message_data.photos

    # Prepare lists to collect data
    photo_ids = []
    image_classes = []
    cluster_labels = []
    cluster_classes = []
    image_rankings = []
    image_orders = []

    # Loop through each photo and collect the required information
    for photo in images_photos:
            photo_ids.append(photo.photoId)
            image_classes.append(photo.imageClass)
            cluster_labels.append(photo.clusterId)
            cluster_classes.append(photo.clusterClass)
            image_rankings.append(photo.selectionScore)
            image_orders.append(photo.selectionOrder)

    # Create a DataFrame from the collected data
    new_image_info_df = pd.DataFrame({
        'image_id': photo_ids,
        'image_class': image_classes,
        'cluster_label': cluster_labels,
        'cluster_class': cluster_classes,
        'ranking': image_rankings,
        'image_order': image_orders
    })

    return new_image_info_df