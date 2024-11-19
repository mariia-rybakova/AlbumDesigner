from ptinfra.azure.pt_file import PTFile
from utils.protos  import ContentCluster_pb2 as cluster_vector


def get_clusters_info(cluster_file, images_dict,logger=None):
    try:
        cluster_info_bytes = PTFile(cluster_file)  # load file
        if not cluster_info_bytes.exists():
            return None
        cluster_info_bytes = cluster_info_bytes.read_blob()
        cluster_descriptor = cluster_vector.ContentClusterMessageWrapper()
        cluster_descriptor.ParseFromString(cluster_info_bytes)

    except Exception as e:
        logger.warning('Cant load cluster data from server: {}. Loading from local directory.'.format(e))
        print('Cant load cluster data from server: {}. will Load it from local directory.'.format(e))

    if cluster_descriptor.WhichOneof("versions") == 'v1':
        message_data = cluster_descriptor.v1
    else:
        logger.error('There is no appropriate version of cluster vector message.')
        raise ValueError('There is no appropriate version of cluster vector message.')

    images_photos = message_data.photos

    for photo in images_photos:
        image_class = photo.imageClass
        cluster_label = photo.clusterId
        cluster_class = photo.clusterClass
        image_ranking = photo.selectionScore
        image_order =  photo.selectionOrder

        if photo.photoId in images_dict:
            images_dict[photo.photoId].update(
                {'image_class': image_class, "cluster_label": cluster_label, 'cluster_class': cluster_class,
                 'ranking': image_ranking, 'image_order': image_order})
        else:
            images_dict[photo.photoId] = {'image_class': image_class, "cluster_label": cluster_label,
                                          'cluster_class': cluster_class, 'ranking': image_ranking,'image_order': image_order}

    return images_dict,None