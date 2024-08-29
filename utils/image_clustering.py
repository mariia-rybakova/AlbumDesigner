def get_clusters_info(cluster_file, images_dict,logger=None):
    try:
        cluster_info_bytes = PTFile(cluster_file)  # load file
        if not cluster_info_bytes.exists():
            return None
        cluster_info_bytes = cluster_info_bytes.read_blob()
        cluster_descriptor = cluster_vector.ContentClusterMessageWrapper()
        cluster_descriptor.ParseFromString(cluster_info_bytes)

    except Exception as e:
        # logger.warning('Cant load cluster data from server: {}. Loading from local directory.'.format(e))
        print('Cant load cluster data from server: {}. will Load it from local directory.'.format(e))

        # load data locally
        try:
            cluster_descriptor = cluster_vector.ContentClusterMessageWrapper()
            with open(cluster_file, 'rb') as f:
                cluster_descriptor.ParseFromString(f.read())
        except Exception as e:
            # logger.warning('Faces data could not be loaded local: {}'.format(e))
            print('Cant load cluster data from local: {}.'.format(e))
            return None

    if cluster_descriptor.WhichOneof("versions") == 'v1':
        message_data = cluster_descriptor.v1
    else:
        raise ValueError('There is no appropriate version of cluster vector message.')

    images_photos = message_data.photos

    for photo in images_photos:
        image_class = photo.imageClass
        cluster_label = photo.clusterId
        cluster_class = photo.clusterClass
        image_ranking = photo.selectionOrder
        if photo.photoId in images_dict:
            images_dict[photo.photoId].update(
                {'image_class': image_class, "cluster_label": cluster_label, 'cluster_class': cluster_class,
                 'ranking': image_ranking})
        else:
            images_dict[photo.photoId] = {'image_class': image_class, "cluster_label": cluster_label,
                                          'cluster_class': cluster_class, 'ranking': image_ranking}

    return images_dict