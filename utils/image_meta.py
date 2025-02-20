import pandas as pd
from ptinfra.azure.pt_file import PTFile
from utils.protos import BGSegmentation_pb2 as meta_vector


def get_photo_meta(file, df, logger=None):
    required_ids = set(df['image_id'].tolist())
    try:
        meta_info_bytes = PTFile(file)  # load file
        if not meta_info_bytes.exists():
            logger.error(f"the meta file {file} does not exist on the server")
            return None
        meta_info_bytes_info_bytes = meta_info_bytes.read_blob()
        meta_descriptor = meta_vector.PhotoBGSegmentationMessageWrapper()
        meta_descriptor.ParseFromString(meta_info_bytes_info_bytes)

    except Exception as e:
        logger.warning('Cant load cluster data from server: {}. Loading from local directory.'.format(e))
        return None

    if meta_descriptor.WhichOneof("versions") == 'v1':
        message_data = meta_descriptor.v1
    else:
        logger.warning('There is no appropriate version of BGSegmentation vector message.')
        raise ValueError('There is no appropriate version of BGSegmentation vector message.')

    images_photos = message_data.photos

    # Prepare lists to collect data
    photo_ids = []
    image_times = []
    scene_orders = []
    image_aspects = []
    image_colors = []
    image_orientations = []
    image_orderInScenes = []
    background_centroids = []
    blob_diameters = []

    # Loop through each photo and collect the required information
    for photo in images_photos:
        if photo.photoId in required_ids:
            photo_ids.append(photo.photoId)
            image_times.append(photo.dateTaken)
            scene_orders.append(photo.sceneOrder)
            image_aspects.append(photo.aspectRatio)
            image_colors.append(photo.colorEnum)
            image_orientations.append('landscape' if photo.aspectRatio >= 1 else 'portrait')
            image_orderInScenes.append(photo.orderInScene)
            background_centroids.append(photo.blobCentroid)
            blob_diameters.append(photo.blobDiameter)

    # Create a DataFrame from the collected data
    additional_image_info_df = pd.DataFrame({
        'image_id': photo_ids,
        'image_time': image_times,
        'scene_order': scene_orders,
        'image_as': image_aspects,
        'image_color': image_colors,
        'image_orientation': image_orientations,
        'image_orderInScene': image_orderInScenes,
        'background_centroid': background_centroids,
        'diameter': blob_diameters
    })

    # Merge the original DataFrame with the new information
    df = df.merge(additional_image_info_df, how='left', on='image_id')

    return df