from ptinfra.azure.pt_file import PTFile
from utils.protos import BGSegmentation_pb2 as meta_vector



def get_photo_meta(file, images_dict, logger=None):
    try:
        meta_info_bytes = PTFile(file)  # load file
        if not meta_info_bytes.exists():
            return images_dict
        meta_info_bytes_info_bytes = meta_info_bytes.read_blob()
        meta_descriptor = meta_vector.PhotoBGSegmentationMessageWrapper()
        meta_descriptor.ParseFromString(meta_info_bytes_info_bytes)

    except Exception as e:
        # logger.warning('Cant load cluster data from server: {}. Loading from local directory.'.format(e))
        print('Cant load BGSegmentation data from server: {}. trying to Load it from local directory.'.format(e))

        # load data locally
        try:
            meta_descriptor = meta_vector.PhotoBGSegmentationMessageWrapper()
            with open(r'C:\Users\karmel\Desktop\PicTime\Projects\AlbumDesigner\proto\proto_files\ai_bgsegmentation.pb',
                      'rb') as f:
                meta_descriptor.ParseFromString(f.read())
        except Exception as e:
            # logger.warning('Faces data could not be loaded local: {}'.format(e))
            print('Cant load BGSegmentation  data from local: {}.'.format(e))
            return None

    if meta_descriptor.WhichOneof("versions") == 'v1':
        message_data = meta_descriptor.v1
    else:
        raise ValueError('There is no appropriate version of BGSegmentation vector message.')

    images_photos = message_data.photos

    for photo in images_photos:
        image_time = photo.dateTaken
        scene_order = photo.sceneOrder
        image_as = photo.aspectRatio
        image_orderInScene = photo.orderInScene
        image_color = photo.colorEnum  # grayscale = 0
        image_orientation = 'landscape' if image_as <= 0.5 else 'portrait'
        background_centroid = photo.blobCentroid
        if photo.photoId in images_dict:
            images_dict[photo.photoId].update(
                {'image_time': image_time, "scene_order": scene_order, 'image_as': image_as,
                 'image_color': image_color, 'image_orientation': image_orientation,
                 'image_orderInScene': image_orderInScene, 'background_centroid':background_centroid, 'diameter':photo.blobDiameter })
        else:
            images_dict[photo.photoId] = {'image_time': image_time, "scene_order": scene_order,
                                          'image_as': image_as,
                                          'image_color': image_color, 'image_orientation': image_orientation,
                                          'image_orderInScene': image_orderInScene, "background_centroid":background_centroid,'diameter':photo.blobDiameter}

    return images_dict