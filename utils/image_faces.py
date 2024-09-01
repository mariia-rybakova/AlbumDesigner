from ptinfra.azure.pt_file import PTFile
from utils.protos import FaceVector_pb2 as face_vector


def get_faces_info(faces_file, images_dict,logger=None):
    try:
        faces_info_bytes = PTFile(faces_file)  # load file
        if not faces_info_bytes.exists():
            return None
        faces_info_bytes = faces_info_bytes.read_blob()
        face_descriptor = face_vector.FaceVectorMessageWrapper()
        face_descriptor.ParseFromString(faces_info_bytes)

    except Exception as e:
        # logger.warning('Cant load cluster data from server: {}. Loading from local directory.'.format(e))
        print('Cant load Face Vector data from server: {}. trying to load it from local directory.'.format(e))

        # load data locally
        try:
            face_descriptor = face_vector.FaceVectorMessageWrapper()
            with open(faces_file, 'rb') as f:
                face_descriptor.ParseFromString(f.read())
        except Exception as e:
            # logger.warning('Faces data could not be loaded local: {}'.format(e))
            print('Cant load Face data from local: {}.'.format(e))
            return None

    if face_descriptor.WhichOneof("versions") == 'v1':
        message_data = face_descriptor.v1
    else:
        raise ValueError('There is no appropriate version of face vector message.')

    images_photos = message_data.photos

    for photo in images_photos:
        number_faces = len(photo.faces)
        faces = photo.faces
        if photo.photoId in images_dict:
            images_dict[photo.photoId].update({'n_faces': number_faces, 'faces_info': faces})
    return images_dict
