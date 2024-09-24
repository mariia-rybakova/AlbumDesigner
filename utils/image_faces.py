from ptinfra.azure.pt_file import PTFile
from utils.protos import FaceVector_pb2 as face_vector


def get_faces_info(faces_file, images_dict,logger=None):
    try:
        faces_info_bytes = PTFile(faces_file)  # load file
        if not faces_info_bytes.exists():
            if logger is not None:
                logger.error("The faces file does not exist in the server")
            return None
        faces_info_bytes = faces_info_bytes.read_blob()
        face_descriptor = face_vector.FaceVectorMessageWrapper()
        face_descriptor.ParseFromString(faces_info_bytes)

    except Exception as e:
        logger.warning('Cant load cluster data from server: {}. Loading from local directory.'.format(e))
        print('Cant load Face Vector data from server: {}. trying to load it from local directory.'.format(e))

    if face_descriptor.WhichOneof("versions") == 'v1':
        message_data = face_descriptor.v1
    else:
        if logger is not None:
            logger.error("'There is no appropriate version of face vector message.'")
        raise ValueError('There is no appropriate version of face vector message.')

    images_photos = message_data.photos

    for photo in images_photos:
        number_faces = len(photo.faces)
        faces = list(photo.faces)
        if photo.photoId in images_dict:
            images_dict[photo.photoId].update({'n_faces': number_faces, 'faces_info': faces})

    return images_dict,None

def load_face_data_from_proto(file, emb_size=512):
    # loading from server
    face_data_bytes = PTFile(file)  # load file
    face_data_bytes = face_data_bytes.read_blob()

    face_descriptor = face_vector.FaceVectorMessageWrapper()
    face_descriptor.ParseFromString(face_data_bytes)

    # face_descriptor = face_vector.FaceVectorMessageWrapper()
    # with open(file, 'rb') as f:
    #     face_descriptor.ParseFromString(f.read())

    if face_descriptor.WhichOneof("versions") == 'v1':
        message_data = face_descriptor.v1
    else:
        raise ValueError('There is no appropriate version of face vector message.')

    photo_id2faces_list = dict()
    face_data = message_data.photos

    for photo_info in face_data:

        faces = list()
        for face_info in photo_info.faces:
            faces.append((face_info.id, [face_info.bbox.x1, face_info.bbox.y1,
                                         face_info.bbox.x2, face_info.bbox.y2]))
        photo_id2faces_list[photo_info.photoId] = faces
    return photo_id2faces_list, message_data

# sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")
#
# load_face_data_from_proto('ptstorage_32://pictures/40/607/40607142/tydzj68uum3cpmy9mb/ai_face_vectors.pb')