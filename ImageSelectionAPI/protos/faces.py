import pandas as pd
from ptinfra.azure.pt_file import PTFile
from files import FaceVector_pb2 as face_vector


def get_faces_info(faces_file, df, logger=None):
    required_ids = set(df['image_id'].tolist())

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
            logger.error("There is no appropriate version of face vector message.")
        raise ValueError("There is no appropriate version of face vector message.")

    images_photos = message_data.photos

    photo_ids = []
    num_faces_list = []


    for photo in images_photos:
        if photo.photoId in required_ids:
            number_faces = len(photo.faces)
            photo_ids.append(photo.photoId)
            num_faces_list.append(number_faces)


    face_info_df = pd.DataFrame({
        'photo_id': photo_ids,
        'n_faces': num_faces_list,
    })

    df = df.merge(face_info_df, how='left', left_on='image_id', right_on='photo_id')
    df.drop(columns=['photo_id'], inplace=True)

    return df
