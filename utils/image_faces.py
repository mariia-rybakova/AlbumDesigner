import pandas as pd
from ptinfra.azure.pt_file import PTFile
from utils.protos import FaceVector_pb2 as face_vector


def get_faces_info(faces_file, df,logger=None):
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

    # Prepare lists for new DataFrame columns
    photo_ids = []
    num_faces_list = []
    faces_info_list = []

    # Extract faces information and prepare for DataFrame update
    for photo in images_photos:
        number_faces = len(photo.faces)
        faces = list(photo.faces)
        photo_ids.append(photo.photoId)
        num_faces_list.append(number_faces)
        faces_info_list.append(faces)

    # Create a temporary DataFrame with the new face information
    face_info_df = pd.DataFrame({
        'photo_id': photo_ids,
        'n_faces': num_faces_list,
        'faces_info': faces_info_list
    })

    # Merge the original DataFrame with the new face information DataFrame
    df = df.merge(face_info_df, how='left', left_on='image_id', right_on='photo_id')

    # Drop the redundant photo_id column if necessary
    df.drop(columns=['photo_id'], inplace=True)

    return df
