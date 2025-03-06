import pandas as pd
from ptinfra.azure.pt_file import PTFile
from utils.protos import PersonInfo_pb2 as person_vector

def get_persons_ids(persons_file, df,logger=None):
    required_ids = set(df['image_id'].tolist())

    try:
        person_info_bytes = PTFile(persons_file)  # load file
        if not person_info_bytes.exists():
            return None
        person_info_bytes = person_info_bytes.read_blob()
        person_descriptor = person_vector.PersonInfoMessageWrapper()
        person_descriptor.ParseFromString(person_info_bytes)

    except Exception as e:
        logger.warning('Cant load cluster data from server: {}. Loading from local directory.'.format(e))
        return None

    if person_descriptor.WhichOneof("versions") == 'v1':
        message_data = person_descriptor.v1
    else:
        logger.error('There is no appropriate version of Person vector message.')
        raise ValueError('There is no appropriate version of Person vector message.')

    identity_info = message_data.identities

    # Prepare lists for DataFrame columns
    photo_ids = []
    persons_ids_list = []

    # Extract persons information and prepare for DataFrame update
    for iden in identity_info:
        id = iden.identityNumeralId
        infos = iden.personInfo
        for im in infos.imagesInfo:
            if im.photoId in required_ids:
                photo_ids.append(im.photoId)
                persons_ids_list.append(id)


    # Create a temporary DataFrame with the new person information
    persons_info_df = pd.DataFrame({
        'image_id': photo_ids,
        'persons_ids': persons_ids_list
    })

    # Aggregate persons_ids for each image_id
    persons_info_df = persons_info_df.groupby('image_id')['persons_ids'].apply(list).reset_index()

    # Merge the original DataFrame with the new person information DataFrame
    df = df.merge(persons_info_df, how='inner', on='image_id')


    df['persons_ids'].fillna(0, inplace=True)

    return df

