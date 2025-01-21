import pandas as pd
from ptinfra.azure.pt_file import PTFile
from .protos import PersonVector_pb2 as person_vector

def get_person_vectors(persons_file, df, logger=None):
    try:
        person_info_bytes = PTFile(persons_file)  # Load file
        if not person_info_bytes.exists():
            return None
        person_info_bytes = person_info_bytes.read_blob()
        person_descriptor = person_vector.PersonVectorMessageWrapper()
        person_descriptor.ParseFromString(person_info_bytes)

    except Exception as e:
        logger.warning('Cannot load cluster data from server: {}. Loading from local directory.'.format(e))
        print('Cannot load cluster data from server: {}. Loading from local directory.'.format(e))
        return None

    if person_descriptor.WhichOneof("versions") == 'v1':
        message_data = person_descriptor.v1
    else:
        logger.error('There is no appropriate version of Person vector message.')
        raise ValueError('There is no appropriate version of Person vector message.')

    images = message_data.photos

    # Create a DataFrame for the photos
    photo_data = []
    for image in images:
        image_id = image.photoId
        number_bodies = len(image.bodies)
        photo_data.append({'image_id': image_id, 'number_bodies': number_bodies})

    photo_df = pd.DataFrame(photo_data)

    # Merge the new data with the existing DataFrame
    df = df.merge(photo_df, how='left', on='image_id')

    # Fill missing 'number_bodies' with 0 if not provided in the photos data
    df['number_bodies'].fillna(0, inplace=True)

    return df
