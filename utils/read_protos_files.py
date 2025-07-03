import io
import numpy as np
import pandas as pd
from ptinfra.azure.pt_file import PTFile
from utils.protos import FaceVector_pb2 as face_vector
from utils.protos import BGSegmentation_pb2 as meta_vector
from utils.protos import PersonInfo_pb2 as person_info
from utils.protos  import ContentCluster_pb2 as content_cluster
from utils.protos import PersonVector_pb2 as person_vector


def get_image_embeddings(file, df, logger):
    embed = {}
    required_ids = set(df['image_id'].tolist())

    try:
        fb = PTFile(file)
        fileBytes = fb.read_blob()
        fileBytes = io.BytesIO(fileBytes)

        header_b = fileBytes.read1(4)
        header = header_b.decode('utf-8')
        if header == 'pai3':
            model_version_b = fileBytes.read(4)
            model_version = int.from_bytes(model_version_b, 'little')
        else:
            model_version = 1
        num_images_b = fileBytes.read1(4)
        num_images = int.from_bytes(num_images_b, 'little')

        for _ in range(num_images):
            photo_id_b = fileBytes.read1(8)
            photo_id = int.from_bytes(photo_id_b, 'little')

            emb_size_b = fileBytes.read1(4)
            emb_size = int.from_bytes(emb_size_b, 'little')
            embedding_b = fileBytes.read1(4 * emb_size)
            embedding = np.frombuffer(embedding_b, dtype='float32').reshape((emb_size,))

            if photo_id in required_ids:
                embed[photo_id] = {'embedding': embedding}

    except Exception as e:
        logger.error(f"Error reading image embeddings from file: {e}")
        return None

    df['embedding'] = df['image_id'].map(lambda x: embed.get(x, {}).get('embedding', np.nan))
    df['model_version'] = model_version
    return df


def get_faces_info(faces_file, df, logger):
    required_ids = set(df['image_id'].tolist())

    try:
        faces_info_bytes = PTFile(faces_file)  # load file
        faces_info_bytes = faces_info_bytes.read_blob()
        face_descriptor = face_vector.FaceVectorMessageWrapper()
        face_descriptor.ParseFromString(faces_info_bytes)

        if face_descriptor.WhichOneof("versions") == 'v1':
            message_data = face_descriptor.v1
        else:
            logger.error("There is no appropriate version of face vector message.")
            return None

        images_photos = message_data.photos

        photo_ids = []
        num_faces_list = []
        faces_info_list = []

        for photo in images_photos:
            if photo.photoId in required_ids:
                number_faces = len(photo.faces)
                faces = list(photo.faces)
                photo_ids.append(photo.photoId)
                num_faces_list.append(number_faces)
                faces_info_list.append(faces)

        face_info_df = pd.DataFrame({
            'photo_id': photo_ids,
            'n_faces': num_faces_list,
            'faces_info': faces_info_list
        })
    except Exception as ex:
        logger.error(f"Error reading face info from file: {ex}")
        return None

    df = df.merge(face_info_df, how='inner', left_on='image_id', right_on='photo_id')
    df.drop(columns=['photo_id'], inplace=True)

    return df


def get_photo_meta(file, df, logger):
    required_ids = set(df['image_id'].tolist())
    try:
        meta_info_bytes = PTFile(file)  # load file
        meta_info_bytes_info_bytes = meta_info_bytes.read_blob()
        meta_descriptor = meta_vector.PhotoBGSegmentationMessageWrapper()
        meta_descriptor.ParseFromString(meta_info_bytes_info_bytes)

        if meta_descriptor.WhichOneof("versions") == 'v1':
            message_data = meta_descriptor.v1
        else:
            logger.warning('There is no appropriate version of image meta message.')
            return None

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

        for photo in images_photos:
            if photo.photoId in required_ids:
                photo_ids.append(photo.photoId)
                image_times.append(photo.dateTaken)
                scene_orders.append(photo.sceneOrder)
                image_aspects.append(photo.aspectRatio)
                image_colors.append(photo.colorEnum)
                image_orientations.append('landscape' if photo.aspectRatio >= 1 else 'portrait')
                image_orderInScenes.append(photo.orderInScene)

                # Safer handling of optional fields
                background_centroids.append(getattr(photo, 'blobCentroid', None))
                blob_diameters.append(getattr(photo, 'blobDiameter', None))

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
    except Exception as ex:
        logger.error(f"Error reading photo meta info from file: {ex}")
        return None

    df = df.merge(additional_image_info_df, how='inner', on='image_id')

    return df


def get_persons_ids(persons_file, df,logger):
    required_ids = set(df['image_id'].tolist())

    try:
        person_info_bytes = PTFile(persons_file)  # load file
        if not person_info_bytes.exists():
            return None
        person_info_bytes = person_info_bytes.read_blob()
        person_descriptor = person_info.PersonInfoMessageWrapper()
        person_descriptor.ParseFromString(person_info_bytes)

        if person_descriptor.WhichOneof("versions") == 'v1':
            message_data = person_descriptor.v1
        else:
            logger.error('There is no appropriate version of Person vector message.')
            return None

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

        if not photo_ids and not persons_ids_list:
            df['persons_ids'] = [[] for _ in range(len(df))]
            return df

        # Aggregate persons_ids for each image_id
        persons_info_df = persons_info_df.groupby('image_id')['persons_ids'].apply(list).reset_index()
    except Exception as ex:
        logger.error(f"Error reading persons info from file: {ex}")
        return None

    # Merge the original DataFrame with the new person information DataFrame
    df = df.merge(persons_info_df, how='inner', on='image_id')
    return df


def get_clusters_info(cluster_file, df,logger):
    required_ids = set(df['image_id'].tolist())

    try:
        cluster_info_bytes = PTFile(cluster_file)  # load file
        if not cluster_info_bytes.exists():
            return None
        cluster_info_bytes = cluster_info_bytes.read_blob()
        cluster_descriptor = content_cluster.ContentClusterMessageWrapper()
        cluster_descriptor.ParseFromString(cluster_info_bytes)

        if cluster_descriptor.WhichOneof("versions") == 'v1':
            message_data = cluster_descriptor.v1
        else:
            logger.error('There is no appropriate version of cluster vector message.')
            raise ValueError('There is no appropriate version of cluster vector message.')

        images_photos = message_data.photos

        # Prepare lists to collect data
        photo_ids = []
        image_classes = []
        cluster_labels = []
        cluster_classes = []
        image_rankings = []
        image_orders = []

        # Loop through each photo and collect the required information
        for photo in images_photos:
            if photo.photoId in required_ids:
                photo_ids.append(photo.photoId)
                image_classes.append(int(photo.imageClass))
                cluster_labels.append(int(photo.clusterId))
                cluster_classes.append(int(photo.clusterClass))
                image_rankings.append(photo.selectionScore)
                image_orders.append(int(photo.selectionOrder))

        # Create a DataFrame from the collected data
        new_image_info_df = pd.DataFrame({
            'image_id': photo_ids,
            'image_class': image_classes,
            'cluster_label': cluster_labels,
            'cluster_class': cluster_classes,
            'ranking': image_rankings,
            'image_order': image_orders
        })
    except Exception as ex:
        logger.error(f"Error reading cluster info from file: {ex}")
        return None

    # Merge the original DataFrame with the new information
    df = df.merge(new_image_info_df, how='inner', on='image_id')

    return df


def get_person_vectors(persons_file, df, logger):
    required_ids = set(df['image_id'].tolist())
    try:
        person_info_bytes = PTFile(persons_file)  # Load file
        if not person_info_bytes.exists():
            return None
        person_info_bytes = person_info_bytes.read_blob()
        person_descriptor = person_vector.PersonVectorMessageWrapper()
        person_descriptor.ParseFromString(person_info_bytes)

        if person_descriptor.WhichOneof("versions") == 'v1':
            message_data = person_descriptor.v1
        else:
            logger.error('There is no appropriate version of Person vector message.')
            raise ValueError('There is no appropriate version of Person vector message.')

        images = message_data.photos

        # Create a DataFrame for the photos
        photo_data = []
        for image in images:
            if image.photoId in required_ids:
                photo_data.append({'image_id': image.photoId, 'number_bodies': len(image.bodies)})

        photo_df = pd.DataFrame(photo_data)
    except Exception as ex:
        logger.error(f"Error reading person vectors from file: {ex}")
        return None

    # Merge the new data with the existing DataFrame
    df = df.merge(photo_df, how='inner', on='image_id')
    df['number_bodies'].fillna(0, inplace=True)

    return df