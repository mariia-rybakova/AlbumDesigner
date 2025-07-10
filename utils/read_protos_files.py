import io
import numpy as np
import pandas as pd
from ptinfra.azure.pt_file import PTFile
from utils.protos import FaceVector_pb2 as face_vector
from utils.protos import BGSegmentation_pb2 as meta_vector
from utils.protos import PersonInfo_pb2 as person_info
from utils.protos  import ContentCluster_pb2 as content_cluster
from utils.protos import PersonVector_pb2 as person_vector


def get_image_embeddings(file, logger):
    embed = {}

    try:
        fb = PTFile(file)
        fileBytes = fb.read_blob()
        fileBytes = io.BytesIO(fileBytes)

        header_b = fileBytes.read1(4)
        header = header_b.decode('utf-8')
        if header == 'pai3':
            model_version_b = fileBytes.read(4)
            model_version = int.from_bytes(model_version_b, 'little')
        elif header == 'pai2':
            model_version = 1
        else:
            raise Exception('Unexpected header {}, clip_file {}'.format(header, file))

        num_images_b = fileBytes.read1(4)
        num_images = int.from_bytes(num_images_b, 'little')

        for _ in range(num_images):
            photo_id_b = fileBytes.read1(8)
            photo_id = int.from_bytes(photo_id_b, 'little')

            emb_size_b = fileBytes.read1(4)
            emb_size = int.from_bytes(emb_size_b, 'little')
            embedding_b = fileBytes.read1(4 * emb_size)
            embedding = np.frombuffer(embedding_b, dtype='float32').reshape((emb_size,))

            embed[photo_id] = {'embedding': embedding}

    except Exception as e:
        logger.error(f"Error reading image embeddings from file: {e}")
        return None

    df = pd.DataFrame([
        {"image_id": photo_id, "embedding": data["embedding"]}
        for photo_id, data in embed.items()
    ])

    df['model_version'] = model_version
    return df


def get_faces_info(faces_file, logger):
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
            number_faces = len(photo.faces)
            faces = list(photo.faces)
            photo_ids.append(photo.photoId)
            num_faces_list.append(number_faces)
            faces_info_list.append(faces)

        face_info_df = pd.DataFrame({
            'image_id': photo_ids,
            'n_faces': num_faces_list,
            'faces_info': faces_info_list
        })

    except Exception as ex:
        logger.error(f"Error reading face info from file: {ex}")
        return None

    return face_info_df


def get_photo_meta(file, logger):
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

        # Add safer handling of photo attributes
        for photo in images_photos:
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

    return additional_image_info_df


def get_persons_ids(persons_file, logger):
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
                photo_ids.append(im.photoId)
                persons_ids_list.append(id)


        # Create a temporary DataFrame with the new person information
        persons_info_df = pd.DataFrame({
            'image_id': photo_ids,
            'persons_ids': persons_ids_list
        })

        # Aggregate persons_ids for each image_id
        persons_info_df = persons_info_df.groupby('image_id')['persons_ids'].apply(list).reset_index()

    except Exception as e:
        logger.error("Error reading persons info from file: {}".format(e))
        return None

    return persons_info_df


def get_clusters_info(cluster_file, logger):
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

    except Exception as e:
        logger.error("Error reading cluster info from file: {}".format(e))
        return None

    return new_image_info_df


def get_person_vectors(persons_file, logger):
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
            photo_data.append({'image_id': image.photoId, 'number_bodies': len(image.bodies)})

        photo_df = pd.DataFrame(photo_data)
        # Fill missing 'number_bodies' with 0 if not provided in the photos data
        photo_df['number_bodies'].fillna(0, inplace=True)

    except Exception as ex:
        logger.error(f"Error reading person vectors from file: {ex}")
        return None

    return photo_df