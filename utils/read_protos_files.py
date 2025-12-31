import io
import os
import traceback
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd

from collections import Counter
from ptinfra.azure.pt_file import PTFile

from utils.reading_tools import generate_dict_key, check_gallery_type, process_content, _flatten
from utils.configs import CONFIGS
from utils.image_queries import generate_query
from utils.protos import FaceVector_pb2 as face_vector
from utils.protos import BGSegmentation_pb2 as meta_vector
from utils.protos import PersonInfo_pb2 as person_info
from utils.protos  import ContentCluster_pb2 as content_cluster
from utils.protos import PersonVector_pb2 as person_vector
from utils.protos import SocialCircle_pb2 as social_circle


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
        id_to_gender = {}
        person_rows = []

        # Extract persons information and prepare for DataFrame update
        for iden in identity_info:
            id = iden.identityNumeralId
            infos = iden.personInfo
            gender = infos.gender
            age = infos.age

            id_to_gender[id] = gender
            best_photo = infos.bestPhoto

            person_rows.append(
                {
                    "identity_id": iden.identityNumeralId,
                    "age": age,
                    "gender": gender,
                    "best_photo_id": best_photo,
                }
            )


            for im in infos.imagesInfo:
                photo_ids.append(im.photoId)
                persons_ids_list.append(id)

        if persons_ids_list:  # Check if anyone was identified at all
            id_counts = Counter(persons_ids_list)

            most_common_persons = id_counts.most_common(2)

            # Extract just the IDs from the tuples into a simple list
            top_person_ids = [person[0] for person in most_common_persons]
        else:
            top_person_ids = []

        # Create a temporary DataFrame with the new person information
        temp_df = pd.DataFrame({
            'image_id': photo_ids,
            'persons_ids': persons_ids_list,
        })

        if not temp_df.empty:
            persons_info_df = temp_df.groupby('image_id')['persons_ids'].apply(list).reset_index()
        else:
            persons_info_df = pd.DataFrame(columns=['image_id', 'persons_ids'])

        persons_info_df['main_persons'] = [top_person_ids for _ in range(len(persons_info_df))]
        persons_info_df['persons_ids'] = persons_info_df['persons_ids'].apply(lambda x: x if isinstance(x, list) else [])

        # add social dataframe to person dataframe
        person_df = pd.DataFrame(person_rows)

    except Exception as e:
        logger.error("Error reading persons info from file: {}".format(e))
        return None

    return persons_info_df,person_df


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
            photo_data.append({'image_id': image.photoId, 'number_bodies': len(image.bodies), "bodies_info":image.bodies})

        photo_df = pd.DataFrame(photo_data)
        # Fill missing 'number_bodies' with 0 if not provided in the photos data
        photo_df.fillna({'number_bodies': 0}, inplace=True)

    except Exception as ex:
        logger.error(f"Error reading person vectors from file: {ex}")
        return None

    return photo_df

def get_social_circle(social_circle_file, logger):
    try:
        social_info_bytes = PTFile(social_circle_file)  # Load file
        if not social_info_bytes.exists():
            return None
        social_info_bytes = social_info_bytes.read_blob()
        social_descriptor = social_circle.SocialCircleMessageWrapper()
        social_descriptor.ParseFromString(social_info_bytes)

        if social_descriptor.WhichOneof("versions") == 'v1':
            message_data = social_descriptor.v1
        else:
            logger.error('There is no appropriate version of Social Circle message.')
            raise ValueError('There is no appropriate version of  Social Circle message.')

        # --- 1. Access the v1 version ---
        data_v1 = message_data

        # --- 2. Extract social circles ---
        social_circles = data_v1.socialCircles

        circle_data = []

        for idx, circle in enumerate(social_circles):
            # identityNumeralId is a list<int32>
            ids = list(circle.identityNumeralId) if circle.identityNumeralId else []

            circle_data.append({
                "circle_index": idx,
                "identity_ids": ids,
                "num_ids": len(ids)
            })

        social_df = pd.DataFrame(circle_data)

        # Optional: if there are no circles, create empty DataFrame
        if social_df.empty:
            social_df = pd.DataFrame(columns=["circle_index", "identity_ids", "num_ids"])

    except Exception as ex:
        logger.error(f"Error reading social circle file from file: {ex}")
        return None

    return social_df

def get_info_protobufs(project_base_url, logger,clip_df=None):
    try:
        start = datetime.now()
        image_file = os.path.join(project_base_url, 'ai_search_matrix.pai')
        faces_file = os.path.join(project_base_url, 'ai_face_vectors.pb')
        persons_file = os.path.join(project_base_url, 'persons_info.pb')
        cluster_file = os.path.join(project_base_url, 'content_cluster.pb')
        segmentation_file = os.path.join(project_base_url, 'bg_segmentation.pb')
        person_vector_file = os.path.join(project_base_url, 'ai_person_vectors.pb')

        files = [image_file, faces_file, persons_file, cluster_file, segmentation_file, person_vector_file]

        # List of functions to run in parallel
        functions = [
            partial(get_faces_info, faces_file),
            partial(get_persons_ids, persons_file),
            partial(get_clusters_info, cluster_file),
            partial(get_photo_meta, segmentation_file),
            partial(get_person_vectors, person_vector_file),

        ]
        
        if clip_df is None:
            functions.insert(0, partial(get_image_embeddings, image_file))

        persons_details_df = None

        results = []
        for idx, func in enumerate(functions):
            result = func(logger)

            # handle special case where function returns (df, extra)
            if isinstance(result, tuple):
                df, extra = result
                result = df

                # assume this is get_persons_ids; store extra as persons_details_df
                persons_details_df = extra

            if result is None:
                return None, None, 'Error in reading data from protobuf file: {}'.format(files[idx])
            elif result.empty or result.shape[0] == 0:
                return None, None, 'There are no required data in protobuf file: {}'.format(files[idx])
            results.append(result)

        if clip_df is not None:
            results.insert(0, clip_df)

        gallery_info_df = results[0]
        for res in results[1:]:
            gallery_info_df = pd.merge(gallery_info_df, res, on="image_id", how="outer")

        # Convert only the specified columns to 'Int64' (nullable integer type)
        columns_to_convert = ["image_class", "cluster_label", "cluster_class", "image_order", "scene_order"]
        gallery_info_df[columns_to_convert] = gallery_info_df[columns_to_convert].astype('Int64')

        before = len(gallery_info_df)

        gallery_info_df = gallery_info_df.dropna(subset=['embedding'])

        gallery_info_df["persons_ids"] = gallery_info_df["persons_ids"].apply(
            lambda x: [] if not isinstance(x, list) else x
        )

        # gallery_info_df = gallery_info_df.dropna(subset=['persons_ids'])


        after = len(gallery_info_df)

        logger.warning(f"Dropped {before - after} rows because embedding is NaN")

        is_wedding = check_gallery_type(gallery_info_df)

        if is_wedding:
            # make Cluster column
            gallery_info_df = gallery_info_df.apply(process_content, axis=1)
            # gallery_info_df = gallery_info_df.merge(processed_df[['image_id', 'cluster_context']],
            #                                         how='left', on='image_id')
            bride_id, groom_id = np.nan, np.nan

            from collections import Counter
            bride_set = Counter(
                _flatten(gallery_info_df.loc[gallery_info_df["cluster_context"] == "bride", "persons_ids"]))
            groom_set = Counter(
                _flatten(gallery_info_df.loc[gallery_info_df["cluster_context"] == "groom", "persons_ids"]))

            main_row = gallery_info_df["main_persons"].dropna().iloc[0]

            # bride_id = bride_set.most_common(1)[0][0] if bride_set else np.nan
            # groom_id = groom_set.most_common(1)[0][0] if groom_set else np.nan

            if bride_set:
                bride_candidates = [id for id, count in bride_set.most_common() if
                                    count == bride_set.most_common(1)[0][1]]
                bride_id = next((id for id in bride_candidates if id in main_row),
                                bride_candidates[0]) if bride_candidates else np.nan
            else:
                bride_id = np.nan


            if groom_set:
                groom_candidates = [id for id, count in groom_set.most_common() if
                                    count == groom_set.most_common(1)[0][1]]
                groom_id = next((id for id in groom_candidates if id in main_row),
                                groom_candidates[0]) if groom_candidates else np.nan
            else:
                groom_id = np.nan

            if np.isnan(bride_id) and not np.isnan(groom_id):
                for person_id in main_row:
                    if person_id != groom_id:
                        bride_id = person_id
                        break
            elif np.isnan(groom_id) and not np.isnan(bride_id):
                for person_id in main_row:
                    if person_id != bride_id:
                        groom_id = person_id
                        break
            elif np.isnan(bride_id) and np.isnan(groom_id):
                if len(main_row) >= 2:
                    bride_id = main_row[0]
                    groom_id = main_row[1]

            if groom_id not in main_row or bride_id not in main_row:
                logger.warning(f"Main persons {main_row} do not contain bride {bride_id} or groom {groom_id}")

            gallery_info_df["bride_id"] = bride_id
            gallery_info_df["groom_id"] = groom_id

            gallery_info_df["main_persons"] = gallery_info_df["main_persons"].apply(
                lambda x: x if isinstance(x, (list, tuple)) else []
            )

            gallery_info_df["persons_ids"] = gallery_info_df["persons_ids"].apply(
                lambda x: x if isinstance(x, (list, tuple)) else []
            )

        # Get Query Content of each image
        if gallery_info_df is not None:
            model_version = gallery_info_df.iloc[0]['model_version']
            if model_version == 1:
                gallery_info_df = generate_query(CONFIGS["queries_file_v2"], gallery_info_df, num_workers=8)
            else:
                #gallery_info_df = generate_query(CONFIGS["queries_file_v2"], gallery_info_df, num_workers=8)
                gallery_info_df = generate_query(CONFIGS["queries_file_v3"], gallery_info_df, num_workers=8)

        logger.debug("Number of images before cleaning the nan values: {}".format(len(gallery_info_df.index)))
        columns_to_check = ["ranking", "image_order", "image_class", "cluster_label", "cluster_class"]
        gallery_info_df = gallery_info_df.dropna(subset=columns_to_check)
        logger.debug("Number of images after cleaning the nan values: {}".format(len(gallery_info_df.index)))
        # make sure it has list values not float nan

        # Cluster people by number of people inside the image
        gallery_info_df['people_cluster'] = gallery_info_df.apply(lambda row: generate_dict_key(row['persons_ids'], row['number_bodies']), axis=1)
        logger.debug("Time for reading files: {}".format(datetime.now() - start))

        # get social circle of the project
        social_circle_file = os.path.join(project_base_url, 'social_circles.pb')
        social_circle_df = get_social_circle(social_circle_file,logger)

        return gallery_info_df, is_wedding,social_circle_df,persons_details_df, None

    except Exception as ex:
        tb = traceback.extract_tb(ex.__traceback__)
        filename, lineno, func, text = tb[-1]
        return None, None,None,None, f'Error in reading protobufs: {ex}. Exception in function: {func}, line {lineno}, file {filename}.'
