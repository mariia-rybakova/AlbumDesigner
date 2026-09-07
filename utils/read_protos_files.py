import os
import traceback
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd

from collections import Counter
from ptinfra.azure.pt_file import PTFile

# Protobuf schemas, the versioned-blob loader and the PAI codec all come from
# ptinfra rather than being vendored here. ptinfra.proto is the shared kernel:
# its pb/ modules are copied verbatim from pic-time/protobufs at a recorded
# commit (ptinfra.proto.PROTOBUFS_COMMIT), so every service parses the same
# schema. Note protobuf keys its global descriptor pool by .proto file name and
# rejects a second registration whose bytes differ, so a local copy of any of
# these cannot coexist with ptinfra's in one process.
from ptinfra.exporter.pai_writer import parse_pai
from ptinfra.proto.pb import BGSegmentation_pb2 as meta_vector
from ptinfra.proto.pb import ContentCluster_pb2 as content_cluster
from ptinfra.proto.pb import FaceVector_pb2 as face_vector
from ptinfra.proto.pb import PersonInfo_pb2 as person_info
from ptinfra.proto.pb import PersonVector_pb2 as person_vector
from ptinfra.proto.pb import SocialCircle_pb2 as social_circle
from ptinfra.read_stage import load_versioned

from utils.reading_tools import generate_dict_key, check_gallery_type, process_content, _flatten
from utils.configs import CONFIGS
from utils.image_queries import generate_query


def get_image_embeddings(file, logger):
    """Read the CLIP embedding matrix (`ai_search_matrix.pai`).

    The PAI v2/v3 binary format is decoded by ptinfra's codec, the same one the
    exporter writes with.
    """
    try:
        model_version, embeddings = parse_pai(PTFile(file).read_blob())
    except Exception as e:
        logger.error(f"Error reading image embeddings from file: {e}")
        return None

    df = pd.DataFrame([
        {"image_id": photo_id, "embedding": embedding}
        for photo_id, embedding in embeddings.items()
    ])

    df['model_version'] = model_version
    return df


def get_faces_info(faces_file, logger):
    try:
        message_data = load_versioned(faces_file, face_vector.FaceVectorMessageWrapper)

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
        message_data = load_versioned(file, meta_vector.PhotoBGSegmentationMessageWrapper)

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
        message_data = load_versioned(persons_file, person_info.PersonInfoMessageWrapper,
                                      missing_ok=True)
        if message_data is None:
            return None

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
        message_data = load_versioned(cluster_file, content_cluster.ContentClusterMessageWrapper,
                                      missing_ok=True)
        if message_data is None:
            return None

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
        message_data = load_versioned(persons_file, person_vector.PersonVectorMessageWrapper,
                                      missing_ok=True)
        if message_data is None:
            return None

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
        message_data = load_versioned(social_circle_file, social_circle.SocialCircleMessageWrapper,
                                      missing_ok=True)
        if message_data is None:
            return None

        social_circles = message_data.socialCircles

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

# ==========================================================================
# Gallery ingest / enrichment
# ==========================================================================
#
# `get_info_protobufs` used to do two different jobs in one pass: read the
# gallery protobufs, *and* derive classification, CLIP tagging and identities
# from what it read. The substage pipeline needs those separable
# (`src/pipeline/ingest` vs `src/pipeline/enrich`), so the work is split into
# the functions below. `get_info_protobufs` is kept as their composition, in
# the original order, for callers that still want it in one call.


def load_gallery_assets(project_base_url, logger, clip_df=None):
    """Pure ingest: decode the gallery protobufs into one photo table.

    Reads and organises only. Nothing here classifies a photo, projects an
    embedding against a query bank, or resolves an identity.

    Returns ``(gallery_info_df, persons_details_df, social_circle_df, error)``.
    """
    try:
        start = datetime.now()
        image_file = os.path.join(project_base_url, 'ai_search_matrix.pai')
        faces_file = os.path.join(project_base_url, 'ai_face_vectors.pb')
        persons_file = os.path.join(project_base_url, 'persons_info.pb')
        cluster_file = os.path.join(project_base_url, 'content_cluster.pb')
        segmentation_file = os.path.join(project_base_url, 'bg_segmentation.pb')
        person_vector_file = os.path.join(project_base_url, 'ai_person_vectors.pb')

        files = [image_file, faces_file, persons_file, cluster_file, segmentation_file, person_vector_file]

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
                return None, None, None, 'Error in reading data from protobuf file: {}'.format(files[idx])
            elif result.empty or result.shape[0] == 0:
                return None, None, None, 'There are no required data in protobuf file: {}'.format(files[idx])
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

        after = len(gallery_info_df)

        logger.warning(f"Dropped {before - after} rows because embedding is NaN")

        # get social circle of the project
        social_circle_file = os.path.join(project_base_url, 'social_circles.pb')
        social_circle_df = get_social_circle(social_circle_file, logger)

        logger.debug("Time for reading files: {}".format(datetime.now() - start))

    except Exception as ex:
        tb = traceback.extract_tb(ex.__traceback__)
        filename, lineno, func, text = tb[-1]
        return None, None, None, f'Error in reading protobufs: {ex}. Exception in function: {func}, line {lineno}, file {filename}.'

    return gallery_info_df, persons_details_df, social_circle_df, None


def classify_gallery_type(gallery_info_df):
    """Wedding vs non-wedding, from the share of unclassified `image_class`."""
    return check_gallery_type(gallery_info_df)


def add_content_class(gallery_info_df):
    """Derive `cluster_context` (the category name) from `cluster_class`."""
    return gallery_info_df.apply(process_content, axis=1)


def _other_partner(solo_counts, named_id):
    """The second partner, read off the solo context that named the first.

    A same-sex couple puts both partners into one solo context and leaves the
    other empty, so the usual path never names the second one: on gallery
    52894932 the `bride` context held identities 5 and 1 exactly **32 times
    each** while the `groom` context was empty. Whoever else that one context is
    full of is the other partner.

    This runs only when `main_persons` did not answer, which is the case it
    exists for -- that list is the model's most-frequent identities and can be
    empty, and when it is, the second partner used to come out as NaN. A NaN
    partner is not a loud failure either: the category filter
    ``persons_ids == [nan]`` simply matches nothing, so one partner would
    quietly vanish from the album.
    """
    for identity, _count in solo_counts.most_common():
        if identity != named_id:
            return identity
    return np.nan


def resolve_bride_groom(gallery_info_df, logger):
    """Resolve the couple identity ids and stamp them on every row.

    Picks the most frequent identity inside the `bride` / `groom` contexts,
    preferring a candidate that also appears in `main_persons`, with fallbacks
    for when one or both sides are missing.
    """
    bride_id, groom_id = np.nan, np.nan

    bride_set = Counter(
        _flatten(gallery_info_df.loc[gallery_info_df["cluster_context"] == "bride", "persons_ids"]))
    groom_set = Counter(
        _flatten(gallery_info_df.loc[gallery_info_df["cluster_context"] == "groom", "persons_ids"]))

    main_row = gallery_info_df["main_persons"].dropna().iloc[0]

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
        if np.isnan(bride_id):
            bride_id = _other_partner(groom_set, groom_id)
    elif np.isnan(groom_id) and not np.isnan(bride_id):
        for person_id in main_row:
            if person_id != bride_id:
                groom_id = person_id
                break
        if np.isnan(groom_id):
            groom_id = _other_partner(bride_set, bride_id)
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

    return gallery_info_df


def add_semantic_tags(gallery_info_df, logger=None):
    """Project each embedding against the query bank for its model version.

    Produces `image_query_content` / `image_subquery_content`. This is a
    search, not a read, which is why it belongs in enrichment rather than
    ingest.
    """
    if gallery_info_df is None:
        return gallery_info_df

    model_version = gallery_info_df.iloc[0]['model_version']
    if model_version == 1:
        return generate_query(CONFIGS["queries_file_v2"], gallery_info_df, num_workers=8)
    return generate_query(CONFIGS["queries_file_v3"], gallery_info_df, num_workers=8)


def require_cluster_data(gallery_info_df, logger):
    """Drop rows missing the cluster columns everything downstream assumes."""
    logger.debug("Number of images before cleaning the nan values: {}".format(len(gallery_info_df.index)))
    columns_to_check = ["ranking", "image_order", "image_class", "cluster_label", "cluster_class"]
    gallery_info_df = gallery_info_df.dropna(subset=columns_to_check)
    logger.debug("Number of images after cleaning the nan values: {}".format(len(gallery_info_df.index)))
    return gallery_info_df


def add_people_cluster(gallery_info_df):
    """Key each photo by who is in it, for people-based grouping."""
    gallery_info_df['people_cluster'] = gallery_info_df.apply(
        lambda row: generate_dict_key(row['persons_ids'], row['number_bodies']), axis=1)
    return gallery_info_df


def get_info_protobufs(project_base_url, logger, clip_df=None):
    """Read + enrich in one call, in the original order.

    Kept for callers outside the substage pipeline. The pipeline runs the
    pieces individually so each one can be replaced on its own.
    """
    try:
        gallery_info_df, persons_details_df, social_circle_df, error = load_gallery_assets(
            project_base_url, logger, clip_df=clip_df)
        if error is not None:
            return None, None, None, None, error

        is_wedding = classify_gallery_type(gallery_info_df)

        if is_wedding:
            gallery_info_df = add_content_class(gallery_info_df)
            gallery_info_df = resolve_bride_groom(gallery_info_df, logger)

        gallery_info_df = add_semantic_tags(gallery_info_df, logger)
        gallery_info_df = require_cluster_data(gallery_info_df, logger)
        gallery_info_df = add_people_cluster(gallery_info_df)

        return gallery_info_df, is_wedding, social_circle_df, persons_details_df, None

    except Exception as ex:
        tb = traceback.extract_tb(ex.__traceback__)
        filename, lineno, func, text = tb[-1]
        return None, None, None, None, f'Error in reading protobufs: {ex}. Exception in function: {func}, line {lineno}, file {filename}.'
