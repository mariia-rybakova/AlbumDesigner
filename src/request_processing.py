import pandas as pd
import numpy as np
import traceback

from datetime import datetime

from pycparser.c_ast import Continue

from utils.configs import CONFIGS
from utils.layouts_tools import generate_layouts_df, get_layouts_data
from utils.read_protos_files import get_info_protobufs
from utils.time_processing import process_gallery_time
from ptinfra.azure.pt_file import PTFile
from ptinfra.utils.gallery import Gallery
import json
from bson.objectid import ObjectId
from qdrant_client import QdrantClient, models
from pymongo import MongoClient
from experiments.plotting import plot_selected_rows_to_pdf
from collections import Counter

def read_layouts_data(message, json_content):
    if 'designInfo' in json_content and json_content['designInfo'] is None:
        if 'designInfoTempLocation' in json_content:
            try:
                fb = PTFile(json_content['designInfoTempLocation'])
                fileBytes = fb.read_blob()
                designInfo = json.loads(fileBytes.decode('utf-8'))
                # logger.info('Read designInfo from blob location: {}'.format(designInfo))
                json_content['designInfo'] = designInfo
                message.content['designInfo'] = designInfo
                album_ar = {'anyPage': 2}
                for part in ['firstPage', 'lastPage', 'anyPage']:
                    if part in designInfo['parts']:
                        album_ar[part] = designInfo['parts'][part]['varient']['productWidth'] / \
                                         designInfo['parts'][part]['varient']['productHeight']
                message.content['album_ar'] = album_ar
            except Exception as e:
                return None, 'Error reading designInfo from blob location {}, error: {}'.format(
                    json_content['designInfoTempLocation'], e)
        else:
            return None, 'Incorrect message structure: {}. Skipping.'.format(json_content)

    message.pagesInfo = dict()
    message.designsInfo = dict()
    message.designsInfo['defaultPackageStyleId'] = json_content['designInfo']['defaultPackageStyleId']

    if 'anyPage' in json_content['designInfo']['parts'] and len(
            json_content['designInfo']['parts']['anyPage']['designIds']) > 0:
        message.designsInfo['anyPageIds'] = json_content['designInfo']['parts']['anyPage']['designIds']
    else:
        message.error = 'no anyPage in designInfo. Skipping.'
        return None, 'No anyPage in designInfo. Skipping message..'

    first_page_layouts_df = None
    last_page_layouts_df = None
    if 'firstPage' in json_content['designInfo']['parts']:
        if len(json_content['designInfo']['parts']['firstPage']['designIds']) > 0:
            message.designsInfo['firstPageDesignIds'] = json_content['designInfo']['parts']['firstPage']['designIds']
            message.pagesInfo['firstPage'] = True
            first_page_layouts_df = generate_layouts_df(json_content['designInfo']['designs'],
                                                       message.designsInfo['firstPageDesignIds'],
                                                       album_ar=message.content.get('album_ar', {'anyPage': 2})['anyPage'])
            message.designsInfo['firstPage_layouts_df'] = first_page_layouts_df

    if 'lastPage' in json_content['designInfo']['parts']:
        if len(json_content['designInfo']['parts']['lastPage']['designIds']) > 0:
            message.designsInfo['lastPageDesignIds'] = json_content['designInfo']['parts']['lastPage']['designIds']
            message.pagesInfo['lastPage'] = True
            last_page_layouts_df = generate_layouts_df(json_content['designInfo']['designs'],
                                                      message.designsInfo['lastPageDesignIds'],
                                                      album_ar=message.content.get('album_ar', {'anyPage': 2})['anyPage'])
            message.designsInfo['lastPage_layouts_df'] = last_page_layouts_df

    if 'cover' in json_content['designInfo']['parts']:
        if len(json_content['designInfo']['parts']['cover']['designIds']) > 0:
            message.designsInfo['coverDesignIds'] = json_content['designInfo']['parts']['cover']['designIds']
            message.pagesInfo['cover'] = True

            # coverPage_layouts_df = generate_layouts_df(json_content['designInfo']['designs'], _msg.designsInfo['coverDesignIds'])
            # _msg.designsInfo['coverPage_layouts_df'] = coverPage_layouts_df

    message.designsInfo['minPages'] = json_content['designInfo']['minPages'] if 'minPages' in json_content[
        'designInfo'] else 1
    message.designsInfo['maxPages'] = json_content['designInfo']['minPages'] if 'maxPages' in json_content[
        'designInfo'] else CONFIGS['max_total_spreads']

    any_page_layouts_df = generate_layouts_df(json_content['designInfo']['designs'], message.designsInfo['anyPageIds'],
                                             album_ar=message.content.get('album_ar', {'anyPage': 2})['anyPage'],
                                             do_mirror=True)

    if not any_page_layouts_df.empty:
        message.designsInfo['anyPagelayouts_df'] = any_page_layouts_df
        layout_id2data, box_id2data = get_layouts_data(any_page_layouts_df, first_page_layouts_df, last_page_layouts_df)
        message.designsInfo['anyPagelayout_id2data'] = layout_id2data
        message.designsInfo['anyPagebox_id2data'] = box_id2data

    return message


def add_scenes_info(gallery_info_df, project_base_url, logger):
    try:
        photos_metadata = Gallery(project_base_url)

        image_id2scene_image_order = dict()
        image_iter = 0
        for scene_idx, scene in enumerate(photos_metadata.scenes):
            for photo in scene.photos:
                try:
                    filename = photo.get_filename()
                    image_id2scene_image_order[np.int64(filename.split('.')[0])] = (scene_idx, image_iter)
                    image_iter += 1
                except Exception as e:
                    pass
    except Exception as e:
        logger.error(f"Error reading scenes info from gallery: {e}")
        return gallery_info_df

    mapped = gallery_info_df['image_id'].map(image_id2scene_image_order)
    mapped_df = pd.DataFrame(mapped.tolist(), columns=['scene_order', 'image_order'])
    for col in ['scene_order', 'image_order']:
        gallery_info_df[col] = mapped_df[col].combine_first(gallery_info_df[col])

    gallery_info_df[['scene_order', 'image_order']] = gallery_info_df[['scene_order', 'image_order']].astype('Int64')

    return gallery_info_df



def is_ceremony_gallery_sat(df: pd.DataFrame, logger=None):
    """
    SAT = gallery eligible to run the kiss-detection logic.

    Rules:
      1) Ceremony start/end must be valid and in logical order (end >= start).
      2) Ceremony start and end occur on the same calendar day.
      3) All ceremony images occur on the same calendar day (not just min/max).

    Returns:
      (eligible: bool, reason: str)
    """
    if df.empty:
        return False, "Empty dataframe"

    # normalize/parse
    tmp = df.copy()
    if "image_time_date" not in tmp.columns or "cluster_context" not in tmp.columns:
        return False, "Missing required columns"

    tmp["ts"] = pd.to_datetime(tmp["image_time_date"], errors="coerce")
    ceremony_df = tmp[tmp["cluster_context"] == "ceremony"]

    if ceremony_df.empty:
        return False, "No ceremony images"

    ceremony_start = ceremony_df["ts"].min()
    ceremony_end = ceremony_df["ts"].max()

    if pd.isna(ceremony_start) or pd.isna(ceremony_end):
        return False, "Ceremony timestamps contain NaT"

    if ceremony_end < ceremony_start:
        return False, "Ceremony end precedes start"

    # same-day check for start/end
    if  pd.Series([ceremony_start, ceremony_end]).dt.date.nunique() != 1:
        return False, "Ceremony spans multiple days (min/max day mismatch)"

    # all ceremony images on same day
    if ceremony_df["ts"].nunique() == 1:
        return False, "Not all ceremony images share the same day"

    if logger:
        logger.info(
            "Ceremony SAT: start=%s end=%s count=%d",
            ceremony_start, ceremony_end, len(ceremony_df)
        )
    return True, "OK"

def identify_kiss_ceremony(df, logger=None):
    eligible, reason = is_ceremony_gallery_sat(df, logger=logger)
    if not eligible:
        if logger:
            logger.warning("Gallery not SAT: %s", reason)

        return df

    # allowed values
    cluster_context_allowed = [None, "kiss", "ceremony"]
    query_allowed = ["kiss", "ceremony"]
    subquery_allowed = [
        "bride and groom kissing romantically",
        "wedding kiss at ceremony",
        "officiant leading wedding ceremony",
    ]

    # --- normalize strings ---
    df = df.copy()
    df["ts"] = pd.to_datetime(df["image_time_date"], errors="coerce")
    df["cluster_context_norm"] = df["cluster_context"].astype(str).str.lower().replace("none", None)
    df["query_norm"] = df["image_query_content"].astype(str).str.lower().str.strip()
    df["subquery_norm"] = df["image_subquery_content"].astype(str).str.lower().str.strip()

    # --- filter dataframe using list membership like your mask style ---
    df_filtered = df[
        df["cluster_context"].apply(lambda x: x in cluster_context_allowed)
        & df["image_query_content"].apply(lambda x: x in query_allowed)
        & df["image_subquery_content"].apply(lambda x: x in subquery_allowed)
        ]

    # --- 1️⃣ get the ceremony group and ceremony time window ---
    ceremony_df = df[df["cluster_context"] == "ceremony"]
    ceremony_start = ceremony_df["ts"].min()
    ceremony_end = ceremony_df["ts"].max()

    if pd.isna(ceremony_start) or pd.isna(ceremony_end):
        selected_rows = df.iloc[0:0]  # empty
    else:
        # --- 2️⃣ find the image(s) where subquery == "officiant leading wedding ceremony" ---
        officiant_df = df[df["image_subquery_content"] == "officiant leading wedding ceremony"]
        officiant_df = officiant_df[officiant_df["ts"].between(ceremony_start, ceremony_end)]

        if officiant_df.empty:
            selected_rows = df.iloc[0:0]
        else:
            # --- 3️⃣ find images containing "kiss" (from your allowed lists) ---
            kiss_df = df_filtered[
                df["image_subquery_content"].apply(
                    lambda x: "kiss" in str(x).lower()
                )
                | df["image_query_content"].apply(
                    lambda x: "kiss" in str(x).lower()
                )
                ]

            last_officiant_ts = officiant_df["ts"].max()

            if pd.isna(last_officiant_ts):
                kiss_near_officiant_df = kiss_df.iloc[0:0]  # no anchors -> empty
            else:
                start = last_officiant_ts - pd.Timedelta(minutes=6)
                end = last_officiant_ts + pd.Timedelta(minutes=6)

                # If you also want to keep it inside the ceremony window, add the second condition
                kiss_near_officiant_df = kiss_df[
                    kiss_df["ts"].between(start, end, inclusive="both")
                    & kiss_df["ts"].between(ceremony_start, ceremony_end, inclusive="both")
                    ]

            # --- 5️⃣ mark and save ---
            df.loc[kiss_near_officiant_df.index, "cluster_context"] = "may kiss bride"
            selected_rows = kiss_near_officiant_df

    # plot_selected_rows_to_pdf(selected_rows)
    # print("plotting done")

    return df



def fetch_vectors_from_qdrant(client: QdrantClient, collection_name: str, project_id: int, logger=None) -> dict:
    """
    Fetch all vectors and their "id" from a Qdrant collection where "projectId" matches the given project_id.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    vectors_with_ids = {}
    offset = None

    logger.info(f'Start fetching vectors from Qdrant collection {collection_name} for projectId {project_id}')

    while True:
        response = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_vectors=True,
            with_payload=False,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="projectId",
                        match=MatchValue(value=project_id)
                    )
                ]
            )
        )

        for point in response[0]:
            vectors_with_ids[point.id] = point.vector

        if response[1] is None:
            break
        offset = response[1]

    logger.info(f'Fetched {len(vectors_with_ids)} vectors from Qdrant')
    return vectors_with_ids


def identify_parents(social_circle_df,persons_details_df, gallery_info_df, logger):
    """
    Identify images that contain bride/groom with their parents and update
    'cluster_context' from 'portrait' to 'portrait with parent' for those images.

    Inputs:
        social_circle_df:
            - columns: ['identity_ids', ...]
            - 'identity_ids' is a list of identityNumeralId in each social circle

        persons_details_df:
            - columns: ['identity_id', 'age', 'gender', ...]
            - one row per identity

        gallery_info_df:
            - columns at least:
                ['image_id', 'persons_ids', 'main_persons', 'cluster_context']
            - 'persons_ids' is list of identities in that image
            - 'main_persons' is a list of [bride_id, groom_id] (or similar)

    Returns:
        Updated gallery_info_df (copy) with 'cluster_context' changed
        from 'portrait' to 'portrait with parent' where relevant.
    """
    if social_circle_df is None:
        logger.info(f'No social circle data found to identify parents')
        return gallery_info_df

    df = gallery_info_df.copy()

    # --- Build lookup dicts for age & gender ---
    id_to_age = persons_details_df.set_index("identity_id")["age"].to_dict()
    id_to_gender = persons_details_df.set_index("identity_id")["gender"].to_dict()

    # Collect all main_persons ids
    main_ids_series = df["main_persons"].dropna()
    main_ids = (
        main_ids_series.explode()
        .dropna()
        .unique()
    )
    # does'nt matter the gender whether its bride or groom
    bride_id = main_ids[0]
    groom_id = main_ids[1]
    bride_age = id_to_age.get(bride_id)
    groom_age = id_to_age.get(groom_id)

    all_person_ids = (
        df["persons_ids"]
        .dropna()
        .explode()
    )

    # # Filter out bride & groom
    # all_person_ids = all_person_ids[~all_person_ids.isin([bride_id, groom_id])]
    #
    # id_counts = Counter(all_person_ids)
    # # Get top 5 most common ids (if <5 exist, get them all)
    # top5_ids = [pid for pid, _ in id_counts.most_common(5)]


    if bride_age is None or groom_age is None:
        logger.info("Missing age information for bride or groom; skipping parent detection.")
        return df

    couple_pairs = set()  # set of frozenset({id1, id2})

    for _, row in social_circle_df.iterrows():
        ids = row.get("identity_ids") or []
        # here only for the couple
        # if len(ids) == 2:
        couple_pairs.update(ids)

    AGE_TOLERANCE = 10.0

    portrait_df = df[df["cluster_context"] == "portrait"].copy()

    def classify_persons(persons_ids):
        """
        Classify a single image based only on persons_ids.

        Rules:
          - must have exactly 4 distinct people
          - must contain bride_id and groom_id
          - the remaining 2 ids must form a couple in social_circle_df
        """
        if not isinstance(persons_ids, (list, tuple)):
            return None

        persons_set = set(persons_ids)

        # exactly 4 distinct people or 3 people
        if len(persons_set) != 4 and len(persons_set) != 3:
            return None

        # must contain both bride & groom
        if bride_id not in persons_set and groom_id not in persons_set:
            return None

        # get the other two
        remaining = list(persons_set - {bride_id, groom_id})
        if len(remaining) != 2:
            # should not happen if len(persons_set) == 4, but be safe
            return None

        pair = set(remaining)

        parent_age_1 = id_to_age.get(remaining[0])
        parent_age_2 = id_to_age.get(remaining[1])

        parent_gender_1 = id_to_gender.get(remaining[0])
        parent_gender_2 = id_to_gender.get(remaining[1])

        # check if this pair is a known "couple" from social circles
        if not (pair & couple_pairs) or (parent_gender_1 == parent_gender_2):
            return None

        if abs(parent_age_1 - (bride_age + 20.0)) <= AGE_TOLERANCE or  abs(parent_age_2 - (groom_age + 20.0)) <= AGE_TOLERANCE or abs(parent_age_2 - (bride_age + 20.0)) <= AGE_TOLERANCE or abs(parent_age_1 - (groom_age + 20.0)) <= AGE_TOLERANCE:
            if bride_id in persons_set and groom_id not in persons_set:
                 return "bride with her parents"
            elif bride_id not in persons_set and groom_id in persons_set:
                 return "groom with his parents"
            else:
                return  "bride and groom with parents"
        else:
            return None

    # classify only portrait rows
    portrait_df["parent_category"] = portrait_df["persons_ids"].apply(classify_persons)


    to_print = portrait_df[
    (portrait_df["parent_category"] == "bride and groom with parents") |
    (portrait_df["parent_category"] == "bride with her parents") |
    (portrait_df["parent_category"] == "groom with his parents")
    ]
    plot_selected_rows_to_pdf(to_print)
    print("plotting done")


    # update cluster_context where a category was found
    mask = portrait_df["parent_category"].notna()
    portrait_df.loc[mask, "cluster_context"] = portrait_df.loc[mask, "parent_category"]

    # push changes back into main df
    df.update(portrait_df)

    # optional: drop helper column if it ends up in df
    df.drop(columns=["parent_category"], inplace=True, errors="ignore")

    if logger:
        updated_count = int(mask.sum())
        logger.info(f"Updated {updated_count} images to 'bride and groom with parents' based on 4-person couples.")

    return df


def read_messages(messages, project_status_collection, qdrant_client, logger):
    enriched_messages = []

    for _msg in messages:
        reading_message_time = datetime.now()

        json_content = _msg.content
        if not (type(json_content) is dict or type(json_content) is list):
            logger.warning('Incorrect message format: {}.'.format(json_content))
        logger.info('Received message: {}/{}'.format(json_content, _msg))

        if 'photos' not in json_content or 'base_url' not in json_content:
            return None, 'There are missing fields in input request: {}. Skipping.'.format(json_content)

        try:
            message = read_layouts_data(_msg, json_content)
            _msg = message
            json_content = _msg.content

            proto_start = datetime.now()
            project_url = json_content['base_url']
            project_id = json_content['projectId']
            # Fetch the document from the collection
            try:
                logger.info(f"Fetch the document from the collection {project_status_collection}")
                if isinstance(project_id, int):
                    doc = project_status_collection.find_one({"_id": project_id},
                                                                  {"isInVectorDatabase": 1, "imageModelVersion": 1})
                else:
                    doc = project_status_collection.find_one({"_id": ObjectId(project_id)},
                                                                  {"isInVectorDatabase": 1, "imageModelVersion": 1})

                if doc is None:
                    logger.info(f"doc not found for project_id {project_id}")
                    is_in_vector_db = None
                    image_model_version = None
                else:
                    logger.info(f"doc found for project_id {project_id}: {doc}")
                    is_in_vector_db = doc.get("isInVectorDatabase")
                    image_model_version = doc.get("imageModelVersion")
            except Exception as ex:
                logger.warning(f"Failed to read one message: {ex}")
                is_in_vector_db = None
                image_model_version = None

            # Retrieve the isInVectorDB field

            if is_in_vector_db is not None and is_in_vector_db == True:
                logger.info(
                    f'Project {project_id} has isInVectorDB = True, loading Clip embeddings from qdrant')
                collection_name = CONFIGS["QDRANT_COLLECTION"][image_model_version]
                try:
                    clip_dict = fetch_vectors_from_qdrant(qdrant_client, collection_name, project_id,
                                                          logger=logger)
                    clip_version = image_model_version
                    _msg.clip_version = clip_version
                    clip_df = pd.DataFrame([
                        {"image_id": photo_id, "embedding": data}
                        for photo_id, data in clip_dict.items()
                    ])
                    clip_df['model_version'] = image_model_version
                except Exception as ex:
                    _msg.error = True
                    raise Exception('Qdrant fetch error: {}'.format(ex))
            else:
                clip_df = None


            gallery_info_df, is_wedding,social_circle_df,persons_details_df, pt_error = get_info_protobufs(project_base_url=project_url, logger=logger,clip_df=clip_df)
            if pt_error is not None:
                return None, pt_error
            logger.info(f"Reading Files protos for  {len(gallery_info_df)} images is: {datetime.now() - proto_start} secs.")

            # add scenes info to gallery_info_df
            gallery_info_df = add_scenes_info(gallery_info_df, project_url, logger)

            # add time data
            gallery_info_df,is_artificial_time = process_gallery_time(_msg, gallery_info_df, logger)

            # detect Ceremony kiss photos
            gallery_info_df = identify_kiss_ceremony(gallery_info_df, logger=logger)

            # parents detection
            gallery_info_df = identify_parents(social_circle_df,persons_details_df,gallery_info_df, logger=logger)

            if not gallery_info_df.empty:
                _msg.content['gallery_photos_info'] = gallery_info_df
                _msg.content['is_wedding'] = is_wedding
                _msg.content['is_artificial_time'] = is_artificial_time
                enriched_messages.append(_msg)
            else:
                return None, 'Failed to enrich image data for message: {}. Skipping.'.format(json_content)

            logger.info(
                f"Reading Time Stage for one Gallery  {len(gallery_info_df)} images is: {datetime.now() - reading_message_time} secs. message id: {_msg.source.id}")

        except Exception as ex:
            tb = traceback.extract_tb(ex.__traceback__)
            filename, lineno, func, text = tb[-1]
            return None, f'Error reading messages at reading stage: {ex}. Exception in function: {func}, line {lineno}, file {filename}.'

    return enriched_messages, None


def convert_int64_to_int(obj):
    if isinstance(obj, dict):
        return {key: convert_int64_to_int(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_int64_to_int(item) for item in obj]
    elif isinstance(obj, np.int64):
        return int(obj)
    else:
        return obj


def customize_box(image_info, box_info, album_ar=2):
    target_ar = box_info['width'] / box_info['height'] * album_ar
    if box_info['orientation'] == 'square':
        crop_x = image_info['cropped_x']
        crop_y = image_info['cropped_y']
        crop_w = image_info['cropped_w']
        crop_h = image_info['cropped_h']

        return crop_x, crop_y, crop_w, crop_h
    else:
        image_ar = float(image_info['image_as'])
        if image_ar > target_ar:
            # Image is too wide, crop horizontally
            new_width_ratio = target_ar / image_ar
            x = (1 - new_width_ratio) / 2
            y = 0.0
            w = new_width_ratio
            h = 1.0
        else:
            # Image is too tall, crop vertically
            new_height_ratio = image_ar / target_ar
            x = 0.0
            y = (1 - new_height_ratio) / 2
            w = 1.0
            h = new_height_ratio

        return x, y, w, h


def sort_boxes(boxes):
    xs = [box['x'] for box in boxes]
    ys = [box['y'] for box in boxes]
    page_flag = [0 if box['x'] < 0.5 else 1 for box in boxes]

    sorted_indices = np.lexsort((xs, ys, page_flag))

    sorted_boxes = [boxes[ind] for ind in sorted_indices]
    return sorted_boxes


def get_mirrored_boxes(boxes):
    mirrored_boxes = [box.copy() for box in boxes]
    for mirrored_box in mirrored_boxes:
        if mirrored_box is not None and 'x' in mirrored_box and 'width' in mirrored_box:
            mirrored_box['x'] = 1 - mirrored_box['x'] - mirrored_box['width']

    return sort_boxes(mirrored_boxes)


def assembly_output(output_list, message, images_df, first_last_pages_data_dict, album_ar = 2,logger=None):
    result_dict = dict()
    result_dict['compositions'] = list()
    result_dict['placementsTxt'] = list()
    result_dict['placementsImg'] = list()
    result_dict['userJobId'] = message.content['userJobId']
    result_dict['compositionPackageId'] = message.content['compositionPackageId']
    result_dict['productId'] = message.content['designInfo']['productId']
    result_dict['packageDesignId'] = None
    result_dict['projectId'] = message.content['projectId']
    result_dict['storeId'] = message.content['storeId']
    result_dict['accountId'] = message.content['accountId']
    result_dict['userId'] = message.content['userId']
    counter_comp_id = 0
    counter_image_id = 0

    original_designs_data = message.content['designInfo']['designs']
    layouts_df = message.designsInfo['anyPagelayouts_df']
    box_id2data = message.designsInfo['anyPagebox_id2data']
    # adding the Album Cover
    if 'cover' in message.pagesInfo.keys():
        result_dict['compositions'].append({"compositionId": counter_comp_id,
                                       "compositionPackageId": message.content['compositionPackageId'],
                                       "designId":  message.designsInfo['coverDesignIds'][0] ,
                                       "styleId": message.designsInfo['defaultPackageStyleId'],
                                       "revisionCounter": 0,
                                       "copies": 1,
                                       "boxes": None,
                                       "logicalSelectionsState": None})
        counter_comp_id += 1

    # adding the first spread image
    if 'firstPage' in first_last_pages_data_dict.keys() and first_last_pages_data_dict['firstPage']['first_images_df'] is not None:
        first_page_data = first_last_pages_data_dict['firstPage']
        first_page_layouts_df = message.designsInfo['firstPage_layouts_df']
        design_id = first_page_layouts_df.loc[first_page_data['design_id']]['id']
        if design_id > 0:
            design_boxes = sort_boxes(original_designs_data[str(design_id)]['boxes'])
        else:
            design_boxes = get_mirrored_boxes(original_designs_data[str(-1*design_id)]['boxes'])
            design_id = -1 * design_id
        left_box_ids = first_page_layouts_df.loc[first_page_data['design_id']]['left_box_ids']
        right_box_ids = first_page_layouts_df.loc[first_page_data['design_id']]['right_box_ids']
        all_box_ids = left_box_ids + right_box_ids
        result_dict['compositions'].append({"compositionId": counter_comp_id,
                                       "compositionPackageId": message.content['compositionPackageId'],
                                       "designId": design_id,
                                       "styleId": message.designsInfo['defaultPackageStyleId'],
                                       "revisionCounter": 0,
                                       "copies": 1,
                                       "boxes": design_boxes,
                                       "logicalSelectionsState": None})

        for idx, box_id in enumerate(all_box_ids):
            x, y, w, h = customize_box(first_page_data['first_images_df'].iloc[idx], box_id2data[(design_id,box_id)],album_ar)
            result_dict['placementsImg'].append({"placementImgId": counter_image_id,
                                            "compositionId": counter_comp_id,
                                            "compositionPackageId": message.content['compositionPackageId'],
                                            "boxId": box_id,
                                            "photoId": first_page_data['first_images_ids'][idx],
                                            "cropX": x,
                                            "cropY": y,
                                            "cropWidth": w,
                                            "cropHeight": h,
                                            "rotate": 0,
                                            "projectId": message.content['projectId'],
                                            "photoFilter": 0,
                                            "photo": None})
        counter_comp_id += 1
        counter_image_id += 1


    # Add images
    for number_groups,group_dict in enumerate(output_list):
        for group_name in group_dict.keys():
            group_result = group_dict[group_name]
            # logger.debug('Group name/result: {}: {}'.format(group_name, group_result))
            total_spreads = len(group_result)
            for i in range(total_spreads):
                group_data = group_result[i]
                if isinstance(group_data, float):
                    continue
                if isinstance(group_data, list):
                    number_of_spreads = len(group_data)

                    for spread_index in range(number_of_spreads):
                        layout_id = group_data[spread_index][0]

                        design_id = layouts_df.loc[layout_id]['id']
                        if design_id > 0:
                            design_boxes = sort_boxes(original_designs_data[str(design_id)]['boxes'])
                        else:
                            design_boxes = get_mirrored_boxes(original_designs_data[str(-1 * design_id)]['boxes'])
                            design_id = -1 * design_id

                        result_dict['compositions'].append({"compositionId": counter_comp_id,
                                                       "compositionPackageId": message.content['compositionPackageId'],
                                                       "designId": design_id,
                                                       "styleId": message.designsInfo['defaultPackageStyleId'],
                                                       "revisionCounter": 0,
                                                       "copies": 1,
                                                       "boxes": design_boxes,
                                                       "logicalSelectionsState": None})

                        cur_layout_info = layouts_df.loc[layout_id]['boxes_info']
                        left_box_ids = layouts_df.loc[layout_id]['left_box_ids']
                        right_box_ids = layouts_df.loc[layout_id]['right_box_ids']

                        left_page_photos = list(group_data[spread_index][1])
                        right_page_photos = list(group_data[spread_index][2])

                        all_box_ids = left_box_ids + right_box_ids
                        all_photos = left_page_photos + right_page_photos

                        # Loop over boxes and plot images
                        for j, box in enumerate(cur_layout_info):
                            box_id = box['id']
                            if box_id not in all_box_ids:
                                print('Some error, cant find box with id: {}'.format(box_id))

                            element_index = all_box_ids.index(box_id)
                            cur_photo = all_photos[element_index]
                            image_id = cur_photo.id

                            image_info = images_df[images_df["image_id"] == image_id]
                            if image_info is None or image_info.empty:
                                continue
                            else:
                                x, y, w, h = customize_box(image_info.iloc[0], box_id2data[(design_id,box_id)],album_ar)
                            result_dict['placementsImg'].append({"placementImgId" : counter_image_id,
                                                            "compositionId" : counter_comp_id,
                                                            "compositionPackageId": message.content['compositionPackageId'],
                                                            "boxId" : box_id,
                                                            "photoId" : image_id,
                                                            "cropX" : x,
                                                            "cropY" : y,
                                                            "cropWidth" : w,
                                                            "cropHeight" : h,
                                                            "rotate" : 0,
                                                            "projectId" : message.content['projectId'],
                                                            "photoFilter" : 0,
                                                            "photo" : None})
                            counter_image_id += 1
                        counter_comp_id += 1


    # adding the last page
    if 'lastPage' in first_last_pages_data_dict.keys() and first_last_pages_data_dict['lastPage'][
        'last_images_df'] is not None:
        last_page_data = first_last_pages_data_dict['lastPage']
        last_page_layouts_df = message.designsInfo['lastPage_layouts_df']
        design_id = last_page_layouts_df.loc[last_page_data['design_id']]['id']
        if design_id > 0:
            design_boxes = sort_boxes(original_designs_data[str(design_id)]['boxes'])
        else:
            design_boxes = get_mirrored_boxes(original_designs_data[str(-1*design_id)]['boxes'])
            design_id = -1 * design_id
        left_box_ids = last_page_layouts_df.loc[last_page_data['design_id']]['left_box_ids']
        right_box_ids = last_page_layouts_df.loc[last_page_data['design_id']]['right_box_ids']
        all_box_ids = left_box_ids + right_box_ids
        result_dict['compositions'].append({"compositionId": counter_comp_id,
                                            "compositionPackageId": message.content['compositionPackageId'],
                                            "designId": design_id,
                                            "styleId": message.designsInfo['defaultPackageStyleId'],
                                            "revisionCounter": 0,
                                            "copies": 1,
                                            "boxes": design_boxes,
                                            "logicalSelectionsState": None})

        for idx, box_id in enumerate(all_box_ids):
            x, y, w, h = customize_box(last_page_data['last_images_df'].iloc[idx], box_id2data[(design_id,box_id)],album_ar)
            result_dict['placementsImg'].append({"placementImgId": counter_image_id,
                                                 "compositionId": counter_comp_id,
                                                 "compositionPackageId": message.content['compositionPackageId'],
                                                 "boxId": box_id,
                                                 "photoId": last_page_data['last_images_ids'][idx],
                                                 "cropX": x,
                                                 "cropY": y,
                                                 "cropWidth": w,
                                                 "cropHeight": h,
                                                 "rotate": 0,
                                                 "projectId": message.content['projectId'],
                                                 "photoFilter": 0,
                                                 "photo": None})

    result_dict = convert_int64_to_int(result_dict)

    final_result = {
        'requestId': message.content['conditionId'],
        'error': message.error,
        'composition': result_dict
    }
    return final_result
