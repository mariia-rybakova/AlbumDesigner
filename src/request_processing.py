import pandas as pd
import numpy as np
import traceback

from datetime import datetime

from pycparser.c_ast import Continue

from utils.configs import CONFIGS
from utils.layouts_tools import generate_layouts_df, get_layouts_data, order_boxes_indices
from ptinfra.azure.pt_file import PTFile
from ptinfra.utils.gallery import Gallery
import json
from qdrant_client import QdrantClient, models
from pymongo import MongoClient
from src.core.models import GroupProcessingResult
from src.smart_cropping import face_aware_crop

def read_layouts_data(message, json_content, logger=None):
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
    message.designsInfo['maxPages'] = json_content['designInfo']['maxPages'] if 'maxPages' in json_content[
        'designInfo'] else CONFIGS['max_total_spreads']

    any_page_layouts_df = generate_layouts_df(json_content['designInfo']['designs'], message.designsInfo['anyPageIds'],
                                             album_ar=message.content.get('album_ar', {'anyPage': 2})['anyPage'],
                                             do_mirror=True)

    # Add dummy firstPage/lastPage from single-box anyPage layouts when missing
    if not any_page_layouts_df.empty:
        needs_first = first_page_layouts_df is None
        needs_last = last_page_layouts_df is None

        if needs_first or needs_last:
            single_box_df = any_page_layouts_df[
                (any_page_layouts_df['number of boxes'] == 1) &
                (any_page_layouts_df['is_mirrored'] == False)
            ]
            single_box_ids = single_box_df['id'].astype(int).unique().tolist()

            if single_box_ids:
                album_ar_val = message.content.get('album_ar', {'anyPage': 2})['anyPage']
                designs_data = json_content['designInfo']['designs']

                if needs_first:
                    first_page_layouts_df = generate_layouts_df(designs_data, single_box_ids, album_ar=album_ar_val)
                    if not first_page_layouts_df.empty:
                        message.designsInfo['firstPageDesignIds'] = single_box_ids
                        message.pagesInfo['firstPage'] = True
                        message.designsInfo['firstPage_layouts_df'] = first_page_layouts_df
                        if logger:
                            logger.info(f"Added dummy firstPage section with {len(single_box_ids)} single-box layout(s) from anyPage")

                if needs_last:
                    last_page_layouts_df = generate_layouts_df(designs_data, single_box_ids, album_ar=album_ar_val)
                    if not last_page_layouts_df.empty:
                        message.designsInfo['lastPageDesignIds'] = single_box_ids
                        message.pagesInfo['lastPage'] = True
                        message.designsInfo['lastPage_layouts_df'] = last_page_layouts_df
                        if logger:
                            logger.info(f"Added dummy lastPage section with {len(single_box_ids)} single-box layout(s) from anyPage")
            else:
                if logger:
                    if needs_first:
                        logger.info("No single-box layouts in anyPage; cannot add dummy firstPage section")
                    if needs_last:
                        logger.info("No single-box layouts in anyPage; cannot add dummy lastPage section")

    if not any_page_layouts_df.empty:
        message.designsInfo['anyPagelayouts_df'] = any_page_layouts_df
        layout_id2data, box_id2data = get_layouts_data(any_page_layouts_df, first_page_layouts_df, last_page_layouts_df, logger=logger)
        message.designsInfo['anyPagelayout_id2data'] = layout_id2data
        message.designsInfo['anyPagebox_id2data'] = box_id2data

    return message


def read_rating_data(message, json_content, logger=None):
    rating_list = json_content.get('rating')

    if not rating_list and json_content.get('ratingTempLocation'):
        try:
            fb = PTFile(json_content['ratingTempLocation'])
            fileBytes = fb.read_blob()
            rating_list = json.loads(fileBytes.decode('utf-8'))
        except Exception as e:
            if logger:
                logger.warning('Failed to read rating from blob {}: {}'.format(
                    json_content['ratingTempLocation'], e))
            return message

    if not rating_list:
        return message

    try:
        rating_df = pd.DataFrame(rating_list).rename(
            columns={'photoId': 'image_id', 'rating': 'user_rating'}
        )[['image_id', 'user_rating']]
        rating_df['image_id'] = rating_df['image_id'].astype(np.int64)
        message.rating_df = rating_df
    except Exception as e:
        if logger:
            logger.warning('Failed to build rating_df: {}'.format(e))

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
                    if "_" in filename:
                        continue
                    name = filename.split('.')[0]

                    if not name.isdigit():
                        continue
                    image_id2scene_image_order[np.int64(filename.split('.')[0])] = (scene_idx, image_iter)
                    image_iter += 1
                except Exception as e:
                    pass
    except Exception as e:
        logger.error(f"Error reading scenes info from gallery: {e}")

    print("Finshed with the photo scene")
    return gallery_info_df

    mapped = gallery_info_df['image_id'].map(image_id2scene_image_order)
    mapped_df = pd.DataFrame(mapped.tolist(), columns=['scene_order', 'image_order'])
    for col in ['scene_order', 'image_order']:
        gallery_info_df[col] = mapped_df[col].combine_first(gallery_info_df[col])

    gallery_info_df[['scene_order', 'image_order']] = gallery_info_df[['scene_order', 'image_order']].astype('Int64')

    return gallery_info_df



# `is_ceremony_gallery_sat` / `identify_kiss_ceremony` lived here. They are
# superseded by src/pipeline/enrich/ceremony_anchor.py, which anchors on the
# median ceremony climax in sequence positions instead of the last officiant
# frame in wall-clock minutes -- the old pair could not run at all on a
# gallery with unusable EXIF timestamps.


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

    # all_person_ids = (
    #     df["persons_ids"]
    #     .dropna()
    #     .explode()
    # )

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

        if abs(parent_age_1 - (bride_age + 15.0)) <= AGE_TOLERANCE or  abs(parent_age_2 - (groom_age + 15.0)) <= AGE_TOLERANCE or abs(parent_age_2 - (bride_age + 15.0)) <= AGE_TOLERANCE or abs(parent_age_1 - (groom_age + 15.0)) <= AGE_TOLERANCE:
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


    # to_print = portrait_df[
    # (portrait_df["parent_category"] == "bride and groom with parents") |
    # (portrait_df["parent_category"] == "bride with her parents") |
    # (portrait_df["parent_category"] == "groom with his parents")
    # ]
    # plot_selected_rows_to_pdf(to_print)
    # print("plotting done")


    # update cluster_context where a category was found
    mask = portrait_df["parent_category"].notna()
    portrait_df.loc[mask, "cluster_context"] = "parents portrait"

    # df.loc[portrait_df.index, ["parent_category", "cluster_context"]] = portrait_df[
    #     ["parent_category", "cluster_context"]]

    df.loc[portrait_df.index, "parent_category"] = portrait_df["parent_category"].astype("object")
    df.loc[portrait_df.index, "cluster_context"] = portrait_df["cluster_context"].astype("object")

    if logger:
        updated_count = int(mask.sum())
        logger.info(f"Updated {updated_count} images to 'bride and groom with parents' based on 4-person couples.")

    return df


def read_messages(messages, project_status_collection, qdrant_client, logger):
    """Read and enrich each incoming message.

    A thin driver over the ingest + enrich substages. The work that used to be
    inlined here now lives in `src/pipeline/ingest` (reading) and
    `src/pipeline/enrich` (everything derived from what was read); see
    `src.pipeline.registry.INGEST` / `ENRICH` for the order.

    Signature and return contract are unchanged: `(messages, error)`, where a
    non-None error aborts the whole batch.
    """
    # Imported here rather than at module scope: the ingest substages import
    # helpers from this module, so a top-level import would be circular.
    from src.pipeline import AlbumContext, Services, build_read

    pipeline = build_read(logger=logger)
    services = Services(
        project_status_collection=project_status_collection,
        qdrant_client=qdrant_client,
    )

    enriched_messages = []

    for _msg in messages:
        reading_message_time = datetime.now()
        logger.info('Received message: {}/{}'.format(_msg.content, _msg))

        context = pipeline.run(
            AlbumContext.from_message(_msg, logger=logger, services=services)
        )

        if context.failed:
            return None, context.error

        if context.photos is None or context.photos.empty:
            return None, 'Failed to enrich image data for message: {}. Skipping.'.format(_msg.content)

        enriched_messages.append(context.sync_to_message())

        logger.info(
            f"Reading Time Stage for one Gallery  {len(context.photos)} images is: "
            f"{datetime.now() - reading_message_time} secs. message id: {_msg.source.id}")

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


def box_target_ar(box_info, album_ar=2):
    """The box's width/height in image terms, which is what a crop is fitted to."""
    return box_info['width'] / box_info['height'] * album_ar


def cover_box(image_info, box_info, album_ar=2, logger=None):
    """`customize_box` for the covers, but positioned to keep the faces.

    Covers only, on purpose. `customize_box` centres its window blind for
    every placement in the album, and widening this to all of them would move
    the crops on every spread; the covers are where it shows, because they are
    the one photo the album opens on and the box is 1.96:1 -- a portrait keeps
    34% of its height there, so a centred band lands below the faces and takes
    chins. Every other placement keeps the existing behaviour.
    """
    if box_info['orientation'] == 'square':
        return customize_box(image_info, box_info, album_ar)

    crop = face_aware_crop(image_info, box_target_ar(box_info, album_ar), logger)
    if crop is None:
        return customize_box(image_info, box_info, album_ar)
    return crop


def customize_box(image_info, box_info, album_ar=2):
    target_ar = box_target_ar(box_info, album_ar)
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
    # Reading order with a y-tolerance so sub-pixel y noise in design data
    # doesn't flip the left/right order of side-by-side boxes. Must match the
    # ordering used to build left_box_ids/right_box_ids (utils.layouts_tools).
    sorted_indices = order_boxes_indices(boxes)
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
    result_dict["fulfillerId"] = message.content['fulfillerId']

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
    logger.info(f"Added Album Cover composition with id {counter_comp_id-1}")
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
            x, y, w, h = cover_box(first_page_data['first_images_df'].iloc[idx], box_id2data[(design_id,box_id)], album_ar, logger)
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

    logger.info(f"Added Album Cover composition with id {counter_comp_id-1}")

    # Add images
    i = 0

    def list_ids(ph_l):
        ids = []
        for ph in ph_l:
            id = ph.id if ph is not None else None
            ids.append(id)
        return ids

    for number_groups, group_dict in enumerate(output_list):
        for group_id, result in group_dict.items():
            if not isinstance(result, GroupProcessingResult):
                logger.warning(f"Unexpected result type for group {group_id}: {type(result)}")
                continue

            for spread in result.spreads:
                i += 2
                logger.info(f'Pages {i - 1}-{i} - left photos ({len(spread.left_photos)}): {list_ids(spread.left_photos)}, '
                            f'right photos ({len(spread.right_photos)}): {list_ids(spread.right_photos)}')

                layout_id = spread.layout_id

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

                left_page_photos = spread.left_photos
                right_page_photos = spread.right_photos

                all_box_ids = left_box_ids + right_box_ids
                all_photos = left_page_photos + right_page_photos

                # Loop over boxes and plot images
                for j, box in enumerate(cur_layout_info):
                    box_id = box['id']
                    if box_id not in all_box_ids:
                        logger.info('Some error, cant find box with id: {}'.format(box_id))

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
                if logger is not None:
                    logger.debug(f'placementImg length {len(result_dict["placementsImg"])}')

    logger.info(f"Added Album any page")
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
            x, y, w, h = cover_box(last_page_data['last_images_df'].iloc[idx], box_id2data[(design_id,box_id)], album_ar, logger)
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

    logger.info(f"Added last page composition with id {counter_comp_id}")

    result_dict = convert_int64_to_int(result_dict)

    final_result = {
        'requestId': message.content['conditionId'],
        'error': message.error,
        'composition': result_dict
    }

    if logger is not None:
        logger.debug(f'final_result placementImg length {len(final_result["composition"]["placementsImg"])}')

    return final_result
