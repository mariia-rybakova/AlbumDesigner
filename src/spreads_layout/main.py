from typing import List, Dict, Any, Optional, Tuple
import math
from gc import collect
import time

import pandas as pd

from src.core.models import AlbumDesignResources, Spread, GroupProcessingResult, SpreadSearchParams
from src.core.photos import get_photos_from_db, Photo
from src.spreads_layout.partitions import get_partitions
from src.spreads_layout.combinations import get_combinations
from src.spreads_layout.spreads.group_of_lists_of_spreads import process_combination, GroupLayoutsLists
from src.spreads_layout.group_layouts import GroupSingleLayout, process_group_lists, assign_photos_order


def get_group_photos_list(cur_group_photos: List[Photo], spread_params: List[float],
                          largest_layout_size: int, logger) -> List[List[Photo]]:
    """
    Split a photo group into smaller sub-groups if it is too large for a single search.

    A group is split when its size relative to the optimal spread parameter suggests
    4+ spreads would be needed. Splits are as equal as possible, distributing remainder
    photos across the first sub-groups.

    Args:
        cur_group_photos: Photos in the group, already sorted.
        spread_params: [mean, std] recommended photos per spread for this context.
        largest_layout_size: Maximum number of boxes in any available layout.
        logger: Logger instance.

    Returns:
        List of photo sub-groups. Contains a single element if no split is needed.
    """
    cur_group_photos_list = []

    optimal_spread_param = min(largest_layout_size, spread_params[0])
    if len(cur_group_photos) / (max(optimal_spread_param - 2 * spread_params[1], 1)) >= 4:
        split_size = min(optimal_spread_param * 3, max(optimal_spread_param, 11))
        number_of_splits = math.ceil(len(cur_group_photos) / split_size)
        logger.info('Condition we split!. Using splitting to {} parts'.format(number_of_splits))

        # Split as equally as possible
        total_items = len(cur_group_photos)
        base_size = total_items // number_of_splits
        remainder = total_items % number_of_splits

        start_idx = 0
        for split_num in range(number_of_splits):
            # Add 1 extra item to the first 'remainder' splits
            current_size = base_size + (1 if split_num < remainder else 0)
            end_idx = start_idx + current_size
            cur_group_photos_list.append(cur_group_photos[start_idx:end_idx])
            start_idx = end_idx
    else:
        cur_group_photos_list.append(cur_group_photos)

    return cur_group_photos_list


def generate_filtered_multi_spreads(photos: List[Photo], layouts_df: pd.DataFrame, spread_params: List[float], params: SpreadSearchParams, logger) -> Optional[List[GroupSingleLayout]]:
    photos_df = pd.DataFrame([photo.__dict__ for photo in photos])
    photos_df = photos_df.sort_values('general_time')

    # stage 1
    partitions = get_partitions(photos_df, spread_params, params, layouts_df=layouts_df)

    # stage 2
    combs = get_combinations(partitions, photos, layouts_df, spread_params, params)

    # stage 3
    # sample
    group_single_layouts = []
    for idx, comb in enumerate(combs):
        # stage 3.1
        # sample
        multispread_layouts = process_combination(comb, photos, layouts_df, params)
        if multispread_layouts is not None:
            group_single_layouts += process_group_lists(multispread_layouts)

        # filter
        if len(group_single_layouts) > 10000:
            group_single_layouts = sorted(group_single_layouts, key=lambda layout: layout.score, reverse=True)[:1000]

    if len(group_single_layouts) == 0:
        return None

    # filter
    filtered = sorted(group_single_layouts, key=lambda layout: layout.weight, reverse=True)
    max_score = filtered[0].weight
    filtered = [layout for layout in filtered if layout.weight / max_score > 0.01]

    return filtered[:1000]


def _find_spreads_for_group(group_photos, layouts_df, spread_params, params: SpreadSearchParams, largest_layout_size, group_name, logger):
    dummy_photo = Photo(id=-1, ar=1.5, color=True, rank=1000000, photo_class='None', cluster_label=1,
                        general_time=1000000, original_context='None')

    attempts = [
        (group_photos, [round(spread_params[0] * d), spread_params[1]])
        for d in [1.0, 0.8, 0.6, 0.4, 0.2]
    ] + [(group_photos + [dummy_photo], spread_params)]

    for cur_photos, cur_spread_params in attempts:
        cur_group_photos_list = get_group_photos_list(cur_photos, cur_spread_params, largest_layout_size, logger)
        groups_filtered_spreads_list = []

        for cur_sub_group_photos in cur_group_photos_list:
            cur_filtered_spreads = generate_filtered_multi_spreads(cur_sub_group_photos, layouts_df, cur_spread_params, params, logger)
            if cur_filtered_spreads is None:
                groups_filtered_spreads_list = None
                break
            else:
                groups_filtered_spreads_list.append((cur_sub_group_photos, cur_filtered_spreads))

        if groups_filtered_spreads_list is not None:
            if cur_photos is not group_photos:
                logger.info("Spread created using dummy photo for group: {}.".format(group_name))
            elif cur_spread_params[0] != spread_params[0]:
                logger.debug("Spreads found with params {}. Group: {}.".format(cur_spread_params, group_name))
            return groups_filtered_spreads_list

    logger.warning("Could not find spreads. Skipping group: {}.".format(group_name))
    return None


def rank_and_select_best_layout(filtered_layouts, sub_group_photos, layout_id2data, design_box_id2data):
    GroupSingleLayout.evaluate_list(filtered_layouts, sub_group_photos, layout_id2data)
    filtered_layouts = sorted(filtered_layouts, key=lambda x: x.weight, reverse=True)

    if not filtered_layouts:
        return None

    best_layout = filtered_layouts[0]

    # Retrieve Photo objects from photo indices
    for spread in best_layout.spreads_layouts:
        spread.resolve_photos(sub_group_photos)

    best_layout = assign_photos_order(best_layout, layout_id2data, design_box_id2data, merge_pages=False)
    return best_layout


def process_group(group_name: Tuple, group_images_df, spread_params: List[float],
                  resources: AlbumDesignResources, is_wedding: bool, params: SpreadSearchParams, logger) -> Optional[Dict[str, GroupProcessingResult]]:
    layouts_df = resources.layouts_df
    layout_id2data = resources.layout_id2data
    design_box_id2data = resources.box_id2data
    # print('\nprocessing group', group_name)

    largest_layout_size = max(list(layouts_df['number of boxes'].unique()))
    start = time.time()
    try:
        if is_wedding and 'dancing' in group_name[1]:
            group_images_df = group_images_df.sort_values(['image_as', 'image_time'])
        else:
            group_images_df = group_images_df.sort_values(['image_time'])

        cur_group_photos = get_photos_from_db(group_images_df, is_wedding)

        local_result = {}
        group_idx = 0
        final_groups_and_spreads = _find_spreads_for_group(cur_group_photos, layouts_df, spread_params, params, largest_layout_size, group_name, logger)

        if final_groups_and_spreads is not None:
            for sub_group_photos, filtered_layouts in final_groups_and_spreads:
                best_layout = rank_and_select_best_layout(filtered_layouts, sub_group_photos,
                                                           layout_id2data, design_box_id2data)

                if best_layout is None:
                    continue

                structured_spreads = [
                    Spread(layout_id=s.layout_idx, left_photos=s.left_page_photos, right_photos=s.right_page_photos)
                    for s in best_layout.spreads_layouts
                ]

                group_id_str = str(group_name[0]) + '_' + group_name[1] if is_wedding else str(group_name[0])
                group_id_str += '*' + str(group_idx)

                local_result[group_id_str] = GroupProcessingResult(group_name=group_id_str, spreads=structured_spreads, score=best_layout.score)
                group_idx += 1

        collect()
        end = time.time()
        logger.info(f"Processed group name {group_name} in {end - start:.2f} seconds.")

        return local_result

    except Exception as ex:
        import traceback
        tb = traceback.extract_tb(ex.__traceback__)
        filename, lineno, func, text = tb[-1]
        logger.error(f"Error processing group_name {group_name}: {ex}. Exception in function: {func}, line {lineno}, file {filename}.")
        return None