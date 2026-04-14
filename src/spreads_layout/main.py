from typing import List, Dict, Any, Optional, Tuple
import math
from gc import collect
import time

import pandas as pd

from src.core.models import AlbumDesignResources, Spread, GroupProcessingResult, SpreadSearchParams
from src.core.photos import get_photos_from_df, Photo
from src.spreads_layout.partitions import get_partitions
from src.spreads_layout.combinations import get_combinations
from src.spreads_layout.group_layouts import GroupSingleLayout, get_group_single_layouts


def split_group_if_needed(group_photos: List[Photo], spread_params: List[float],
                          largest_layout_size: int, logger) -> List[List[Photo]]:
    """
    Split a photo group into smaller sub-groups if it is too large for a single search.

    A group is split when its size relative to the optimal spread parameter suggests
    4+ spreads would be needed. Splits are as equal as possible, distributing remainder
    photos across the first sub-groups.

    Args:
        group_photos: Photos in the group, already sorted.
        spread_params: [mean, std] recommended photos per spread for this context.
        largest_layout_size: Maximum number of boxes in any available layout.
        logger: Logger instance.

    Returns:
        List of photo sub-groups. Contains a single element if no split is needed.
    """
    subgroups = []

    optimal_spread_param = min(largest_layout_size, spread_params[0])
    if len(group_photos) / (max(optimal_spread_param - 2 * spread_params[1], 1)) >= 4:
        split_size = min(optimal_spread_param * 3, max(optimal_spread_param, 11))
        number_of_splits = math.ceil(len(group_photos) / split_size)
        logger.info('Condition we split!. Using splitting to {} parts'.format(number_of_splits))

        # Split as equally as possible
        total_items = len(group_photos)
        base_size = total_items // number_of_splits
        remainder = total_items % number_of_splits

        start_idx = 0
        for split_num in range(number_of_splits):
            # Add 1 extra item to the first 'remainder' splits
            current_size = base_size + (1 if split_num < remainder else 0)
            end_idx = start_idx + current_size
            subgroups.append(group_photos[start_idx:end_idx])
            start_idx = end_idx
    else:
        subgroups.append(group_photos)

    return subgroups


def find_spreads_layouts_for_subgroup(photos: List[Photo], layouts_df: pd.DataFrame,
                                      layout_id2data: Dict[int, Any], spread_params: List[float],
                                      params: SpreadSearchParams, logger) -> Optional[List[GroupSingleLayout]]:
    """
    Find candidate spread layouts for a single sub-group of photos.

    Runs the three-stage layout search pipeline:
    1. Partition photos into spread-sized chunks based on spread_params.
    2. Generate combinations of partitions with compatible layouts.
    3. Sample, score, and filter GroupSingleLayout candidates.

    Args:
        photos: Photos in this sub-group.
        layouts_df: DataFrame of available layout designs.
        layout_id2data: Mapping from layout index to layout metadata.
        spread_params: [mean, std] recommended photos per spread.
        params: Search parameters controlling sampling limits.
        logger: Logger instance.

    Returns:
        Sorted list of GroupSingleLayout candidates (best first), or None if
        no valid layouts were found.
    """
    photos_df = pd.DataFrame([photo.__dict__ for photo in photos])
    photos_df = photos_df.sort_values('general_time')

    # stage 1
    partitions = get_partitions(photos_df, spread_params, params, layouts_df=layouts_df)

    # stage 2
    combs = get_combinations(partitions, photos, layouts_df, spread_params, params)

    # stage 3
    group_single_layouts = get_group_single_layouts(combs, photos, layouts_df, params, layout_id2data)

    return group_single_layouts


def find_spreads_layouts_for_group(group_photos: List[Photo], layouts_df: pd.DataFrame,
                                   layout_id2data: Dict[int, Any], spread_params: List[float],
                                   params: SpreadSearchParams, largest_layout_size: int,
                                   group_name: Tuple, logger) -> Optional[List[Tuple[List[Photo], List[GroupSingleLayout]]]]:
    """
    Find spread layouts for an entire group, with fallback attempts.

    Tries progressively relaxed spread parameters (scaling mean down by
    1.0, 0.8, 0.6, 0.4, 0.2) and a final attempt with a dummy photo appended.
    For each attempt, splits the group into sub-groups if needed and runs
    find_spreads_layouts_for_subgroup on each. Returns on the first successful attempt.

    Args:
        group_photos: All photos in this group, already sorted.
        layouts_df: DataFrame of available layout designs.
        layout_id2data: Mapping from layout index to layout metadata.
        spread_params: [mean, std] recommended photos per spread.
        params: Search parameters controlling sampling limits.
        largest_layout_size: Maximum number of boxes in any available layout.
        group_name: Tuple identifying the group (used for logging).
        logger: Logger instance.

    Returns:
        List of (sub_group_photos, layout_candidates) tuples on success, or None
        if all attempts failed.
    """
    dummy_photo = Photo(id=-1, ar=1.5, color=True, rank=1000000, photo_class='None', cluster_label=1,
                        general_time=1000000, original_context='None')

    attempts = [
        (group_photos, [round(spread_params[0] * d), spread_params[1]])
        for d in [1.0, 0.8, 0.6, 0.4, 0.2]
    ] + [(group_photos + [dummy_photo], spread_params)]

    for cur_photos, cur_spread_params in attempts:
        cur_subgroups = split_group_if_needed(cur_photos, cur_spread_params, largest_layout_size, logger)
        group_final_layouts = []

        for subgroup_photos in cur_subgroups:
            subgroup_spreads_layouts = find_spreads_layouts_for_subgroup(subgroup_photos, layouts_df, layout_id2data, cur_spread_params, params, logger)
            if subgroup_spreads_layouts is None:
                group_final_layouts = None
                break
            else:
                group_final_layouts.append((subgroup_photos, subgroup_spreads_layouts))

        if group_final_layouts is not None:
            if cur_photos is not group_photos:
                logger.info("Spread created using dummy photo for group: {}.".format(group_name))
            elif cur_spread_params[0] != spread_params[0]:
                logger.debug("Spreads found with params {}. Group: {}.".format(cur_spread_params, group_name))
            return group_final_layouts

    logger.warning("Could not find spreads. Skipping group: {}.".format(group_name))
    return None


def select_best_layout_for_subgroup(subgroup_layouts: List[GroupSingleLayout], subgroup_photos: List[Photo],
                                    layout_id2data: Dict[int, Any],
                                    design_box_id2data: Dict[Tuple[int, int], Any]) -> Optional[GroupSingleLayout]:
    """
    Select the top-ranked layout for a sub-group and resolve photo assignments.

    Takes the first (best-scored) layout from the sorted candidates, resolves
    photo indices to Photo objects, and assigns photos to layout boxes by
    orientation and area.

    Args:
        subgroup_layouts: Sorted list of GroupSingleLayout candidates (best first).
        subgroup_photos: Photos in this sub-group.
        layout_id2data: Mapping from layout index to layout metadata.
        design_box_id2data: Mapping from (layout_id, box_id) to box properties.

    Returns:
        The best GroupSingleLayout with photos resolved and ordered, or None
        if subgroup_layouts is empty.
    """
    if not subgroup_layouts:
        return None

    best_layout = subgroup_layouts[0]
    best_layout.resolve_and_order(subgroup_photos, layout_id2data, design_box_id2data, merge_pages=False)

    return best_layout


def structure_layout(best_layout: GroupSingleLayout, group_name: Tuple,
                     group_idx: int, is_wedding: bool = True) -> Tuple[str, GroupProcessingResult]:
    """
    Convert a GroupSingleLayout into the output GroupProcessingResult format.

    Builds a list of Spread objects from the layout's resolved photo assignments
    and generates a unique group ID string from the group name and sub-group index.

    Args:
        best_layout: The selected layout with photos resolved and ordered.
        group_name: Tuple identifying the group.
        group_idx: Sub-group index (for groups that were split).
        is_wedding: Whether this is a wedding album (affects group ID format).

    Returns:
        Tuple of (group_id_str, GroupProcessingResult).
    """
    structured_spreads = [
        Spread(layout_id=s.layout_idx, left_photos=s.left_page_photos, right_photos=s.right_page_photos)
        for s in best_layout.spreads_layouts
    ]

    group_id_str = str(group_name[0]) + '_' + group_name[1] if is_wedding else str(group_name[0])
    group_id_str += '*' + str(group_idx)

    structured_group = GroupProcessingResult(group_name=group_id_str, spreads=structured_spreads,
                                                               score=best_layout.score)

    return group_id_str, structured_group


def select_best_layout_for_group(final_groups_and_layouts: Optional[List[Tuple[List[Photo], List[GroupSingleLayout]]]],
                                 layout_id2data: Dict[int, Any],
                                 design_box_id2data: Dict[Tuple[int, int], Any],
                                 group_name: Tuple, is_wedding: bool) -> Dict[str, GroupProcessingResult]:
    """
    Select the best layout for each sub-group and structure the results.

    Iterates over sub-groups, picks the best layout for each via
    select_best_layout_for_subgroup, converts it to the output format via
    structure_layout, and collects all results into a dict keyed by group ID.

    Args:
        final_groups_and_layouts: List of (sub_group_photos, layout_candidates)
            tuples from find_spreads_layouts_for_group, or None.
        layout_id2data: Mapping from layout index to layout metadata.
        design_box_id2data: Mapping from (layout_id, box_id) to box properties.
        group_name: Tuple identifying the group.
        is_wedding: Whether this is a wedding album.

    Returns:
        Dict mapping group_id_str to GroupProcessingResult for each sub-group
        that produced a valid layout. Empty dict if input is None or all failed.
    """
    local_result = {}
    group_idx = 0
    if final_groups_and_layouts is not None:
        for subgroup_photos, subgroup_layouts in final_groups_and_layouts:
            best_layout = select_best_layout_for_subgroup(subgroup_layouts, subgroup_photos, layout_id2data, design_box_id2data)
            if best_layout is None:
                continue

            group_id_str, structured_group = structure_layout(best_layout, group_name, group_idx, is_wedding)

            local_result[group_id_str] = structured_group
            group_idx += 1

    return local_result


def process_group(group_name: Tuple, group_images_df: pd.DataFrame, spread_params: List[float],
                  resources: AlbumDesignResources, is_wedding: bool, params: SpreadSearchParams,
                  logger) -> Optional[Dict[str, GroupProcessingResult]]:
    """
    Top-level entry point: process a single photo group into album spreads.

    Orchestrates the full layout pipeline for one group:
    1. Sorts photos (by time, or by aspect ratio + time for dancing groups).
    2. Converts the DataFrame rows into Photo objects.
    3. Finds candidate spread layouts with fallback attempts.
    4. Selects the best layout per sub-group and structures the output.

    Args:
        group_name: Tuple identifying the group (e.g. (cluster_idx, context)).
        group_images_df: DataFrame of images belonging to this group.
        spread_params: [mean, std] recommended photos per spread.
        resources: AlbumDesignResources containing layouts_df, layout_id2data, box_id2data.
        is_wedding: Whether this is a wedding album.
        params: Search parameters controlling sampling limits.
        logger: Logger instance.

    Returns:
        Dict mapping group_id_str to GroupProcessingResult, or None on error.
    """
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

        group_photos = get_photos_from_df(group_images_df, is_wedding)

        final_groups_and_layouts = find_spreads_layouts_for_group(group_photos, layouts_df, layout_id2data, spread_params,
                                                                  params, largest_layout_size, group_name, logger)

        local_result = select_best_layout_for_group(final_groups_and_layouts, layout_id2data, design_box_id2data, group_name, is_wedding)

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