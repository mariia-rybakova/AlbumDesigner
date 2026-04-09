from typing import List, Tuple, Iterable, Callable, Any, Optional

import pandas as pd

from src.core.models import AlbumDesignResources
from utils.album_tools import get_images_per_groups, get_missing_columns, split_groups
from src.groups_operations.merge import (get_merge_candidates_bridegroom, get_merge_candidates_other,
                                         update_with_merges_bridegroom, update_with_merges_other,
                                         BRIDE_CENTRIC_CLASSES, GROOM_CENTRIC_CLASSES,
                                         force_merge_portrait_singleton)
from src.groups_operations.split import (get_number_of_spreads, is_split_needed, get_split_points,
                                         split_big_group, split_diverse_group,
                                         update_groups_size, update_group_sub_index)
from utils.configs import CONFIGS


# Splitting
def get_groups_time(groups):
    general_times_list = list()
    group_key2time_list = dict()
    for group_key, group in groups:
        group_times = group['general_time'].values
        general_times_list.extend(group_times)
        group_key2time_list[group_key] = sorted(group_times)
    return sorted(general_times_list), group_key2time_list


def handle_wedding_splitting(photos_df, resources: AlbumDesignResources, logger=None):
    # handle splitting
    look_up_table = resources.look_up_table.table if hasattr(resources, 'look_up_table') else {}
    split_df = photos_df[photos_df['group_size'] >= CONFIGS['max_img_split']]
    split_groups_ = split_df.groupby(['time_cluster', 'cluster_context', 'group_sub_index'])
    general_times_list, group_key2time_list = get_groups_time(split_groups_)

    for group_key, group in split_groups_:
        group_spread_size = look_up_table.get(group_key[1], [10])[0]
        # Calculate average number of spreads for this group
        number_of_spreads = get_number_of_spreads(group, group_spread_size)
        # Check if group is too big and need to be split
        if is_split_needed(number_of_spreads, group_spread_size, group_key):
            # Split big group
            updated_group = split_big_group(group, group_spread_size)
        else:
            # Check if group is diverse in time and get split time points
            split_points = get_split_points(general_times_list, group_key2time_list[group_key], group_key=group_key[1])
            updated_group = split_diverse_group(group, split_points)

        update_group_sub_index(photos_df, updated_group, logger)

    update_groups_size(photos_df)
    return photos_df


# Merging
# Bride and groom
def handle_wedding_bride_groom_merge(photos_df, logger=None):
    def flatten(list_of_tuples):
        return [item for group in list_of_tuples for item in group]

    merge_df = photos_df[(photos_df['group_size'] < CONFIGS['max_img_split']) &
                         ((photos_df['cluster_context'].isin(flatten(BRIDE_CENTRIC_CLASSES))) |
                          (photos_df['cluster_context'].isin(flatten(GROOM_CENTRIC_CLASSES))))]
    targets_df = photos_df.copy()

    merge_groups = merge_df.groupby(['time_cluster', 'cluster_context', 'group_sub_index'])
    general_times_list, _ = get_groups_time(photos_df.groupby(['time_cluster', 'cluster_context', 'group_sub_index']))

    for cent_idx in range(len(BRIDE_CENTRIC_CLASSES)):
        merge_candidates = get_merge_candidates_bridegroom(merge_groups, targets_df, general_times_list, cent_idx=cent_idx)

        update_with_merges_bridegroom(photos_df, merge_groups, merge_candidates, cent_idx)

    return photos_df


# Other groups
def _update_group_spreads(photos_df: pd.DataFrame, look_up_table: dict):
    """
    Calculate group spread ratios for each photo group.
    """
    def compute_spread(row):
        if row['cluster_context'] in look_up_table:
            return row['group_size'] / look_up_table[row['cluster_context']][0]
        return 1

    photos_df['group_spreads'] = photos_df.apply(compute_spread, axis=1)


def _filter_merge_candidate_photos(df_chunk: pd.DataFrame, size_limit: int) -> pd.DataFrame:
    """
    Filter photo groups eligible for merging.

    Args:
        df_chunk (pd.DataFrame): Subset of photos_df (special or regular).
        size_limit (int): Maximum allowed merge times for this subset.

    Returns:
        pd.DataFrame: Filtered DataFrame of merge candidates.
    """
    return df_chunk[
        ((df_chunk['group_size'] < CONFIGS['max_img_split']) | (df_chunk['group_spreads'] < 1))
        & (df_chunk['merge_allowed'] == True)
        & (df_chunk['groups_merged'] < size_limit)
    ]


def process_wedding_merging(photos_df, resources: AlbumDesignResources, logger=None):
    look_up_table = resources.look_up_table.table if hasattr(resources, 'look_up_table') else {}
    possible_boxes_numbers = list(resources.layouts_df['number of boxes'].unique())

    _update_group_spreads(photos_df, look_up_table)     # add 'group_spreads' field

    mask_special = photos_df['cluster_context'].isin(['None', 'other'])
    df_special = photos_df[mask_special].copy()
    df_regular = photos_df[~mask_special].copy()

    merge_special_df = _filter_merge_candidate_photos(df_special, CONFIGS['none_limit_times'])
    merge_regular_df = _filter_merge_candidate_photos(df_regular, CONFIGS['merge_limit_times'])

    merge_df = pd.concat([merge_special_df, merge_regular_df])
    merge_groups = merge_df.groupby(['time_cluster', 'cluster_context', 'group_sub_index'])
    if merge_groups.ngroups == 0:
        return photos_df, False

    targets_df = photos_df.copy()
    targets_df = targets_df[(targets_df['merge_allowed'] == True) &
                            (targets_df['groups_merged'] < CONFIGS['merge_limit_times'])]

    general_times_list, _ = get_groups_time(photos_df.groupby(['time_cluster', 'cluster_context', 'group_sub_index']))

    merge_candidates = get_merge_candidates_other(merge_groups, targets_df, general_times_list, possible_boxes_numbers)

    if len(merge_candidates) == 0:
        return photos_df, False

    update_with_merges_other(photos_df, merge_groups, merge_candidates)
    return photos_df, True


# Portrait singleton resolution
def _try_add_unused_photo(photos_df, group_key, singleton_group, all_gallery_df, resources, logger):
    """
    Attempt to add an unused photo from the full gallery to grow a portrait singleton to size 2.

    Photo selection logic is TBD. Currently returns False so the merge fallback is always used.

    Args:
        photos_df: The current album DataFrame.
        group_key: Key of the singleton group.
        singleton_group: DataFrame containing the single portrait photo.
        all_gallery_df: Full gallery DataFrame (before AI selection).
        resources: AlbumDesignResources instance.
        logger: Logger instance.

    Returns:
        True if a photo was added, False otherwise.
    """
    selected_ids = set(photos_df['image_id'].values)
    unused_df = all_gallery_df[~all_gallery_df['image_id'].isin(selected_ids)]
    if unused_df.empty:
        return False

    # TODO: Implement photo selection logic (prefer same time_cluster, cluster_context, closest general_time)
    return False


def _resolve_singletons(photos_df, resources, manual_selection, logger, all_gallery_df=None):
    """
    Find and resolve all singleton groups (size 1) remaining after the merge pipeline.

    For manual selection: force-merge into the closest timeline group.
    For non-manual selection: try adding an unused photo first, fall back to force-merge.

    Args:
        photos_df: The full DataFrame of photos (modified in-place).
        resources: AlbumDesignResources instance.
        manual_selection: Whether this is a manual selection album.
        logger: Logger instance.
        all_gallery_df: Full gallery DataFrame for non-manual photo addition (optional).

    Returns:
        The modified photos_df.
    """
    groups = photos_df.groupby(['time_cluster', 'cluster_context', 'group_sub_index'])
    possible_boxes_numbers = list(resources.layouts_df['number of boxes'].unique())

    group_sizes = {k: len(g) for k, g in groups}
    logger.info(f"_resolve_singletons called. Groups: {group_sizes}")

    # Re-create groupby since the previous one was consumed by iteration
    groups = photos_df.groupby(['time_cluster', 'cluster_context', 'group_sub_index'])

    # Collect all singleton groups sorted by general_time for deterministic order
    singletons = []
    for group_key, group in groups:
        if len(group) == 1:
            singletons.append((group_key, group))

    if not singletons:
        logger.info("No singletons found to resolve")
        return photos_df

    singletons.sort(key=lambda x: x[1]['general_time'].iloc[0])
    logger.info(f"Found {len(singletons)} singleton group(s) to resolve: {[s[0] for s in singletons]}")

    for group_key, singleton_group in singletons:
        # Non-manual: try adding an unused photo first
        if not manual_selection and all_gallery_df is not None:
            if _try_add_unused_photo(photos_df, group_key, singleton_group, all_gallery_df, resources, logger):
                logger.info(f"Resolved singleton {group_key} by adding unused photo")
                continue

        # Force-merge into closest timeline group
        merged = force_merge_portrait_singleton(
            photos_df, group_key, singleton_group, possible_boxes_numbers, logger
        )
        if merged:
            logger.info(f"Resolved singleton {group_key} by force-merge")
        else:
            logger.warning(f"Could not resolve singleton {group_key}. No valid merge target.")

    return photos_df


# Illegal groups split/merge pipeline
def _get_groups(photos_df: pd.DataFrame, manual_selection: bool, logger) -> pd.DataFrame:
    # Check if required columns exist
    missing = get_missing_columns({'time_cluster', 'cluster_context', 'cluster_label'}, photos_df, logger)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    photos_df['group_sub_index'] = -1
    photos_df['group_size'] = -1
    if not manual_selection:
        df_special, df_regular, groups_special = split_groups(photos_df)
        for idx, (key, group_df) in enumerate(groups_special):
            group_size = len(group_df)
            df_special.loc[group_df.index, 'group_sub_index'] = idx
            df_special.loc[group_df.index, 'group_size'] = group_size
    else:
        df_special = photos_df.copy().iloc[0:0]
        df_regular = photos_df.copy()

    # Update regular groups
    update_groups_size(df_regular, clusters=['time_cluster', 'cluster_context'])

    photos_df = pd.concat([df_special, df_regular], ignore_index=True)
    return photos_df


def process_wedding_illegal_groups(photos_df, resources: AlbumDesignResources, manual_selection, logger=None,
                                   max_iterations=500, all_gallery_df=None):
    photos_df = _get_groups(photos_df, manual_selection, logger)

    iteration = 0
    try:
        photos_df = handle_wedding_splitting(photos_df, resources, logger)

        photos_df['merge_allowed'] = True
        photos_df.loc[photos_df['group_size'] == 24, 'merge_allowed'] = False
        photos_df['original_context'] = photos_df['cluster_context'].copy()
        photos_df['groups_merged'] = 1
        photos_df = handle_wedding_bride_groom_merge(photos_df, logger)

        while True:
            # Build groups_to_change directly here
            if iteration >= max_iterations:
                logger.warning(f"Maximum iterations ({max_iterations}) reached in process_illegal_groups. Exiting to avoid infinite loop.")
                break

            photos_df, was_merge = process_wedding_merging(photos_df, resources, logger)
            if not was_merge:
                break

            iteration += 1

        # Resolve portrait singletons that survived merging
        photos_df = _resolve_singletons(
            photos_df, resources, manual_selection, logger, all_gallery_df
        )
    except Exception as ex:
        import traceback
        tb = traceback.extract_tb(ex.__traceback__)
        filename, lineno, func, text = tb[-1]
        logger.error(f"Groups management error: {str(ex)}. Exception in function: {func}, line {lineno}, file {filename}")
        return None, None

    groups = photos_df.groupby(['time_cluster', 'cluster_context', 'group_sub_index'])
    group2images = get_images_per_groups(groups)
    logger.info(f"Final number of groups for the album: {len(groups)}")
    logger.info(f"Final groups after illegal handling: {group2images}")
    return groups, group2images
