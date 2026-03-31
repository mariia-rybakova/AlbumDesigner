from typing import List, Tuple, Iterable, Callable, Any, Optional

import pandas as pd

from src.core.models import AlbumDesignResources
from utils.album_tools import get_images_per_groups, get_missing_columns, split_groups
from src.groups_operations.merge import (get_merge_candidates_bridegroom, get_merge_candidates_other,
                                         update_with_merges_bridegroom, update_with_merges_other,
                                         BRIDE_CENTRIC_CLASSES, GROOM_CENTRIC_CLASSES)
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

    merge_candidates = get_merge_candidates_other(merge_groups, targets_df, general_times_list)

    if len(merge_candidates) == 0:
        return photos_df, False

    update_with_merges_other(photos_df, merge_groups, merge_candidates)
    return photos_df, True


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
                                   max_iterations=500):
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
