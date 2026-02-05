from typing import List, Tuple, Iterable, Callable, Any, Optional

import pandas as pd

from src.core.models import AlbumDesignResources
from utils.album_tools import get_images_per_groups, get_missing_columns, split_groups
from src.groups_operations.groups_splitting_merging import merge_illegal_group_by_time, split_illegal_group_by_time, split_illegal_group_in_certain_point
from utils.configs import CONFIGS


BRIDE_CENTRIC_CLASSES = [('bride', 'getting hair-makeup', 'bride getting dressed'), ('bride party',)]
GROOM_CENTRIC_CLASSES = [('groom', 'suit'), ('groom party',)]


def get_groups_time(groups):
    general_times_list = list()
    group_key2time_list = dict()
    for group_key, group in groups:
        group_times = group['general_time'].values
        general_times_list.extend(group_times)
        group_key2time_list[group_key] = sorted(group_times)
    return sorted(general_times_list), group_key2time_list


def check_time_based_split_needed(general_times_list, group_time_list, group_key):
    if len(group_time_list) < 2:
        return False, None
    if group_key not in ['walking the aisle', 'bride', 'groom', 'bride and groom', 'groom party', 'bride party', 'portrait']:
        return False, None

    split_points = list()
    for i in range(len(group_time_list) - 1):
        start_time = group_time_list[i]
        end_time = group_time_list[i + 1]

        count_between = sum(start_time < t < end_time for t in general_times_list)

        if count_between > 2:
            split_points.append(start_time)

    if len(split_points) > 0:
        return True, split_points

    return False, None


def _get_splitting_score(group: pd.DataFrame, group_spread_size: int) -> int:
    """
    Calculate the splitting score for a photo group.

    Args:
        group (DataFrame): The group of photos.
        group_spread_size (int): Recommended number of photos per spread for this group.

    Returns:
        int: The splitting score (rounded), or 0 if spread size is invalid.
    """
    if group_spread_size > 0:
        return round(group['group_size'].iloc[0] / group_spread_size)
    return 0


def _is_split_needed(splitting_score: int, group_spread_size: int, group_key: Tuple[str, str, int]) -> bool:
    """
    Determine whether a photo group should be split into subgroups.

    A split is considered necessary if:
      - The splitting score exceeds the minimum (`CONFIGS['min_split_score']`).
      - The splitting score equals the minimum and the group spread size > 5.
      - The splitting score equals 2 and the group spread size >= 12.
      - The group spread size >= 24.
    Additionally, groups with 'cant_split' in their cluster_context are excluded.

    Args:
        splitting_score (int):
            Score indicating how strongly the group should be split.
        group_spread_size (int):
            Recommended number of photos per spread for this group.
        group_key (Tuple[str, str, int]):
            Key identifying the group (time_cluster, cluster_context, group_sub_index).

    Returns:
        bool:
            True if the group should be split, False otherwise.
    """
    return (
            (
                    splitting_score > CONFIGS['min_split_score']
                    or (splitting_score == CONFIGS['min_split_score'] and group_spread_size > 5)
                    or (splitting_score == 2 and group_spread_size >= 12)
                    or group_spread_size >= 24
            )
            and 'cant_split' not in group_key[1]
    )


def _update_group_sub_index(photos_df: pd.DataFrame, updated_group: pd.DataFrame, logger) -> None:
    """
    Update the `group_sub_index` field in original DataFrame for rows from an updated group.

    The sub_index is parsed from the second-to-last part of the group's
    `cluster_context` string (split by underscores). If parsing fails,
    a warning is logged.

    Args:
        photos_df (pd.DataFrame):
            The full DataFrame of photos to update.
        updated_group (pd.DataFrame):
            A DataFrame with the group whose sub_index needs updating.
        logger (logging.Logger):
            Logger instance used to record warnings.

    Returns:
        None: Updates are applied directly to `photos_df`.
    """
    if updated_group is not None:
        for row_index in updated_group.index:
            # Use the last part of the cluster_context as the sub_index
            try:
                context_parts = updated_group.loc[row_index, 'cluster_context'].split('_')
                sub_index = int(context_parts[-2])
                photos_df.loc[row_index, 'group_sub_index'] = sub_index
            except (IndexError, ValueError):
                logger.warning(
                    f"Could not parse sub_index from context: {updated_group.loc[row_index, 'cluster_context']}")


def _update_groups_size(photos_df: pd.DataFrame,
                        clusters: List[str] = ['time_cluster', 'cluster_context', 'group_sub_index']) -> None:
    """
    Recalculate the `group_size` field after splitting groups into subgroups.

    Groups are defined by the specified cluster keys. The size of each group
    is recalculated and updated in the DataFrame.

    Args:
        photos_df (pd.DataFrame):
            The full DataFrame of photos to update.
        clusters (List[str], optional):
            List of column names used to group the DataFrame.
            Defaults to ['time_cluster', 'cluster_context', 'group_sub_index'].

    Returns:
        None: Updates are applied directly to `photos_df`.
    """
    photo_groups = photos_df.groupby(clusters)
    for group_key, group in photo_groups:
        group_size = len(group)
        photos_df.loc[group.index, 'group_size'] = group_size


def handle_wedding_splitting(photos_df, resources: AlbumDesignResources, logger=None):
    # handle splitting
    look_up_table = resources.look_up_table.table if hasattr(resources, 'look_up_table') else {}
    split_df = photos_df[photos_df['group_size'] >= CONFIGS['max_img_split']]
    split_groups_ = split_df.groupby(['time_cluster', 'cluster_context', 'group_sub_index'])
    general_times_list, group_key2time_list = get_groups_time(split_groups_)

    count = 0
    for group_key, group in split_groups_:
        group_spread_size = look_up_table.get(group_key[1], [10])[0]
        splitting_score = _get_splitting_score(group, group_spread_size)

        if _is_split_needed(splitting_score, group_spread_size, group_key):
            updated_group, labels_count = split_illegal_group_by_time(group, group_spread_size, count)
            count += 1
        else:
            if_split, split_points = check_time_based_split_needed(general_times_list, group_key2time_list[group_key],
                                                                   group_key=group_key[1])
            if if_split:
                updated_group, labels_count = split_illegal_group_in_certain_point(group, split_points, count)
                count += 1
            else:
                updated_group = None

        _update_group_sub_index(photos_df, updated_group, logger)

    _update_groups_size(photos_df)
    return photos_df


def _filter_merge_targets_bridegroom(targets_df: pd.DataFrame, group: pd.DataFrame, group_key) -> pd.DataFrame:
    """
    Filter potential merge targets for a bride/groom group.

    This function selects candidate groups from `targets_df` that:
        - Belong to the same time cluster as the current group.
        - Can be merged with the current group without exceeding the maximum
        number of images allowed per album spread (`CONFIGS['max_imges_per_spread']`).

    Args:
        targets_df (pd.DataFrame): DataFrame containing candidate groups for merging.
        group (pd.DataFrame): The current bride/groom group being considered for merging.
        group_key (tuple): Key identifying the group (time_cluster, cluster_context, group_sub_index).

    Returns:
        pd.DataFrame: A filtered DataFrame of merge target groups that meet the criteria.
    """
    return targets_df[(targets_df['time_cluster'] == group_key[0]) &
                      (targets_df['group_size'] + len(group) <= CONFIGS['max_imges_per_spread'])]


def _get_main_groups_bridegroom(merge_target_groups: Iterable[Tuple[Tuple[str, str, int], pd.DataFrame]],
                                group_key: Tuple[str, str, int], cent_idx: int) -> List[pd.DataFrame]:
    """
    Filter merge target groups to find valid bride/groom pairs.

    This function selects candidate groups from `merge_target_groups` that:
      - Are not the same as the current group (`group_key`).
      - Belong to complementary bride/groom class pairs defined by
        `BRIDE_CENTRIC_CLASSES[cent_idx]` and `GROOM_CENTRIC_CLASSES[cent_idx]`.

    Args:
        merge_target_groups (Iterable[Tuple[Tuple[str, str, int], pd.DataFrame]]):
            An iterable of (group_key, group DataFrame) pairs representing potential merge targets.
        group_key (Tuple[str, str, int]):
            The key of the current group (time_cluster, cluster_context, group_sub_index).
        cent_idx (int):
            Index pointing to the bride/groom class pairing to check against.

    Returns:
        List[pd.DataFrame]:
            A list of DataFrames representing groups that are valid bride/groom merge candidates.
    """

    return [
        m_group for m_key, m_group in merge_target_groups
        if (
                m_key != group_key and
                (
                    (group_key[1] in BRIDE_CENTRIC_CLASSES[cent_idx] and m_key[1] in GROOM_CENTRIC_CLASSES[cent_idx])
                    or
                    (group_key[1] in GROOM_CENTRIC_CLASSES[cent_idx] and m_key[1] in BRIDE_CENTRIC_CLASSES[cent_idx])
                )
        )
    ]


def _get_merge_candidates(
        _filter_merge_targets: Callable[[pd.DataFrame, pd.DataFrame, Tuple[str, str, int]], pd.DataFrame],
        _get_main_groups: Callable[[Iterable[Tuple[Tuple[str, str, int], pd.DataFrame]], Tuple[str, str, int], Any], List[pd.DataFrame]],
        merge_groups: Iterable[Tuple[Tuple[str, str, int], pd.DataFrame]],
        targets_df: pd.DataFrame,
        general_times_list: List[float],
        *args,
        **kwargs
    ) -> List[Tuple[Tuple[str, str, int], pd.DataFrame, float]]:
    """
    Identify merge candidates for photo groups based on time proximity and filtering rules.

    This function iterates through groups in `merge_groups`, applies a filtering function
    to find potential merge targets, and then uses a main group selection function to
    determine valid candidates. It evaluates time differences via `merge_illegal_group_by_time`
    and returns a sorted list of merge candidates.

    Args:
        _filter_merge_targets (Callable):
            Function to filter potential merge targets. Must accept (targets_df, group, group_key).
        _get_main_groups (Callable):
            Function to select main groups from merge_target_groups. Must accept
            (merge_target_groups, group_key, *args, **kwargs).
        merge_groups (Iterable[Tuple[Tuple[str, str, int], pd.DataFrame]]):
            Iterable of (group_key, group DataFrame) pairs representing groups to be merged.
        targets_df (pd.DataFrame):
            DataFrame containing candidate groups for merging.
        general_times_list (List[float]):
            List of all photo times used to calculate temporal differences.
        *args:
            Additional positional arguments passed to `_get_main_groups`.
        **kwargs:
            Additional keyword arguments passed to `_get_main_groups`.

    Returns:
        List[Tuple[Tuple[str, str, int], pd.DataFrame, float]]:
            A sorted list of merge candidates, where each tuple contains:
              - group_key: The key of the current group.
              - selected_cluster: The chosen partner group DataFrame.
              - selected_time_difference: The time difference used for sorting.
    """
    merge_candidates = list()

    for group_key, group in merge_groups:
        merge_targets = _filter_merge_targets(targets_df, group, group_key)
        merge_target_groups = merge_targets.groupby(['time_cluster', 'cluster_context', 'group_sub_index'])
        main_groups = _get_main_groups(merge_target_groups, group_key, *args, **kwargs)
        selected_cluster, selected_time_difference = merge_illegal_group_by_time(main_groups, group,
                                                                                 general_times_list,
                                                                                 max_images_per_spread=CONFIGS['max_imges_per_spread'])

        if selected_cluster is not None:
            merge_candidates.append((group_key, selected_cluster, selected_time_difference))

    merge_candidates = sorted(merge_candidates, key=lambda x: x[2])
    return merge_candidates


# Convenience wrapper for filtering merge candidates in bride/groom groups
_get_merge_candidates_bridegroom = lambda *args, **kwargs: _get_merge_candidates(_filter_merge_targets_bridegroom, _get_main_groups_bridegroom, *args, **kwargs)


def _is_bride_groom_pair(group_key: Tuple[str, str, int], selected_cluster: pd.DataFrame, cent_idx: int) -> bool:
    """
    Check if the given group and selected cluster form a valid bride/groom pair.

    Args:
        group_key (tuple): Key of the current group (time_cluster, cluster_context, group_sub_index).
        selected_cluster (DataFrame): Candidate group to merge with.
        cent_idx (int): Index pointing to the bride/groom class pairing.

    Returns:
        bool: True if the groups are opposite bride/groom classes, False otherwise.
    """
    cluster_context = selected_cluster['cluster_context'].iloc[0]

    bride_condition = (
        group_key[1] in BRIDE_CENTRIC_CLASSES[cent_idx]
        and cluster_context in GROOM_CENTRIC_CLASSES[cent_idx]
    )

    groom_condition = (
        group_key[1] in GROOM_CENTRIC_CLASSES[cent_idx]
        and cluster_context in BRIDE_CENTRIC_CLASSES[cent_idx]
    )

    return bride_condition or groom_condition


def _get_merged_group_bridegroom(to_merge_group: pd.DataFrame, selected_cluster: pd.DataFrame,
                                 group_key: Tuple[str, str, int], cent_idx: int
                                 ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Attempt to merge a bride/groom group with its selected partner.

    This function checks if the given group and selected cluster form a valid
    bride/groom pair. If they do, and their sizes differ by at least 2, it
    merges the first `min_len` rows of each group into a new merged group and
    keeps the remaining rows in a reminder group.

    Args:
        to_merge_group (pd.DataFrame):
            The current group being considered for merging.
        selected_cluster (pd.DataFrame):
            The candidate group to merge with.
        group_key (Tuple[str, str, int]):
            Key identifying the current group (time_cluster, cluster_context, group_sub_index).
        cent_idx (int):
            Index pointing to the bride/groom class pairing.

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
            - merged_group: DataFrame containing merged rows from both groups.
            - reminder_group: DataFrame containing leftover rows.
            Returns (None, None) if no valid merge is possible.
    """
    if _is_bride_groom_pair(group_key, selected_cluster, cent_idx):
        if abs(len(to_merge_group) - len(selected_cluster)) >= 2:
            min_len = min(len(to_merge_group), len(selected_cluster))
            merged_group = pd.concat([to_merge_group.head(min_len), selected_cluster.head(min_len)])
            reminder_group = pd.concat([to_merge_group.tail(len(to_merge_group) - min_len),
                                        selected_cluster.tail(len(selected_cluster) - min_len)])
            return merged_group, reminder_group
    return None, None


def _update_merged_photos_bridegroom(photos_df: pd.DataFrame, to_merge_group: pd.DataFrame, selected_cluster: pd.DataFrame,
                                     merged_group: pd.DataFrame, reminder_group: pd.DataFrame) -> None:
    """
    Update the photo DataFrame after merging bride/groom groups.

    This function updates metadata for both the merged group and the reminder group:
      - Reminder group: updates `group_size`.
      - Merged group: updates `cluster_context`, `groups_merged`, `group_size`,
        assigns a new `group_sub_index`, and sets `merge_allowed` to False.

    Args:
        photos_df (pd.DataFrame):
            The full DataFrame of photos to update.
        to_merge_group (pd.DataFrame):
            The original group being merged.
        selected_cluster (pd.DataFrame):
            The partner group used in the merge.
        merged_group (pd.DataFrame):
            The resulting merged group.
        reminder_group (pd.DataFrame):
            The leftover group after merging.

    Returns:
        None: Updates are applied directly to `photos_df`.
    """
    # Update reminder group photos
    for row_index in reminder_group.index:
        photos_df.loc[row_index, 'group_size'] = len(reminder_group)
    # Update merged group photos
    new_sub_index = photos_df['group_sub_index'].max() + 1
    for row_index in merged_group.index:
        photos_df.loc[row_index, 'cluster_context'] = selected_cluster['cluster_context'].iloc[0]
        photos_df.loc[row_index, 'groups_merged'] = to_merge_group['groups_merged'].iloc[0] + \
                                                    selected_cluster['groups_merged'].iloc[0]
        photos_df.loc[row_index, 'group_size'] = len(merged_group)
        photos_df.loc[row_index, 'group_sub_index'] = new_sub_index
        photos_df.loc[row_index, 'merge_allowed'] = False


def _update_with_merges(
        _get_merged_group: Callable[..., Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]],
        _update_merged_photos: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], None],
        photos_df: pd.DataFrame,
        merge_groups: Any,
        merge_candidates: List[Tuple[Tuple[str, str, int], pd.DataFrame, float]],
        *args,
        **kwargs
    ) -> None:
    """
    Apply merges to photo groups based on merge candidates.

    This function iterates through merge candidates, retrieves the corresponding
    groups, checks for duplicates, and applies merging logic. It updates the
    main DataFrame using the provided helper functions.

    Args:
        _get_merged_group (Callable):
            Function that attempts to merge two groups and returns (merged_group, reminder_group).
        _update_merged_photos (Callable):
            Function that updates the DataFrame after a merge.
        photos_df (pd.DataFrame):
            The full DataFrame of photos to update.
        merge_groups (pandas.core.groupby.generic.DataFrameGroupBy):
            Grouped DataFrame object (e.g., from `groupby`) containing groups to merge.
        merge_candidates (List[Tuple[Tuple[str, str, int], pd.DataFrame, float]]):
            List of merge candidates, each containing:
              - group_key: The key of the group to merge.
              - selected_cluster: The partner group DataFrame.
              - selected_time_difference: The time difference used for sorting.
        *args:
            Additional positional arguments passed to `_get_merged_group`.
        **kwargs:
            Additional keyword arguments passed to `_get_merged_group`.

    Returns:
        None: Updates are applied directly to `photos_df`.
    """
    current_merges = set()
    for group_key, selected_cluster, selected_time_difference in merge_candidates:
        to_merge_group = merge_groups.get_group(group_key)
        selected_key = (selected_cluster['time_cluster'].iloc[0], selected_cluster['cluster_context'].iloc[0],
                        selected_cluster['group_sub_index'].iloc[0])

        if group_key in current_merges or selected_key in current_merges:
            continue

        merged_group, reminder_group = _get_merged_group(to_merge_group, selected_cluster, group_key, *args, **kwargs)
        if merged_group is None:
            continue

        # Update df
        _update_merged_photos(photos_df, to_merge_group, selected_cluster, merged_group, reminder_group)

        current_merges.add(group_key)
        current_merges.add(selected_key)


# Convenience wrapper for bride/groom merges
_update_with_merges_bridegroom = lambda *args, **kwargs: _update_with_merges(_get_merged_group_bridegroom,
                                                                             _update_merged_photos_bridegroom, *args, **kwargs)


def handle_wedding_bride_groom_merge(photos_df, logger=None):
    merge_df = photos_df[(photos_df['group_size'] < CONFIGS['max_img_split']) &
                         ((photos_df['cluster_context'].isin(BRIDE_CENTRIC_CLASSES)) |
                          (photos_df['cluster_context'].isin(GROOM_CENTRIC_CLASSES)))]
    targets_df = photos_df.copy()

    merge_groups = merge_df.groupby(['time_cluster', 'cluster_context', 'group_sub_index'])
    general_times_list, _ = get_groups_time(photos_df.groupby(['time_cluster', 'cluster_context', 'group_sub_index']))

    for cent_idx in range(len(BRIDE_CENTRIC_CLASSES)):
        merge_candidates = _get_merge_candidates_bridegroom(merge_groups, targets_df, general_times_list, cent_idx=cent_idx)

        _update_with_merges_bridegroom(photos_df, merge_groups, merge_candidates, cent_idx)

    return photos_df


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


def _filter_merge_targets_other(targets_df: pd.DataFrame, group: pd.DataFrame, group_key: Tuple[str, str, int]) -> pd.DataFrame:
    """
    Find potential merge targets for a given group.

    Args:
        targets_df (pd.DataFrame): Candidate groups DataFrame.
        group (pd.DataFrame): The group being considered for merging.
        group_key (tuple): Key of the group (time_cluster, cluster_context, group_sub_index).

    Returns:
        pd.DataFrame: Filtered DataFrame of merge targets.
    """
    return targets_df[
        (targets_df['time_cluster'] == group_key[0]) &
        (targets_df['group_size'] + len(group) <= CONFIGS['max_imges_per_spread']) &
        (group['groups_merged'].iloc[0] + targets_df['groups_merged'] <= CONFIGS['merge_limit_times']) &
        (group['group_spreads'].iloc[0] + targets_df['group_spreads'] <= 2.1)
    ]


def _get_main_groups_other(merge_target_groups: Iterable[Tuple[Tuple[str, str, int], pd.DataFrame]],
                           group_key: Tuple[str, str, int]) -> List[pd.DataFrame]:
    """
    Retrieve merge target groups excluding the current group.

    This function filters out the group identified by `group_key` from the
    provided `merge_target_groups` and returns all other candidate groups.

    Args:
        merge_target_groups (Iterable[Tuple[Tuple[str, str, int], pd.DataFrame]]):
            An iterable of (group_key, group DataFrame) pairs representing potential merge targets.
        group_key (Tuple[str, str, int]):
            The key of the current group (time_cluster, cluster_context, group_sub_index).

    Returns:
        List[pd.DataFrame]:
            A list of DataFrames representing groups that are valid merge candidates,
            excluding the one matching `group_key`.
    """
    return [m_group for m_key, m_group in merge_target_groups if m_key != group_key]


# Wrapper for merge candidates using "other" filtering logic
_get_merge_candidates_other = lambda *args, **kwargs: _get_merge_candidates(_filter_merge_targets_other, _get_main_groups_other, *args, **kwargs)


def _get_merged_group_other(to_merge_group: pd.DataFrame, selected_cluster: pd.DataFrame, *args, **kwargs
                            ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Merge two non-bride/groom groups into a single group.

    Unlike bride/groom merging, this function simply concatenates
    the two groups without balancing their sizes. No reminder group
    is created.

    Args:
        to_merge_group (pd.DataFrame):
            The current group being considered for merging.
        selected_cluster (pd.DataFrame):
            The candidate group to merge with.
        *args:
            Additional positional arguments (unused).
        **kwargs:
            Additional keyword arguments (unused).

    Returns:
        Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
            - merged_group: DataFrame containing all rows from both groups.
            - None: No reminder group is produced in this case.
    """
    merged_group = pd.concat([to_merge_group, selected_cluster])
    return merged_group, None


def _update_merged_photos_other(photos_df: pd.DataFrame, to_merge_group: pd.DataFrame, selected_cluster: pd.DataFrame,
                                merged_group: pd.DataFrame, *args, **kwargs):
    """
    Update the photo DataFrame after merging non-bride/groom groups.

    This function updates metadata for all rows in the merged group:
      - Sets `cluster_context` to that of the selected cluster.
      - Updates `groups_merged` as the sum of both groups.
      - Updates `group_size` to the size of the merged group.
      - Sets `group_sub_index` to that of the selected cluster.
      - Disables further merging (`merge_allowed = False`) if
        the merge limit is reached.

    Args:
        photos_df (pd.DataFrame):
            The full DataFrame of photos to update.
        to_merge_group (pd.DataFrame):
            The original group being merged.
        selected_cluster (pd.DataFrame):
            The partner group used in the merge.
        merged_group (pd.DataFrame):
            The resulting merged group.
        *args:
            Additional positional arguments (unused).
        **kwargs:
            Additional keyword arguments (unused).

    Returns:
        None: Updates are applied directly to `photos_df`.
    """
    for row_index in merged_group.index:
        photos_df.loc[row_index, 'cluster_context'] = selected_cluster['cluster_context'].iloc[0]
        photos_df.loc[row_index, 'groups_merged'] = to_merge_group['groups_merged'].iloc[0] + \
                                                    selected_cluster['groups_merged'].iloc[0]
        photos_df.loc[row_index, 'group_size'] = len(merged_group)
        photos_df.loc[row_index, 'group_sub_index'] = selected_cluster['group_sub_index'].iloc[0]
        if photos_df.loc[row_index, 'groups_merged'] >= CONFIGS['merge_limit_times']:
            photos_df.loc[row_index, 'merge_allowed'] = False


# Wrapper for applying merges using "other" merging logic
_update_with_merges_other = lambda *args, **kwargs: _update_with_merges(_get_merged_group_other,
                                                                        _update_merged_photos_other, *args, **kwargs)


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

    merge_candidates = _get_merge_candidates_other(merge_groups, targets_df, general_times_list)

    if len(merge_candidates) == 0:
        return photos_df, False

    _update_with_merges_other(photos_df, merge_groups, merge_candidates)
    return photos_df, True


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
    _update_groups_size(df_regular, clusters=['time_cluster', 'cluster_context'])

    photos_df = pd.concat([df_special, df_regular], ignore_index=True)
    return photos_df


def process_wedding_illegal_groups(photos_df, resources: AlbumDesignResources, manual_selection, logger=None,
                                   max_iterations=500):
    photos_df = _get_groups(photos_df, manual_selection, logger)

    iteration = 0
    try:
        photos_df = handle_wedding_splitting(photos_df, resources, logger)

        photos_df['merge_allowed'] = True
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
