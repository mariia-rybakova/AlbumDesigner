from typing import List, Tuple, Iterable, Callable, Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

from utils.configs import CONFIGS


SIMILAR_CLASSES_L1 = [
    ['bride', 'bride getting dressed', 'getting hair-makeup', 'bride party'],
    ['bride', 'groom'],
    ['ceremony', 'walking the aisle'],
    ['food', 'settings', 'invite', 'detail'],
    ['dancing', 'entertainment'],
    ]
SIMILAR_CLASSES_L2 = [
    ['bride', 'bride getting dressed', 'getting hair-makeup', 'wedding dress', 'accessories'],
    ['bride', 'groom', 'bride and groom'],
    ['bride', 'bride party'],
    ['groom', 'groom party'],
    ['ceremony', 'walking the aisle', 'speech'],
    ['portrait', 'very large group', 'full party', 'large_portrait', 'small_portrait', 'couple'],
    ['accessories', 'food', 'settings', 'invite', 'detail', 'vehicle', 'inside vehicle', 'rings', 'suit'],
    ['groom', 'suit'],
    ['bride and groom', 'kiss', 'rings', 'first dance']
    ]

BRIDE_CENTRIC_CLASSES = [('bride', 'getting hair-makeup', 'bride getting dressed'), ('bride party',)]
GROOM_CENTRIC_CLASSES = [('groom', 'suit'), ('groom party',)]


def add_class_preference(illegal_group, selected_group, time_diff):
    """Modifies time difference based on content class pairs"""
    if illegal_group is None or selected_group is None:
        return time_diff

    illegal_group_key = illegal_group['cluster_context'].iloc[0]
    merge_target_key = selected_group['cluster_context'].iloc[0]
    if not all([illegal_group_key, merge_target_key]):
        return time_diff

    source_class = illegal_group_key.split('_')[0] if '_' in illegal_group_key else illegal_group_key
    target_class = merge_target_key.split('_')[0] if '_' in merge_target_key else merge_target_key

    multiplied = False
    # Prefer merging similar classes
    if source_class == target_class:
        time_diff *= 0.2
        multiplied = True

    # Prefer merging related classes
    if not multiplied:
        for similar_list in SIMILAR_CLASSES_L1:
            if source_class in similar_list and target_class in similar_list:
                time_diff *= 0.3
                multiplied = True
                break
    if not multiplied:
        for similar_list in SIMILAR_CLASSES_L2:
            if source_class in similar_list and target_class in similar_list:
                time_diff *= 0.5

    bride_centric_list = SIMILAR_CLASSES_L1[0]
    groom_centric_list = SIMILAR_CLASSES_L2[3]
    # Prefer not merging bride and groom classes with different size
    if (source_class in bride_centric_list and target_class in groom_centric_list or
        source_class in groom_centric_list and target_class in bride_centric_list):
        photos_diff = abs(illegal_group.shape[0] - selected_group.shape[0])
        time_diff *= (1 + photos_diff * 0.25)

    return time_diff


def merge_illegal_group_by_time(main_groups, illegal_group, general_times_list, max_images_per_spread=24):
    """
    Merge illegal group with the closest group by time that meets size requirements.

    Args:
        main_groups: List of DataFrame groups to potentially merge with
        illegal_group: DataFrame of the group to be merged

    Returns:
        tuple: (modified_illegal_group, combined_group, selected_cluster_content_index)
    """

    # Calculate mean time of the illegal group
    intended_group_time = illegal_group['general_time'].values.mean()

    # Calculate time range for the illegal group
    illegal_min_time = illegal_group['general_time'].min()
    illegal_max_time = illegal_group['general_time'].max()

    time_differences = []
    valid_groups = []
    long_distance_groups=[]
    long_time_differences = []
    for group in main_groups:
        # Calculate mean time and time range for the current group
        group_times = group['general_time'].values
        group_mean_time = group_times.mean()
        group_min_time = group_times.min()
        group_max_time = group_times.max()

        # Check if there are more than 2 images in between the groups
        images_in_between = sum(illegal_max_time < t < group_min_time or group_max_time < t < illegal_min_time
                                for t in general_times_list)
        if images_in_between > 2:
            min_time_diff = np.min(np.abs(group_times - intended_group_time))
            updated_time_diff = add_class_preference(illegal_group, group, min_time_diff)
            long_time_differences.append(updated_time_diff)
            long_distance_groups.append(group)

            continue  # Skip this group if more than 2 images are between the time ranges

        # Calculate the minimum time difference between the illegal group and this group
        min_time_diff = np.min(np.abs(group_times - intended_group_time))
        updated_time_diff = add_class_preference(illegal_group, group, min_time_diff)
        time_differences.append(updated_time_diff)
        valid_groups.append(group)

    # If no valid groups are found, return None
    if not valid_groups and long_distance_groups:
        valid_groups = long_distance_groups
        time_differences = long_time_differences
    elif not valid_groups and not long_distance_groups:
        return None, None
    # Sort by time differences and find the best group for merging
    time_differences = np.array(time_differences)
    sorted_indices = np.argsort(time_differences)

    for idx in sorted_indices:
        selected_cluster = valid_groups[idx]
        len_combine_group = len(selected_cluster) + len(illegal_group)

        # Check if the combination meets size requirements
        if len_combine_group <= max_images_per_spread:
            selected_time_difference = time_differences[idx]
            return selected_cluster, selected_time_difference

    # If no suitable group is found, return None
    return None, None


# Main logic
# Merge candidates
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
        (group['group_spreads'].iloc[0] + targets_df['group_spreads'] <= 2.1)
    ]


def _get_main_groups_bridegroom(merge_target_groups: Iterable[Tuple[Tuple[str, str, int], pd.DataFrame]],
                                group_key: Tuple[str, str, int], group: pd.DataFrame, cent_idx: int) -> List[pd.DataFrame]:
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
        group:
            Group to be merged
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


def count_contexts(group: pd.DataFrame):
    contexts = group['original_context'].copy()
    contexts = contexts.replace({'None': '*', 'other': '*'})
    return contexts.nunique()


def _get_main_groups_other(merge_target_groups: Iterable[Tuple[Tuple[str, str, int], pd.DataFrame]],
                           group_key: Tuple[str, str, int], group: pd.DataFrame,
                           possible_boxes_numbers: List[int]) -> List[pd.DataFrame]:
    """
    Retrieve merge target groups excluding the current group.

    This function filters out the group identified by `group_key` from the
    provided `merge_target_groups` and returns all other candidate groups.

    Args:
        merge_target_groups (Iterable[Tuple[Tuple[str, str, int], pd.DataFrame]]):
            An iterable of (group_key, group DataFrame) pairs representing potential merge targets.
        group_key (Tuple[str, str, int]):
            The key of the current group (time_cluster, cluster_context, group_sub_index).
        group:
            Group to be merged
        possible_boxes_numbers:
            List of numbers of photo boxes allowed per spread in at least one layout.

    Returns:
        List[pd.DataFrame]:
            A list of DataFrames representing groups that are valid merge candidates,
            excluding the one matching `group_key`.
    """
    return [
        m_group for m_key, m_group in merge_target_groups
        if (
                m_key != group_key
                and (len(group) + len(m_group) < 12 or len(group) + len(m_group) in possible_boxes_numbers)
                and count_contexts(pd.concat([group, m_group])) <= CONFIGS['merge_limit_times']
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
        main_groups = _get_main_groups(merge_target_groups, group_key, group, *args, **kwargs)
        selected_cluster, selected_time_difference = merge_illegal_group_by_time(main_groups, group,
                                                                                 general_times_list,
                                                                                 max_images_per_spread=CONFIGS['max_imges_per_spread'])

        if selected_cluster is not None:
            merge_candidates.append((group_key, selected_cluster, selected_time_difference))

    merge_candidates = sorted(merge_candidates, key=lambda x: x[2])
    return merge_candidates


# Convenience wrapper for filtering merge candidates in bride/groom groups
get_merge_candidates_bridegroom = lambda *args, **kwargs: _get_merge_candidates(_filter_merge_targets_bridegroom, _get_main_groups_bridegroom, *args, **kwargs)
# Wrapper for merge candidates using "other" filtering logic
get_merge_candidates_other = lambda *args, **kwargs: _get_merge_candidates(_filter_merge_targets_other, _get_main_groups_other, *args, **kwargs)


# Merge updates
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
        reminder_group_size = abs(len(to_merge_group) - len(selected_cluster))

        if reminder_group_size >= 2 or reminder_group_size == 0:
            min_len = min(len(to_merge_group), len(selected_cluster))
            merged_group = pd.concat([to_merge_group.head(min_len), selected_cluster.head(min_len)])
            reminder_group = pd.concat([to_merge_group.tail(len(to_merge_group) - min_len),
                                        selected_cluster.tail(len(selected_cluster) - min_len)])
            return merged_group, reminder_group
    return None, None


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
        photos_df.loc[row_index, 'groups_merged'] = count_contexts(merged_group)
        photos_df.loc[row_index, 'group_size'] = len(merged_group)
        photos_df.loc[row_index, 'group_sub_index'] = new_sub_index
        photos_df.loc[row_index, 'merge_allowed'] = False


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
        bigger_group = to_merge_group if len(to_merge_group) > len(selected_cluster) else selected_cluster
        photos_df.loc[row_index, 'cluster_context'] = bigger_group['cluster_context'].iloc[0]
        photos_df.loc[row_index, 'groups_merged'] = count_contexts(merged_group)
        photos_df.loc[row_index, 'group_size'] = len(merged_group)
        photos_df.loc[row_index, 'group_sub_index'] = bigger_group['group_sub_index'].iloc[0]
        if photos_df.loc[row_index, 'groups_merged'] >= CONFIGS['merge_limit_times']:
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
update_with_merges_bridegroom = lambda *args, **kwargs: _update_with_merges(_get_merged_group_bridegroom,
                                                                             _update_merged_photos_bridegroom, *args, **kwargs)
# Wrapper for applying merges using "other" merging logic
update_with_merges_other = lambda *args, **kwargs: _update_with_merges(_get_merged_group_other,
                                                                        _update_merged_photos_other, *args, **kwargs)

