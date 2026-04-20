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
def get_groups_time(groups) -> Tuple[List[float], dict]:
    """
    Extract and sort photo timestamps from grouped photo data.

    Iterates over all groups, collects their 'general_time' values, and builds
    both a global sorted timeline and a per-group sorted time list.

    Args:
        groups: A pandas GroupBy object yielding (group_key, group DataFrame) pairs,
            where each DataFrame contains a 'general_time' column.

    Returns:
        A tuple of:
          - general_times_list: All photo times across all groups, sorted.
          - group_key2time_list: Dict mapping each group key to its sorted list of times.
    """
    general_times_list = list()
    group_key2time_list = dict()
    for group_key, group in groups:
        group_times = group['general_time'].values
        general_times_list.extend(group_times)
        group_key2time_list[group_key] = sorted(group_times)
    return sorted(general_times_list), group_key2time_list


def handle_wedding_splitting(photos_df: pd.DataFrame, resources: AlbumDesignResources, logger=None) -> pd.DataFrame:
    """
    Split oversized or temporally diverse photo groups into smaller subgroups.

    Iterates over groups whose size exceeds `CONFIGS['max_img_split']` and applies
    one of two splitting strategies:
      - **Size-based split** (`split_big_group`): used when the group occupies too many
        spreads, determined by `is_split_needed`.
      - **Time-based split** (`split_diverse_group`): used when the group spans
        disjoint time ranges with other photos in between.

    After splitting, all group sizes are recalculated.

    Args:
        photos_df: DataFrame of photos with columns including 'group_size',
            'time_cluster', 'cluster_context', 'group_sub_index', and 'general_time'.
        resources: Album design resources containing the look-up table that maps
            cluster contexts to recommended spread sizes.
        logger: Optional logger instance for recording warnings during updates.

    Returns:
        The updated photos DataFrame with split groups reflected in
        'group_sub_index' and 'group_size' columns.
    """
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
def handle_wedding_bride_groom_merge(photos_df: pd.DataFrame, logger=None) -> pd.DataFrame:
    """
    Merge complementary bride-centric and groom-centric photo groups.

    For each pairing defined in `BRIDE_CENTRIC_CLASSES` / `GROOM_CENTRIC_CLASSES`,
    identifies small groups (below `CONFIGS['max_img_split']`) that belong to
    bride or groom categories, finds the best merge partner by time proximity,
    and merges them while balancing group sizes.

    Args:
        photos_df: DataFrame of photos with columns including 'group_size',
            'time_cluster', 'cluster_context', 'group_sub_index', and 'general_time'.
        logger: Optional logger instance (currently unused).

    Returns:
        The updated photos DataFrame with bride/groom groups merged.
    """
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
def _update_group_spreads(photos_df: pd.DataFrame, look_up_table: dict) -> None:
    """
    Calculate group spread ratios and store them in a new 'group_spreads' column.

    For each row, divides the group's size by the recommended spread size from
    the look-up table. Groups whose cluster context is not in the table default
    to a ratio of 1.

    Args:
        photos_df: DataFrame of photos with 'cluster_context' and 'group_size' columns.
            Modified in place by adding a 'group_spreads' column.
        look_up_table: Dict mapping cluster context strings to lists where the
            first element is the recommended number of photos per spread.
    """
    def compute_spread(row: pd.Series) -> float:
        if row['cluster_context'] in look_up_table:
            return row['group_size'] / look_up_table[row['cluster_context']][0]
        return 1

    photos_df['group_spreads'] = photos_df.apply(compute_spread, axis=1)


def _filter_merge_candidate_photos(df_chunk: pd.DataFrame, size_limit: int) -> pd.DataFrame:
    """
    Filter photo groups eligible for merging.

    A group is eligible if:
      - Its size is below the split threshold or its spread ratio is < 1.
      - Merging is still allowed (`merge_allowed` is True).
      - The number of prior merges is below `size_limit`.

    Args:
        df_chunk: Subset of photos_df (special or regular), expected to contain
            'group_size', 'group_spreads', 'merge_allowed', and 'groups_merged' columns.
        size_limit: Maximum allowed number of cumulative merges for this subset.

    Returns:
        Filtered DataFrame containing only rows belonging to merge-eligible groups.
    """
    return df_chunk[
        ((df_chunk['group_size'] < CONFIGS['max_img_split']) | (df_chunk['group_spreads'] < 1))
        & (df_chunk['merge_allowed'] == True)
        & (df_chunk['groups_merged'] < size_limit)
    ]


def process_wedding_merging(photos_df: pd.DataFrame, resources: AlbumDesignResources, logger=None) -> Tuple[pd.DataFrame, bool]:
    """
    Run a single iteration of merging small photo groups.

    Splits candidates into special ('None'/'other') and regular groups, each with
    its own merge-count limit. Eligible groups are those below the split threshold
    or with a spread ratio < 1 that still have merge attempts remaining. The best
    merge partner is selected by time proximity, and groups are merged in place.

    Args:
        photos_df: DataFrame of photos, modified in place with updated
            'cluster_context', 'group_sub_index', 'group_size', 'groups_merged',
            and 'merge_allowed' columns.
        resources: Album design resources containing the look-up table for
            recommended spread sizes.
        logger: Optional logger instance (currently unused).

    Returns:
        A tuple of:
          - The updated photos DataFrame.
          - True if at least one merge was performed, False otherwise.
    """
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
    Attempt to add an unused photo from the full gallery to grow a singleton to size 2.

    Selection priority:
      1. Same time_cluster + same cluster_context, closest by general_time
      2. Same time_cluster + any class, closest by general_time

    Args:
        photos_df: The current album DataFrame (modified in-place).
        group_key: Key of the singleton group (time_cluster, cluster_context, group_sub_index).
        singleton_group: DataFrame containing the single photo.
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

    if 'time_cluster' not in unused_df.columns:
        return False

    singleton_time = singleton_group['general_time'].iloc[0]
    singleton_time_cluster = group_key[0]
    singleton_context = group_key[1]

    # Filter to same time_cluster
    same_cluster = unused_df[unused_df['time_cluster'] == singleton_time_cluster]
    if same_cluster.empty:
        return False

    # Try same class first, fallback to any class in same time_cluster
    same_class = same_cluster[same_cluster['cluster_context'] == singleton_context]
    pool = same_class if not same_class.empty else same_cluster

    # Pick closest by general_time
    candidate_idx = (pool['general_time'] - singleton_time).abs().idxmin()
    candidate = pool.loc[candidate_idx]

    # Build new row with all columns photos_df expects
    new_row = {}
    for col in photos_df.columns:
        if col in candidate.index:
            new_row[col] = candidate[col]

    # Override group-specific metadata to join the singleton's group
    new_row['time_cluster'] = singleton_time_cluster
    new_row['cluster_context'] = singleton_context
    new_row['group_sub_index'] = group_key[2]
    new_row['group_size'] = 2
    new_row['merge_allowed'] = False
    new_row['original_context'] = candidate.get('cluster_context', singleton_context)
    new_row['groups_merged'] = 1
    if 'group_spreads' in photos_df.columns:
        new_row['group_spreads'] = 1

    # Add row in-place and update singleton's group_size
    new_index = photos_df.index.max() + 1
    photos_df.loc[new_index] = new_row
    photos_df.loc[singleton_group.index, 'group_size'] = 2

    logger.info(f"Added unused photo {candidate.get('image_id', '?')} to singleton {group_key}")
    return True


def _get_singletons(groups):
    singletons = []
    for group_key, group in groups:
        if len(group) == 1:
            singletons.append((group_key, group))
    return singletons


def _process_singleton(photos_df, group_key, singleton_group, resources, manual_selection, possible_boxes_numbers,
                      logger, all_gallery_df=None):

    # Re-check: a prior merge in this loop may have altered this group
    # (e.g. another singleton was force-merged INTO this one, making it size > 1,
    #  or this photo was already moved to a different group).
    current_mask = (
            (photos_df['time_cluster'] == group_key[0]) &
            (photos_df['cluster_context'] == group_key[1]) &
            (photos_df['group_sub_index'] == group_key[2])
    )
    current_group = photos_df[current_mask]
    if len(current_group) == 0:
        logger.info(f"Singleton {group_key} already resolved by a prior merge")
        return
    if len(current_group) != 1:
        logger.info(f"Singleton {group_key} is now size {len(current_group)} after a prior merge, skipping")
        return
    singleton_group = current_group  # use fresh reference

    # Prefer force-merge (keeps original selection intact) over adding an unused photo
    merged = force_merge_portrait_singleton(
        photos_df, group_key, singleton_group, possible_boxes_numbers, logger
    )
    if merged:
        logger.info(f"Resolved singleton {group_key} by force-merge")
        return

    # Fallback for non-manual: add an unused photo from the gallery
    if not manual_selection and all_gallery_df is not None:
        if _try_add_unused_photo(photos_df, group_key, singleton_group, all_gallery_df, resources, logger):
            logger.info(f"Resolved singleton {group_key} by adding unused photo")
            return

    logger.warning(f"Could not resolve singleton {group_key}. No valid merge target.")


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
    singletons = _get_singletons(groups)

    if not singletons:
        logger.info("No singletons found to resolve")
        return photos_df

    singletons.sort(key=lambda x: x[1]['general_time'].iloc[0])
    logger.info(f"Found {len(singletons)} singleton group(s) to resolve: {[s[0] for s in singletons]}")

    for group_key, singleton_group in singletons:
        _process_singleton(photos_df, group_key, singleton_group, resources, manual_selection, possible_boxes_numbers,
                          logger, all_gallery_df)

    return photos_df


# Illegal groups split/merge pipeline
def _get_groups(photos_df: pd.DataFrame, manual_selection: bool, logger) -> pd.DataFrame:
    """
    Initialize photo groups by assigning 'group_sub_index' and 'group_size' columns.

    In automatic mode, splits photos into special and regular subsets using
    `split_groups`, then assigns sub-indices and sizes to special groups
    individually. In manual mode, all photos are treated as regular groups.
    Regular group sizes are computed by ('time_cluster', 'cluster_context').

    Args:
        photos_df: DataFrame of photos with 'time_cluster', 'cluster_context',
            and 'cluster_label' columns.
        manual_selection: If True, skips special-group splitting and treats all
            photos as regular groups.
        logger: Logger instance used for column validation warnings.

    Returns:
        A new DataFrame with 'group_sub_index' and 'group_size' populated,
        combining special and regular groups.

    Raises:
        ValueError: If required columns are missing from photos_df.
    """
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


def process_wedding_illegal_groups(
        photos_df: pd.DataFrame, resources: AlbumDesignResources, manual_selection: bool,
        logger=None, max_iterations: int = 500, all_gallery_df=None
    ) -> Tuple[Optional[Any], Optional[dict]]:
    """
    Full pipeline for splitting and merging photo groups into legal album spreads.

    Executes the following steps in order:
      1. Initialize groups from the photos DataFrame.
      2. Split oversized or temporally diverse groups.
      3. Merge complementary bride/groom groups.
      4. Iteratively merge remaining small groups until no more merges are possible
         or `max_iterations` is reached.

    Args:
        photos_df: DataFrame of photos with classification and time columns.
        resources: Album design resources containing the look-up table for
            recommended spread sizes.
        manual_selection: If True, skips special-group splitting and treats all
            photos as regular groups.
        logger: Optional logger instance for info/warning/error messages.
        max_iterations: Safety limit for the iterative merge loop to prevent
            infinite execution.

    Returns:
        A tuple of:
          - groups: A pandas GroupBy object of the final photo groups, or None on error.
          - group2images: Dict mapping group keys to their image lists, or None on error.
    """
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
        return None, None, None

    groups = photos_df.groupby(['time_cluster', 'cluster_context', 'group_sub_index'])
    group2images = get_images_per_groups(groups)
    logger.info(f"Final number of groups for the album: {len(groups)}")
    logger.info(f"Final groups after illegal handling: {group2images}")
    return groups, group2images, photos_df
