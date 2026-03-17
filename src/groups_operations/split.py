from typing import List, Tuple, Iterable, Callable, Any, Optional

import numpy as np
import pandas as pd
from k_means_constrained import KMeansConstrained
from sklearn.metrics import silhouette_score

from utils.configs import CONFIGS


# split diverse group
def split_diverse_group(illegal_group: Optional[pd.DataFrame],
                        split_points: Optional[List[float]]) -> Optional[pd.DataFrame]:
    """
    Split a group into subgroups at temporal split points.

    Assigns incremental 'group_sub_index' values based on which time segment
    each photo falls into. Segments are defined by the provided split points.

    Args:
        illegal_group: DataFrame of photos to split, with a 'general_time' column.
            Modified in place. May be None or empty.
        split_points: Sorted list of time values at which to split the group.
            May be None if no split is needed.

    Returns:
        The updated DataFrame with new 'group_sub_index' values, or None if
        the group is empty/None or no split points are provided.
    """
    if illegal_group is None or illegal_group.empty or split_points is None:
        return None

    for i, start_time in enumerate([-1] + split_points):
        illegal_group.loc[(illegal_group['general_time'] > start_time), 'group_sub_index'] = i

    return illegal_group


# split big group
def get_large_spread_split_size(single_spread_size: int) -> int:
    """
    Choose the nearest valid large-spread size (16 or 24) for splitting.

    Picks whichever of 16 or 24 is closer to `single_spread_size`.

    Args:
        single_spread_size: The recommended number of photos per spread.

    Returns:
        16 or 24, whichever is closer to the input.
    """
    if 24 - single_spread_size < single_spread_size - 16:
        return 24
    else:
        return 16


def get_small_spread_split_size(single_spread_size: int) -> int:
    """
    Determine the split chunk size for small spreads.

    For spreads with 12+ photos, uses the spread size directly. For smaller
    spreads, multiplies by `CONFIGS['min_split_score']` to ensure enough
    photos per subgroup.

    Args:
        single_spread_size: The recommended number of photos per spread.

    Returns:
        The number of photos per split chunk.
    """
    if single_spread_size >= 12:
        return single_spread_size
    else:
        # Number of photos for allowed amount of spreads
        return single_spread_size * CONFIGS['min_split_score']


def _split_by_chunk_size(illegal_group: pd.DataFrame, split_size: int, n_samples: int) -> Optional[pd.DataFrame]:
    """
    Split a group into fixed-size chunks ordered by image importance and time.

    Sorts the group by 'image_as' (descending) and 'general_time' (ascending),
    then divides it into consecutive chunks of `split_size` photos, assigning
    each chunk a unique 'group_sub_index'.

    Args:
        illegal_group: DataFrame of photos to split.
        split_size: Number of photos per chunk.
        n_samples: Total number of photos in the group.

    Returns:
        A DataFrame with updated 'group_sub_index' values, or None if
        `split_size` is invalid or >= `n_samples`.
    """
    # If the split size is invalid or larger than the group, don't split
    if split_size <= 0 or split_size >= n_samples:
        return None

    # Create chunks of size `split_size`
    illegal_group = illegal_group.sort_values(['image_as', 'general_time'], ascending=[False, True])
    chunks = [illegal_group.iloc[i:i + split_size] for i in range(0, n_samples, split_size)]

    # Assign unique index to each chunk
    for i, chunk in enumerate(chunks):
        chunk.loc[:, 'group_sub_index'] = i

    # Combine all chunks into a single DataFrame
    updated_group = pd.concat(chunks)
    return updated_group


def _clusterize(illegal_group: pd.DataFrame, feature: np.ndarray, n_clusters: int,
                size_min: int, size_max: int, silhouette: bool = False) -> Optional[np.ndarray]:
    """
    Apply constrained K-Means clustering and assign labels to the group.

    Clusters the provided features using `KMeansConstrained` with size
    constraints. Optionally validates clustering quality via silhouette score,
    returning None if the score is below 0.15.

    Args:
        illegal_group: DataFrame to update in place with a 'group_sub_index' column.
        feature: 2D array of features to cluster on (e.g. time values reshaped to (-1, 1)).
        n_clusters: Number of clusters to create.
        size_min: Minimum number of samples per cluster.
        size_max: Maximum number of samples per cluster.
        silhouette: If True, validate clustering quality and return None
            if the silhouette score is below 0.15.

    Returns:
        Array of cluster labels, or None if silhouette validation fails.
    """
    # Apply constrained K-Means clustering
    clf = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=size_min,
        size_max=size_max,
        random_state=0
    )
    labels = clf.fit_predict(feature)

    # Estimate clustering quality
    if silhouette:
        silhouette_avg = silhouette_score(feature, labels)
        if silhouette_avg < 0.15:
            return None

    # Assign label values as subindex to the DataFrame
    illegal_group.loc[:, 'group_sub_index'] = labels
    return labels


def _split_with_time_clustering(illegal_group: pd.DataFrame, n_clusters: int,
                                size_min: int, size_max: int) -> Optional[pd.DataFrame]:
    """
    Split a group into temporally ordered subgroups using constrained K-Means.

    Clusters photos by their 'general_time' values, then remaps cluster labels
    so that earlier time clusters receive lower 'group_sub_index' values.

    Args:
        illegal_group: DataFrame of photos to split, with a 'general_time' column.
            Modified in place.
        n_clusters: Number of clusters to create.
        size_min: Minimum number of photos per cluster.
        size_max: Maximum number of photos per cluster.

    Returns:
        The updated DataFrame with temporally ordered 'group_sub_index' values,
        or None if clustering fails.
    """
    try:
        # Get time features and normalize them
        time_features = illegal_group["general_time"].values.reshape(-1, 1)

        # Apply constrained K-Means clustering on time
        _ = _clusterize(illegal_group, time_features, n_clusters, size_min, size_max,
                                 silhouette = False)

        # Sort groups by mean time for temporal ordering
        mean_times = illegal_group.groupby('group_sub_index')['general_time'].mean()
        sorted_clusters = mean_times.sort_values().index

        # Rename labels to ensure earlier time group gets a lower number
        mapping = {i: j for j, i in enumerate(sorted_clusters)}
        illegal_group.loc[:, 'group_sub_index'] = illegal_group['group_sub_index'].map(mapping)

        return illegal_group

    except Exception as e:
        print(f"Error during temporal splitting: {str(e)}")
        return None


def split_big_group(illegal_group: pd.DataFrame, single_spread_size: int) -> Optional[pd.DataFrame]:
    """
    Split an illegal group into subgroups based on time or size.

    Args:
        illegal_group (DataFrame): The group to be split.
        single_spread_size (int): Threshold for determining split sizes.

    Returns:
        Modified DataFrame with updated 'group_sub_index', or None if splitting fails.
    """
    # Calculate `max_size_splits` based on `single_spread_size`
    n_samples = len(illegal_group)

    # Split large spread group by chunks
    if single_spread_size >= 16:
        # Spreads with more than 16 photos can be size 16 or size 24
        split_size = get_large_spread_split_size(single_spread_size)
        return _split_by_chunk_size(illegal_group, split_size, n_samples)

    # Otherwise, use clustering to split
    else:
        split_size = get_small_spread_split_size(single_spread_size)
        n_clusters = max(2, int(np.ceil(n_samples / split_size)))
        size_min = max(1, n_samples // n_clusters)
        size_max = min(split_size, n_samples)
        return _split_with_time_clustering(illegal_group, n_clusters, size_min, size_max)


# Main logic
def get_split_points(general_times_list: List[float], group_time_list: List[float],
                     group_key: str) -> Optional[List[float]]:
    """
    Identify temporal split points where a group is interrupted by other photos.

    For eligible wedding categories, examines consecutive pairs of timestamps in
    the group and marks a split point wherever more than 2 photos from other
    groups fall between them.

    Args:
        general_times_list: Sorted list of all photo times across the album.
        group_time_list: Sorted list of times for the current group.
        group_key: The cluster context string (e.g. 'bride', 'ceremony').
            Only specific wedding categories are eligible for splitting.

    Returns:
        A list of time values at which to split the group, or None if the group
        is too small, not an eligible category, or has no gaps.
    """
    if len(group_time_list) < 2:
        return None
    if group_key not in ['walking the aisle', 'bride', 'groom', 'bride and groom', 'groom party', 'bride party', 'portrait']:
        return None

    split_points = list()
    for i in range(len(group_time_list) - 1):
        start_time = group_time_list[i]
        end_time = group_time_list[i + 1]

        count_between = sum(start_time < t < end_time for t in general_times_list)
        if count_between > 2:
            split_points.append(start_time)

    if len(split_points) == 0:
        return None

    return split_points


def get_number_of_spreads(group: pd.DataFrame, group_spread_size: int) -> int:
    """
    Calculate average number of spreads for a photo group.

    Args:
        group (DataFrame): The group of photos.
        group_spread_size (int): Recommended number of photos per spread for this group.

    Returns:
        int: The splitting score (rounded), or 0 if spread size is invalid.
    """
    if group_spread_size > 0:
        return round(group['group_size'].iloc[0] / group_spread_size)
    return 0


def is_split_needed(number_of_spreads: int, group_spread_size: int, group_key: Tuple[str, str, int]) -> bool:
    """
    Determine whether a photo group should be split into subgroups.

    A split is considered necessary if:
      - The number_of_spreads exceeds the minimum (`CONFIGS['min_split_score']`).
      - The number_of_spreads equals the minimum and the group spread size > 5.
      - The number_of_spreads equals 2 and the group spread size >= 12.
      - The group spread size >= 24.
    Additionally, groups with 'cant_split' in their cluster_context are excluded.

    Args:
        number_of_spreads (int):
            Average number of spreads for group.
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
                    number_of_spreads > CONFIGS['min_split_score']
                    or (number_of_spreads > 1 and group_spread_size >= 12)
                    or group_spread_size >= 24
            )
            and 'cant_split' not in group_key[1]
    )


def update_group_sub_index(photos_df: pd.DataFrame, updated_group: pd.DataFrame, logger) -> None:
    """
    Update the `group_sub_index` field in original DataFrame for rows from an updated group.

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
            photos_df.loc[row_index, 'group_sub_index'] = updated_group.loc[row_index, 'group_sub_index']


def update_groups_size(photos_df: pd.DataFrame,
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

