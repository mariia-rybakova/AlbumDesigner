import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import pairwise_distances


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


def _normalize_feature(feature):
    min_value = min(feature)
    max_value = max(feature)

    if min_value == max_value:
        feature_normalized = [0] * len(feature)
    else:
        feature_normalized = [(value - min_value) / (max_value - min_value) for value in feature]
    return feature_normalized


# def split_illegal_group(illegal_group, count):
#     illegal_group_features = illegal_group['embedding'].values.tolist()
#     illegal_time_features = [time for time in illegal_group["general_time"]]
#     cluster_labels = [cluster_label for cluster_label in illegal_group["cluster_label"]]
#
#     cluster_labels_nor = _normalize_feature(cluster_labels)
#     time_features_nor = _normalize_feature(illegal_time_features)
#     combined_features = np.column_stack((illegal_group_features, time_features_nor, cluster_labels_nor))
#
#     n_samples = len(combined_features)
#     n_clusters = 2
#     size_min = max(1, n_samples // 4)
#     size_max = n_samples
#     labels = _clusterize(illegal_group, combined_features, n_clusters, size_min, size_max,
#                          silhouette = True)
#     if labels is None:
#         return None
#
#     return illegal_group


def _mean_or_first_element(feature):
    if feature.shape[0] > 1:
        return feature.mean()
    return feature[0]


def _get_embedding_feature(illegal_group):
    illegal_group_features = illegal_group['embedding'].values
    illegal_group_features = _mean_or_first_element(illegal_group_features)

    if len(illegal_group_features.shape) == 1:
        return illegal_group_features.reshape(1, -1)
    return illegal_group_features


def merge_illegal_group(main_groups, illegal_group):
    # Aggregate features within illegal group
    intded_group_fe = _get_embedding_feature(illegal_group)
    inteded_group_time = illegal_group['general_time'].values.mean()
    intded_group_fe_with_time = np.column_stack((intded_group_fe, inteded_group_time))

    # Aggregate features within each group
    clusters_features = [group['embedding'].values.copy() for group in main_groups]
    group_features = [_mean_or_first_element(group) for group in clusters_features]
    group_features_np = np.array(group_features)
    main_groups_time_without_illegal = [group['general_time'].values.mean() for group in main_groups]
    groups_combined_features = np.column_stack((group_features_np, main_groups_time_without_illegal))

    # Compute distances
    dist_to_illegal_group = pairwise_distances(intded_group_fe_with_time, groups_combined_features,
                                               metric='cosine')

    # Find the index of the group with the minimum distance to illegal_group
    min_distance_idx = np.argmin(dist_to_illegal_group)
    selected_cluster = main_groups[min_distance_idx]

    len_combine_group = len(selected_cluster) + len(illegal_group)
    # We dont want to split the 2 images or less group per spread
    while len(selected_cluster) != 44 and len(selected_cluster) > 38 or len_combine_group > 38 and len_combine_group != 44 and len(main_groups) != 2:
        dist_to_illegal_group = np.delete(dist_to_illegal_group, min_distance_idx)
        if len(dist_to_illegal_group) == 0:
            break
        min_distance_idx = np.argmin(dist_to_illegal_group)

        # Identify the selected group corresponding to the second highest mean distance
        selected_cluster = main_groups[min_distance_idx]
        len_combine_group = len(selected_cluster) + len(illegal_group)

        # If the condition is met, break the loop
        if len(selected_cluster) <= 38 or len(selected_cluster) == 44 or len_combine_group > 38 and len_combine_group != 44:
            break

    selected_cluster_content_index = list(selected_cluster['cluster_context'])[0]
    illegal_group.loc[:,'cluster_context'] = selected_cluster_content_index
    illegal_group.loc[:,'cluster_context_2nd'] = 'merged'
    combine_groups = pd.concat([selected_cluster, illegal_group], ignore_index=False)

    return illegal_group, combine_groups, selected_cluster_content_index


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