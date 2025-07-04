import numpy as np
import pandas as pd
import ast

from collections import Counter
from k_means_constrained import KMeansConstrained
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_score


def split_illegal_group(illegal_group, count):
    illegal_group_features = illegal_group['embedding'].values.tolist()
    illegal_time_features = [time for time in illegal_group["general_time"]]

    cluster_labels = [cluster_label for cluster_label in illegal_group["cluster_label"]]
    min_labels = min(cluster_labels)
    max_labels = max(cluster_labels)

    min_value = min(illegal_time_features)
    max_value = max(illegal_time_features)

    if min_labels == max_labels:
        cluster_labels_nor = [0] * len(cluster_labels)
    else:
        cluster_labels_nor = [(label - min_labels) / (max_labels - min_labels) for label in cluster_labels]

    if min_value == max_value:
        time_features_nor = [0] * len(illegal_time_features)
    else:
        time_features_nor = [(time - min_value) / (max_value - min_value) for time in illegal_time_features]

    combined_features = np.column_stack((illegal_group_features, time_features_nor,cluster_labels_nor))

    n_samples = len(combined_features)
    size_min = max(1, n_samples // 4)
    clf = KMeansConstrained(
        n_clusters=2,
        size_min=size_min,
        size_max=len(combined_features),
        random_state=0)
    clf.fit_predict(combined_features)

    labels = clf.labels_
    silhouette_avg = silhouette_score(combined_features, labels)

    if silhouette_avg < 0.15:
        return None,None

    content_cluster_origin = illegal_group['cluster_context'].values[0]
    labels = [f'{content_cluster_origin}_{label}_{count}' for label in labels]
    label_counts = Counter(labels)
    # Assign cluster labels with a prefix or suffix to distinguish them from other groups
    # illegal_group['main_content_cluster'] = [f"{content_cluster_origin}_{label}" for label in labels]
    illegal_group.loc[:,'cluster_context'] = [label for label in labels]

    return illegal_group, label_counts


def split_illegal_group_by_time(illegal_group, count):
    """
    Split illegal group into two subgroups based on temporal clustering.

    Args:
        illegal_group: DataFrame containing the group to be split
        count: Integer counter for naming new clusters

    Returns:
        tuple: (modified_DataFrame with new cluster labels, label_counts) or (None, None) if splitting fails
    """
    # Get time features and normalize them
    time_features = illegal_group["general_time"].values.reshape(-1, 1)

    # Early return if all times are identical
    if np.all(time_features == time_features[0]):
        return None, None

    # Calculate minimum group size
    n_samples = len(illegal_group)
    size_min = max(1, n_samples // 4)

    try:
        # Apply constrained k-means clustering on time values
        clf = KMeansConstrained(
            n_clusters=2,
            size_min=size_min,
            size_max=len(time_features),
            random_state=0
        )
        labels = clf.fit_predict(time_features)

        # Calculate silhouette score to evaluate clustering quality
        silhouette_avg = silhouette_score(time_features, labels)

        # If clustering quality is poor, don't split
        if silhouette_avg < 0.15:
            return None, None

        # Create new cluster labels with original context preserved
        content_cluster_origin = illegal_group['cluster_context'].values[0]
        new_labels = [f'{content_cluster_origin}_{label}_{count}' for label in labels]

        # Count occurrences of each label
        label_counts = Counter(new_labels)

        # Check if split is too uneven (one group < 25% of total)
        total_samples = sum(label_counts.values())
        if any(count < total_samples * 0.25 for count in label_counts.values()):
            return None, None

        # Assign new cluster labels to the DataFrame
        illegal_group.loc[:, 'cluster_context'] = new_labels

        # Sort groups by mean time to ensure temporal ordering
        mean_times = illegal_group.groupby('cluster_context')['general_time'].mean()
        sorted_clusters = mean_times.sort_values().index

        # Rename labels to ensure earlier time group gets lower number
        mapping = {old: f'{content_cluster_origin}_{i}_{count}'
                   for i, old in enumerate(sorted_clusters)}
        illegal_group.loc[:, 'cluster_context'] = illegal_group['cluster_context'].map(mapping)

        return illegal_group, Counter(illegal_group['cluster_context'])

    except Exception as e:
        print(f"Error during temporal splitting: {str(e)}")
        return None, None


def merge_illegal_group(main_groups, illegal_group):
    clusters_features = [group['embedding'].values.copy() for group in main_groups]

    # Aggregate features within each group
    group_features = [group.mean() if group.shape[0] > 1 else group[0] for group in clusters_features]
    group_features_np = np.array(group_features)


    if illegal_group['embedding'].values.shape[0] > 1 :
        illegal_group_features = illegal_group['embedding'].values.mean()
    else:
        illegal_group_features = illegal_group['embedding'].values[0]

    if len(illegal_group_features.shape) == 1:
        intded_group_fe = illegal_group_features.reshape(1, -1)
    else:
        intded_group_fe = illegal_group_features

    inteded_group_time = illegal_group['general_time'].values.mean()
    intded_group_fe_with_time = np.column_stack((intded_group_fe, inteded_group_time))

    main_groups_time_without_illegal = [group['general_time'].values.mean() for i, group in enumerate(main_groups)]
    groups_combined_features = np.column_stack((group_features_np, main_groups_time_without_illegal))

    dist_to_illegal_group = pairwise_distances(intded_group_fe_with_time,groups_combined_features,
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


def merge_illegal_group_by_time(main_groups, illegal_group):
    """
    Merge illegal group with the closest group by time that meets size requirements.

    Args:
        main_groups: List of DataFrame groups to potentially merge with
        illegal_group: DataFrame of the group to be merged

    Returns:
        tuple: (modified_illegal_group, combined_group, selected_cluster_content_index)
    """
    intended_group_time = illegal_group['general_time'].values.mean()
    main_groups_time_without_illegal = [group['general_time'].values.mean() for i, group in enumerate(main_groups)]

    # Calculate time differences and create sorted indices
    time_differences = np.abs(np.array(main_groups_time_without_illegal) - intended_group_time)
    sorted_indices = np.argsort(time_differences)

    selected_cluster_content_index = None
    selected_cluster = None

    # Try groups in order of temporal proximity
    for idx in sorted_indices:
        selected_cluster = main_groups[idx]
        len_combine_group = len(selected_cluster) + len(illegal_group)

        # Check if the combination meets size requirements
        if len_combine_group <= 38:
            selected_cluster_content_index = selected_cluster['cluster_context'].iloc[0]
            break

    # If still no suitable group found, use the temporally closest group regardless of size
    if selected_cluster_content_index is None:
        closest_idx = sorted_indices[0]
        selected_cluster = main_groups[closest_idx]
        selected_cluster_content_index = selected_cluster['cluster_context'].iloc[0]

    illegal_group.loc[:, 'cluster_context'] = selected_cluster_content_index
    illegal_group.loc[:, 'cluster_context_2nd'] = 'merged'
    combined_group = pd.concat([selected_cluster, illegal_group], ignore_index=False)

    return illegal_group, combined_group, selected_cluster_content_index