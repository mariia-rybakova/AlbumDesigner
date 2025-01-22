import numpy as np
import pandas as pd
import ast

from collections import Counter
from k_means_constrained import KMeansConstrained
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_score

def split_illegal_group(illegal_group):
    count= "NA"
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

    if silhouette_avg > 0.15:
        #logger.info("Clustering is good for gallery group.", )
        print("Clustering is good for gallery group.", )
    else:
        #logger.info("Clustering is bad.so we wont split it")
        print("Clustering is good for gallery group.", )
        return None, None

    content_cluster_origin = illegal_group['cluster_context'].values[0]
    labels = [f'{content_cluster_origin}_{label}_{count}' for label in labels]
    label_counts = Counter(labels)
    # Assign cluster labels with a prefix or suffix to distinguish them from other groups
    # illegal_group['main_content_cluster'] = [f"{content_cluster_origin}_{label}" for label in labels]
    illegal_group.loc[:,'cluster_context'] = [label for label in labels]

    return illegal_group, label_counts

def parse_embedding(embedding_str):
    try:
        # Ensure the string has proper commas and brackets
        cleaned_str = embedding_str.replace(' ', ', ').replace('[,', '[').replace(',]', ']')
        return ast.literal_eval(cleaned_str)
    except Exception as e:
        print(f"Error parsing: {embedding_str} - {e}")
        return None


def merge_illegal_group(main_groups, illegal_group):
    clusters_features = [group['embedding'].values for group in main_groups]

    # Aggregate features within each group
    group_features = [np.mean(group, axis=0) for group in clusters_features]
    group_features_np = np.array(group_features)

    illegal_group_features = illegal_group['embedding'].values.tolist()
    illegal_group_features_np = np.array(illegal_group_features)

    if len(illegal_group_features_np.shape) == 1:
        intded_group_fe = illegal_group_features_np.reshape(1, -1)
    else:
        intded_group_fe = illegal_group_features_np

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
    while len(selected_cluster) != 44 and len(
            selected_cluster) > 38 or len_combine_group > 38 and len_combine_group != 44 and len(main_groups) != 2:
        dist_to_illegal_group = np.delete(dist_to_illegal_group, min_distance_idx)
        if len(dist_to_illegal_group) == 0:
            break
        min_distance_idx = np.argmin(dist_to_illegal_group)

        # Identify the selected group corresponding to the second highest mean distance
        selected_cluster = main_groups[min_distance_idx]
        len_combine_group = len(selected_cluster) + len(illegal_group)

        # If the condition is met, break the loop
        if len(selected_cluster) <= 38 or len(
                selected_cluster) == 44 or len_combine_group > 38 and len_combine_group != 44:
            break

    selected_cluster_content_index = list(selected_cluster['cluster_context'])[0]
    illegal_group.loc[:,'cluster_context'] = selected_cluster_content_index
    illegal_group.loc[:,'cluster_context_2nd'] = 'merged'
    combine_groups = pd.concat([selected_cluster, illegal_group], ignore_index=False)

    return illegal_group, combine_groups, selected_cluster_content_index
