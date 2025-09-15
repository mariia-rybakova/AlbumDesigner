import copy
import pandas as pd
import numpy as np

from statistics import median
from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN


def convert_to_timestamp(time_integer):
    try:
        dt = datetime.fromtimestamp(time_integer)
    except Exception:
        dt = datetime.now()
    return dt


def process_image_time_row(args):
    """Processes a single row to compute the general time."""
    row_time, first_image_time = args
    row_time = copy.deepcopy(row_time)
    cur_timestamp = convert_to_timestamp(row_time['image_time'])

    # cur_day_time = int(cur_timestamp.hour * 3600 + cur_timestamp.minute * 60 + cur_timestamp.second)
    diff_from_first = (cur_timestamp - first_image_time).total_seconds()
    general_time = int(diff_from_first)

    row_time['general_time'] = int(general_time)
    return row_time


def process_image_time(data_df):
    # Convert image_time to timestamp
    data_df['image_time_date'] = data_df['image_time'].apply(lambda x: convert_to_timestamp(x))

    # Sort by timestamp to find the first image time
    data_df = data_df.sort_values(by='image_time_date')
    first_image_time = data_df.iloc[0]['image_time_date']

    time_data_dict = data_df[['image_id', 'image_time']].to_dict('records')

    args_list = [(row, first_image_time) for row in time_data_dict]

    # Using Pool to process each row in parallel
    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     processed_rows = list(executor.map(process_image_time_row, args_list))

    processed_rows = []
    for args in args_list:
        processed_rows.append(process_image_time_row(args))
    processed_df = pd.DataFrame(processed_rows)
    processed_df = data_df.merge(processed_df[['image_id', 'general_time']], how='left', on='image_id')

    sorted_by_time_df = processed_df.sort_values(by="general_time", ascending=True)
    image_id2general_time = dict(zip(processed_df['image_id'], processed_df['general_time']))

    return sorted_by_time_df, image_id2general_time


def get_time_clusters_gmm(X):
    # Determine the optimal number of clusters using Bayesian Information Criterion (BIC)
    n_components = np.arange(1, 15)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X) for n in n_components]
    bics = [m.bic(X) for m in models]
    # Select the model with the lowest BIC
    best_n = n_components[np.argmin(bics)]
    gmm = GaussianMixture(n_components=best_n, covariance_type='full', random_state=0)
    gmm.fit(X)
    initial_clusters = gmm.predict(X)
    return initial_clusters, best_n


def get_time_clusters_dbscan(X):
    for min_samples_possible in [3, 5, 7]:
        dbscan = DBSCAN(eps=1200, min_samples=min_samples_possible)  # eps is in minutes, adjust as needed
        clusters = dbscan.fit_predict(X)
        best_n = len(set(clusters))
        if best_n <= 10:
            break
    return clusters, best_n


def get_time_clusters(selected_df,all_photos_df=None):
    # Cluster by time
    general_time_df = selected_df[['general_time']]
    X = general_time_df.values.reshape(-1, 1)
    if all_photos_df is not None:
        all_X = all_photos_df['image_time'].values.reshape(-1, 1)
        initial_clusters, best_n = get_time_clusters_dbscan(all_X)

        # Assign clusters to all_photos_df
        all_photos_df['time_cluster'] = initial_clusters

        # Map clusters to selected_df based on matching id
        selected_df = selected_df.merge(
            all_photos_df[['image_id', 'time_cluster']],
            on='image_id',
            how='left'
        )
        initial_clusters = selected_df['time_cluster'].values
    else:

        initial_clusters, best_n = get_time_clusters_dbscan(X)

    if -1 in initial_clusters:
        noise_mask = initial_clusters == -1
        valid_mask = initial_clusters != -1

        if np.any(valid_mask):  # Only if there are valid clusters
            for i in np.where(noise_mask)[0]:
                # Find distances to all valid points
                distances = np.abs(X[i] - X[valid_mask].flatten())
                nearest_idx = np.where(valid_mask)[0][np.argmin(distances)]
                initial_clusters[i] = initial_clusters[nearest_idx]
        else:
            # If all points are noise, assign them to a single cluster
            initial_clusters = np.zeros_like(initial_clusters)
            best_n = 1

        # Recalculate best_n excluding noise points
        best_n = len(set(initial_clusters[initial_clusters != -1]))

    # Calculate mean time for each cluster
    cluster_means = {}
    for cluster_id in range(best_n):
        mask = initial_clusters == cluster_id
        mean_time = np.mean(X[mask])
        cluster_means[cluster_id] = mean_time

    # Sort clusters by their mean time
    sorted_clusters = dict(sorted(cluster_means.items(), key=lambda item: item[1]))
    cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted_clusters.keys())}
    clusters = np.array([cluster_mapping[c] for c in initial_clusters])

    return clusters


def merge_time_clusters_by_context(sorted_df, context_clusters_list, logger=None):
    """
    Modifies time clusters based on context clusters.
    For each context cluster in the list, sets the same time cluster for all rows with that context cluster.
    Uses the time cluster that appears most frequently among the identities in that context group.

    Args:
        sorted_df (pd.DataFrame): DataFrame with 'time_cluster' and 'cluster_context' columns
        context_clusters_list (list): List of context clusters to process

    Returns:
        pd.DataFrame: Modified DataFrame with updated time clusters
    """
    if sorted_df is None or sorted_df.shape[0] == 0 or context_clusters_list is None:
        return sorted_df
    try:
        df = sorted_df.copy()

        for context_cluster in context_clusters_list:
            # Get rows with this context cluster
            mask = df['cluster_context'] == context_cluster
            context_group = df[mask]

            if len(context_group) > 0:
                # Find the most common time cluster in this context group
                most_common_time = context_group['time_cluster'].mode().iloc[0]

                # Update time cluster for all rows with this context cluster
                df.loc[mask, 'time_cluster'] = most_common_time

        return df
    except Exception as ex:
        if logger is not None:
            logger.warning(f"Issue in merge_time_clusters_by_context: {ex}. Returning df without changes.")
        return sorted_df

def sort_groups_by_time(groups_list, logger):
    try:
        groups_time_list = list()
        for number_groups, group_dict in enumerate(groups_list):
            photos_time_list = list()
            # Sort spreads inside each group
            for group_name in group_dict.keys():
                group_result = group_dict[group_name]
                total_spreads = len(group_result)
                for i in range(total_spreads):
                    group_data = group_result[i]
                    if isinstance(group_data, float):
                        continue
                    if isinstance(group_data, list):
                        number_of_spreads = len(group_data)
                        # Sort spreads inside this group_data by min general_time of photos in the spread
                        def spread_time_key(spread):
                            left_page_photos = list(spread[1])
                            right_page_photos = list(spread[2])
                            all_photos = left_page_photos + right_page_photos
                            times = [photo.general_time for photo in all_photos]
                            return min(times) if times else float('inf')
                        group_data.sort(key=spread_time_key)
                        # After sorting, continue as before
                        for spread_index in range(number_of_spreads):
                            left_page_photos = list(group_data[spread_index][1])
                            right_page_photos = list(group_data[spread_index][2])
                            all_photos = left_page_photos + right_page_photos
                            for cur_photo in all_photos:
                                cur_photo_time = cur_photo.general_time
                                photos_time_list.append(cur_photo_time)
            groups_time_list.append((group_dict, median(photos_time_list) if len(photos_time_list) > 0 else float('inf')))
        groups_time_list = sorted(groups_time_list, key=lambda x: x[1])
        sorted_data_list = [group_data for group_data, _ in groups_time_list]

        return sorted_data_list
    except Exception as ex:
        logger.warning(f"Error in sorting groups by time: {ex}. Return groups without sorting by time.")
        return groups_list
