import copy
import pandas as pd
import numpy as np

from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from concurrent.futures import ThreadPoolExecutor

def read_timestamp(timestamp_str):
    try:
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")


def convert_to_timestamp(time_integer):
    return datetime.fromtimestamp(time_integer)


def process_image_time_row(args):
    """Processes a single row to compute the general time."""
    row_time, first_image_time = args
    row_time = copy.deepcopy(row_time)
    cur_timestamp = convert_to_timestamp(row_time['image_time'])

    if 0 <= cur_timestamp.hour <= 4:
        general_time = int((cur_timestamp.hour + 24) * 60 + cur_timestamp.minute)
    else:
        general_time = int(cur_timestamp.hour * 60 + cur_timestamp.minute)

    diff_from_first = cur_timestamp - first_image_time
    general_time += diff_from_first.days * 1440

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
    dbscan = DBSCAN(eps=20, min_samples=1)  # eps is in minutes, adjust as needed
    clusters = dbscan.fit_predict(X)
    best_n = len(set(clusters))
    return clusters, best_n


def get_time_clusters(general_time_df):
    # Cluster by time
    X = general_time_df.values.reshape(-1, 1)
    initial_clusters, best_n = get_time_clusters_dbscan(X)

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
            groups_time_list.append((group_dict, sum(photos_time_list) / len(photos_time_list) if len(photos_time_list) > 0 else float('inf')))
        groups_time_list = sorted(groups_time_list, key=lambda x: x[1])
        sorted_data_list = [group_data for group_data, _ in groups_time_list]

        return sorted_data_list
    except Exception as ex:
        logger.warning(f"Error in sorting groups by time: {ex}. Return groups without sorting by time.")
        return groups_list
