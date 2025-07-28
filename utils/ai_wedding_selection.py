import re
import math
import pandas as pd
import numpy as np
import random
from typing import Dict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from collections import Counter

from utils.parser import CONFIGS,relations,selection_threshold
from utils.wedding_selection_tools import get_clusters,select_non_similar_images
from utils.time_processing import convert_to_timestamp
from utils.person_clustering import modified_run_person_clustering_experiment
from utils.time_orientation_clustering import select_by_new_clustering_experiment

def process_time(images_time):
    general_times = list()
    first_image_time = images_time[0]
    for cur_timestamp in images_time:
        # general_time = cur_timestamp.hour * 60 + cur_timestamp.minute + cur_timestamp.second / 60
        if 0 <= cur_timestamp.hour <= 4:
            # Treat times between midnight and 4 AM as if they belong to the previous day
            general_time = int((cur_timestamp.hour + 24) * 60 + cur_timestamp.minute)
        else:
            general_time = int(cur_timestamp.hour * 60 + cur_timestamp.minute)

            # Calculate difference in days and convert to minutes
        diff_from_first = cur_timestamp - first_image_time
        general_time += diff_from_first.days * 1440
        general_times.append(general_time)

    return general_times


def filter_by_majority(cluster_df,cluster_images, cluster_name, id_column='persons_ids', count_column='number_bodies'):
    """
    Filters rows where:
    - The majority ID appears alone in the list
    - number_boides doesn't exceed max_boides

    Parameters:
    - df: Input DataFrame
    - id_column: Column containing list of person IDs
    - count_column: Column with the count to check (number_boides)
    - max_boides: Maximum allowed value for number_boides

    Returns:
    - Filtered DataFrame
    """
    max_boides = 2 if cluster_name == 'bride and groom' else 1

    # Create a copy to avoid SettingWithCopyWarning
    df = cluster_df.copy()

    solo_counter = Counter()

    for row in df[id_column]:
        if len(row) == 1:
            solo_counter[row[0]] += 1

    # Step 3: Select the ID with the highest score in both categories
    # Sort first by solo (descending), then by co-occur (descending)
    if solo_counter:
        # take the first most solo occur
        best_id = solo_counter.most_common(1)[0][0]
        filtered_df = df[df['persons_ids'].apply(lambda x: x == [best_id])]
        remaining_image_ids = set(filtered_df['image_id'])
        images_filtered = [img_id for img_id in cluster_images if img_id in remaining_image_ids]
        return filtered_df, images_filtered
    else:
        return df,cluster_images

def cluster_by_time(df, eps=0.3, min_samples=2,metric='l1'):
        """
        Cluster images based on their timestamps using DBSCAN

        Parameters:
        - df: DataFrame containing image data
        - time_column: Name of column containing datetime objects
        - eps: Maximum time difference (in minutes) between samples in the same cluster
        - min_samples: Minimum number of images required to form a cluster

        Returns:
        - DataFrame with added 'time_cluster' column
        """

        # Convert datetime to processed minute values
        df['image_time_date'] = df['image_time'].apply(lambda x: convert_to_timestamp(x))
        processed_times = process_time(df['image_time_date'].tolist())

        # Reshape for DBSCAN (needs 2D array)
        time_values = np.array(processed_times).reshape(-1, 1)

        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples,metric=metric).fit(time_values)
        labels = clustering.labels_

        # Add cluster labels to DataFrame
        df = df.copy()
        df['sub_group_time_cluster'] = labels

        return df


def calculate_similarity_scores(im_embedding, ten_photos_embeddings):
    # Ensure im_embedding is a 2D array
    im_embedding_2d = np.array(im_embedding).reshape(1, -1)

    # Convert ten_photos_embeddings to a 2D array
    ten_photos_embeddings_2d = np.array(ten_photos_embeddings)

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(im_embedding_2d, ten_photos_embeddings_2d)

    # Flatten the result to a 1D array
    similarity_scores = similarity_scores.flatten()

    return similarity_scores

def calcuate_tags_score(tags_features, image_features):
    tags_scores = []
    if len(tags_features) == 0:
        return 0.0000002

    for tag,tag_feature in tags_features.items():
        similarity = tag_feature @ image_features
        # maximum similarity by query
        max_query_similarity = np.max(similarity, axis=0)
        tags_scores.append(max_query_similarity)
    # get the highest score of tags similarity
    sorted_scores = sorted(tags_scores, reverse=True)
    return sorted_scores[0]


def calculate_scores(row_data,selected_photos_df, people_ids, tags):
    # persons score
    if len(people_ids) == 0:
        person_score = 1.0  # No selection constraint, treat as perfect
    else:
        if 'persons_ids' in row_data.index and row_data['persons_ids']:
            persons_in_image = row_data['persons_ids']
            selected_count = sum(1 for p in persons_in_image if p in people_ids)
            total_count = len(persons_in_image)

            # Weighted score: proportion of selected people
            person_score = selected_count / total_count

            # Optional bonus if it's a solo selected person
            if total_count == 1 and selected_count == 1:
                person_score = 0.99  # near-perfect
        else:
            person_score = CONFIGS['person_score'] # No person data and people_ids is not empty ⇒ low relevance

    # 10 images similarity score
    if 'embedding' in row_data.index:
        if selected_photos_df.empty:
            similarity_score = 0.00000005
        else:
            similarity_scores = calculate_similarity_scores(row_data['embedding'].tolist(), selected_photos_df['embedding'].tolist())
            similarity_score = abs(similarity_scores.max())
    else:
        similarity_score = CONFIGS['similarity_score']

    if 'image_class' in row_data.index:
        if selected_photos_df.empty:
            class_matching_score = 1.0
        else:
            image_class = row_data['image_class']
            ten_photos_class = selected_photos_df['image_class'].values.tolist()
            class_match_counts = ten_photos_class.count(image_class)
            ratio = class_match_counts / len(ten_photos_class)

            # Optional nonlinear boost
            if ratio >= 0.8:
                class_matching_score = 1.0
            elif ratio >= 0.5:
                class_matching_score = 0.9
            elif ratio >= 0.3:
                class_matching_score = 0.7
            elif ratio > 0:
                class_matching_score = 0.5
            else:
                class_matching_score = 0.1  # mismatch
    else:
        class_matching_score = CONFIGS['class_matching_penalty'] # penalize missing info

    if len(tags) !=0:
        tags_score = calcuate_tags_score(tags, row_data['embedding'])
    else:
        tags_score = 1

    return class_matching_score, similarity_score, person_score, tags_score


def get_scores(df, selected_photos_df, people_ids, tags_features):

    images_scores = {}

    class_scores = []
    similarity_scores = []
    person_scores = []
    tags_scores = []

    for index,data in df.iterrows():
        image_id = data['image_id']
        class_score, sim_score, person_score, tag_score  = calculate_scores(data,selected_photos_df, people_ids, tags_features)
        class_scores.append(class_score)
        similarity_scores.append(sim_score)
        person_scores.append(person_score)
        tags_scores.append(tag_score)

        images_scores[image_id] = None  # Placeholder for now

    df['class_score'] = class_scores
    df['similarity_score'] = similarity_scores
    df['person_score'] = person_scores
    df['tags_score'] = tags_scores

    def normalize_column(col):
        min_val, max_val = col.min(), col.max()
        if max_val - min_val < CONFIGS['ε']:
            return np.ones_like(col) * 0.5  # Avoid division by near-zero; return neutral score
        return (col - min_val) / (max_val - min_val)

    df['class_score_norm'] = normalize_column(df['class_score'])
    df['similarity_score_norm'] = normalize_column(df['similarity_score'])
    df['person_score_norm'] = normalize_column(df['person_score'])
    df['tags_score_norm'] = normalize_column(df['tags_score'])

    # Invert and normalize image order
    df['image_order_score'] = 1 / (df['image_order'] + CONFIGS['ε'])
    df['image_order_score_norm'] = normalize_column(df['image_order_score'])

    #Ranking is better than order cause its normalized and higher is better
    w1 = 0.2  # class
    w2 = 0.2  # similarity
    w3 = 0.4  # person importance
    w4 = 0.1  # tags
    w5 = 0.2  # image Rank

    total_weight = w1 + w2 + w3 + w4 + w5
    w1 /= total_weight
    w2 /= total_weight
    w3 /= total_weight
    w4 /= total_weight
    w5 /= total_weight

    df['total_score'] = (
            w1 * df['class_score_norm'] +
            w2 * df['similarity_score_norm'] +
            w3 * df['person_score_norm'] +
            w4 * df['tags_score_norm'] + w5 * df['ranking']
            # w5 * df['image_order_score_norm']
    )

    sorted_df = df.sort_values(by='total_score', ascending=False)
    sorted_scores = list(zip(sorted_df['image_id'], sorted_df['total_score']))

    return sorted_scores,sorted_df

def load_event_mapping(csv_path: str):
    event_mapping = {}
    try:
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            # Clean sub event name
            raw_event = str(row['sub event'])
            event = re.sub(r"^['\"]+|['\"]+$", '', raw_event).lower().strip()

            for col in df.columns:
                if col.lower().strip() == 'sub event':
                    continue

                category = col.lower().strip()
                value = str(row[col]).strip().lower()

                if category not in event_mapping:
                    event_mapping[category] = {}

                # Process value
                if value == 'yes':
                    event_mapping[category][event] = {'type': 'yes', 'value': 'yes'}
                elif value == 'no':
                    event_mapping[category][event] = {'type': 'no', 'value': 'no'}
                elif '%' in value:
                    try:
                        percentage = float(value.replace('%', ''))
                        event_mapping[category][event] = {'type': 'percentage', 'value': percentage}
                    except ValueError:
                        event_mapping[category][event] = {'type': 'unknown', 'value': value}
                else:
                    try:
                        numeric_val = float(value)
                        event_mapping[category][event] = {'type': 'numeric', 'value': numeric_val}
                    except ValueError:
                        event_mapping[category][event] = {'type': 'unknown', 'value': value}
        return event_mapping

    except Exception as e:
        print(f"Error loading event mapping: {e}")
        return {}


def calculate_selection_revised(n_actual_dict: Dict, lookup_table: Dict, event_mapping: Dict,
                                density: int = 3, logger=None) -> Dict:
    """
    Calculates image selection for photo album spreads, allocating a percentage of the total
    actual images to events marked as 'percentage' type. This version ensures a more
    direct and logical allocation based on percentages of the available photo pool.

    Args:
        category: The focus category (e.g., 'bride and groom').
        n_actual: Total number of actual images available for the category.
        lookup_table: Dictionary with base configurations (n_target, std_target) for each event.
        event_mapping: Dictionary mapping events to category-specific rules ('yes', 'no', 'percentage').
        density: Density factor (1-5) for calculating how many images fit into a single spread.

    Returns:
        A dictionary with selection results for each event, including the number of
        images to select, the calculated number of spreads, and the reasoning for the decision.
    """
    try:
        # Density factors determine how many images are packed into one spread.
        density_factors = {1: 0.5, 2: 1.0, 3: 1.25, 4: 1.75, 5: 2.0}
        density_factor = density_factors.get(density, 1.0)

        event_selection_dict = {}
        for event,n_actual in n_actual_dict.items():
            if event in ['None', 'other']:
                event_selection_dict[event] = 0
                continue

            n_target, std_target = lookup_table.get(event, (0, 0))
            n_target = max(1, round(n_target * density_factor))
            selection = n_target + np.random.gamma(shape=5, scale=std_target)
            event_selection_dict[event] = min(round(selection), n_actual)

    except Exception as e:
        logger.error("Error reading messages: {}".format(e))
        raise e

    return event_selection_dict


def smart_wedding_selection(df, user_selected_photos, people_ids, focus, tags_features,density,
                            logger):
    error_message = None
    ai_images_selected = []
    category_picked = {}

    orientation_time_categories = {
        'bride', 'groom', 'bride and groom', 'bride party', 'groom party', 'speech',
        'full party', 'walking the aisle', 'bride getting dress', 'getting hair-makeup',
        'first dance', 'cake cutting', 'ceremony', 'dancing'
    }

    persons_categories = {'portrait', 'very large group'}

    # Load configs and mapping
    event_mapping = load_event_mapping(CONFIGS['focus_csv_path'])
    actual_number_images_dict = {
        cluster_name: len(cluster_df)
        for cluster_name, cluster_df in df.groupby('cluster_context')
    }

    final_allocation = calculate_selection_revised(
        actual_number_images_dict,
        relations[focus],
        event_mapping[focus],
        density,
        logger
    )

    user_selected_photos_df = df[df['image_id'].isin(user_selected_photos)]

    for iteration, (cluster_name, cluster_df) in enumerate(df.groupby('cluster_context')):
        n_actual = len(cluster_df)
        logger.info("====================================")
        logger.info(f"Starting with {cluster_name} and actual number  of images {n_actual}")


        category_picked.setdefault(cluster_name, {})
        category_picked[cluster_name]['actual'] = n_actual

        is_small_group = (
                n_actual < CONFIGS['small_groups']
                and cluster_name not in CONFIGS['events_disallowing_small_images']
        )

        if cluster_name  in ['other', 'None', 'couple'] or is_small_group:
            continue

        # Get scores for each image
        scores,scored_df = get_scores(cluster_df, user_selected_photos_df, people_ids, tags_features)
        # Skip cluster if all scores are zero or below threshold
        if all(score <= 0 for _, score in scores):
            continue

        image_order_dict = scored_df.set_index('image_id')['total_score'].to_dict()
        available_images_scores = [(image_id, score) for image_id,score in scores
                                   if score > selection_threshold[cluster_name]]

        available_img_ids = [image_id for image_id, _ in available_images_scores]

        user_selected_ids= [
            image_id for image_id in available_img_ids
            if image_id in user_selected_photos_df['image_id'].values
        ]

        available_img_ids_wo_user =[
            image_id for image_id in available_img_ids
            if image_id not in set(user_selected_ids)
        ]

        # Add user selections and adjust allocation
        if len(user_selected_ids) > 0:
            ai_images_selected.extend(user_selected_ids)
            category_picked[cluster_name]['selected'] = len(user_selected_ids)
            final_allocation[cluster_name] -= len(user_selected_ids)

        need = final_allocation[cluster_name]
        has = len(available_img_ids_wo_user)
        # If enough remaining or too few to process more
        if need <= 2 or has <= need:
            to_add = available_img_ids_wo_user[:need]
            ai_images_selected.extend(to_add)
            category_picked[cluster_name]['selected'] = category_picked[cluster_name].get('selected', 0) + len(to_add)

        # -------- Cluster-specific selection strategies --------
        elif cluster_name in orientation_time_categories:
            # Cluster by time and find most solo person
            df_with_time_cluster = cluster_by_time(scored_df)
            images_filtered = available_img_ids_wo_user

            if cluster_name in ['bride', 'groom', 'bride and groom']:
                solo_counter = Counter(
                    row[0] for row in df_with_time_cluster['persons_ids'] if len(row) == 1
                )

                if solo_counter:
                    if cluster_name in ["bride", "groom"]:
                        # Use top solo person
                        best_id = solo_counter.most_common(1)[0][0]
                        filtered_df = df_with_time_cluster[
                            df_with_time_cluster['persons_ids'].apply(lambda x: x == [best_id])
                        ]

                    elif cluster_name == "bride and groom":
                        # Use top two solo persons (bride and groom)
                        top_two_ids = [pid for pid, _ in solo_counter.most_common(2)]

                        # Filter images that contain both top two IDs or have 2 faces or 2 bodies
                        def has_both_ids_or_two_faces_bodies(row):
                            has_both_ids = all(pid in row['persons_ids'] for pid in top_two_ids)
                            has_two_faces = row.get('nfaces', 0) == 2
                            has_two_bodies = row.get('num_bodies', 0) == 2
                            return has_both_ids or has_two_faces or has_two_bodies

                        filtered_df = df_with_time_cluster[
                            df_with_time_cluster.apply(has_both_ids_or_two_faces_bodies, axis=1)]

                    # Final filtering based on available images
                    filtered_ids = set(filtered_df['image_id'])
                    images_filtered = [
                        img_id for img_id in available_img_ids_wo_user
                        if img_id in filtered_ids
                    ]

            df_clustered  = df_with_time_cluster.set_index('image_id')
            grayscale_images = [img for img in images_filtered if df_clustered.at[img, 'image_color'] == 0]
            num_grayscale = len(grayscale_images)
            gray_needed = random.choice([1, 2]) if num_grayscale > CONFIGS['grays_scale_limit'] else (
                1 if grayscale_images else 0)

            # Select grayscale images
            selected_gray_image = []
            if gray_needed:
                gray_df = df_clustered.loc[grayscale_images].sort_values(by='total_score', ascending=False)
                selected_gray_image = gray_df.head(gray_needed).index.tolist()
                ai_images_selected.extend(selected_gray_image)
                need -= len(selected_gray_image)

            colored_images = [
                img for img in images_filtered
                if img not in grayscale_images or img in selected_gray_image
            ]
            filtered_colored_df = df_clustered.loc[colored_images]

            # If exact fit, return them directly
            if need == len(colored_images):
                selected_imgs = colored_images
            else:
                selected_imgs = select_by_new_clustering_experiment(
                    needed_count=need,
                    df_all_data=filtered_colored_df.reset_index()
                )
            # Remove duplicates and finalize selection
            #selected_imgs = remove_similar_images(cluster_name, need, images_filtered, filtered_df)
            ai_images_selected.extend(selected_imgs)
            category_picked[cluster_name]['selected'] = category_picked[cluster_name].get('selected', 0) + len(
                selected_imgs)
        elif cluster_name in persons_categories:
            df_scored  = scored_df.set_index('image_id')

            # Identify grayscale images
            grayscale_images = [img for img in available_img_ids_wo_user if df_scored.at[img, 'image_color'] == 0]
            num_grayscale = len(grayscale_images)
            gray_number_needed = random.choice([1, 2]) if num_grayscale > CONFIGS['grays_scale_limit'] else (
                1 if grayscale_images else 0)

            selected_gray_image = []
            if gray_number_needed:
                gray_df = df_scored.loc[grayscale_images].sort_values(by='total_score', ascending=False)
                selected_gray_image = gray_df.head(gray_number_needed).index.tolist()
                ai_images_selected.extend(selected_gray_image)
                need -= len(selected_gray_image)

            colored_images = [
                img for img in available_img_ids_wo_user
                if img not in grayscale_images or img in selected_gray_image
            ]
            filtered_df = df_scored.loc[colored_images]

            if need == len(colored_images):
                selected_imgs = colored_images
            else:
                selected_imgs = modified_run_person_clustering_experiment(
                    input_images_for_category=colored_images,
                    df_all_data=scored_df,
                    needed_count=need,
                    image_cluster_dict=image_order_dict
                )

            ai_images_selected.extend(selected_imgs)
            category_picked[cluster_name]['selected'] = category_picked[cluster_name].get('selected', 0) + len(
                selected_imgs)
    else:
            clusters_ids = get_clusters(df.reset_index())  # df was indexed earlier
            selected_imgs = select_non_similar_images(cluster_name, clusters_ids, df.reset_index(), need)
            ai_images_selected.extend(selected_imgs)
            category_picked[cluster_name]['selected'] = category_picked[cluster_name].get('selected', 0) + len(
                selected_imgs)

    logger.info("The final Selection for each category: %s", category_picked)
    logger.info(f"Total images: {len(ai_images_selected)}")
    logger.info("*******************************************************")

    return ai_images_selected, error_message
