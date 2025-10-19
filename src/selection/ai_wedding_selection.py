import re
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from collections import Counter

from utils.configs import CONFIGS
from utils.lookup_table_tools import wedding_lookup_table
from utils.configs import relations,selection_threshold
from utils.selection.wedding_selection_tools import get_clusters,select_non_similar_images
from utils.time_processing import convert_to_timestamp
from src.selection.person_clustering import person_max_union_selection
from utils.selection.time_orientation_selection import select_images_by_time_and_style,identify_temporal_clusters,filter_similarity
from utils.selection.testing_selection import filter_similarity_diverse,filter_similarity_diverse_new,select_remove_similar



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


def time_clusters_fixed_span(df,logger, time_col="image_time_date", minutes=4, out_col="sub_group_time_cluster"):

    try:
        df = df.copy()
        # df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.sort_values(time_col).reset_index(drop=True)

        dt = pd.Timedelta(minutes=minutes)
        cluster_id = 0
        cluster_start = None
        labels = []

        for t in df[time_col]:
            if cluster_start is None:
                cluster_start = t
                labels.append(cluster_id)
                continue
            if t - cluster_start > dt:
                cluster_id += 1
                cluster_start = t
            labels.append(cluster_id)

        # remove single image clusters
        if labels:
            counts = pd.Series(labels).value_counts()
            singleton_ids = set(counts[counts == 1].index)

            if singleton_ids:
                labels_merged = labels[:]
                n = len(labels_merged)

                for i,lab in enumerate(labels):
                    if lab not in singleton_ids:
                        continue

                    if i > 0:
                        labels_merged[i] = labels_merged[i-1]
                    elif i + 1 <n :
                        labels_merged[i] = labels_merged[i+1]

                labels = labels_merged

        df[out_col] = labels

    except Exception as e:
        logger.error(f"Error in time_clusters_fixed_span function: {e}")

    return df

def cluster_by_time(df,logger, eps=0.3, min_samples=2,metric='l1'):
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
        try:
            processed_times = process_time(df['image_time_date'].tolist())
            # Reshape for DBSCAN (needs 2D array)
            time_values = np.array(processed_times).reshape(-1, 1)

            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=eps, min_samples=min_samples,metric=metric).fit(time_values)
            labels = clustering.labels_

            # Add cluster labels to DataFrame
            df = df.copy()
            df['sub_group_time_cluster'] = labels

            df = df.sort_values('image_time_date').reset_index(drop=True)

            # Fix -1 labels by inheriting previous cluster
            col = "sub_group_time_cluster"

            s = df[col].replace(-1, pd.NA)  # treat -1 as missing
            s = s.ffill().bfill()  # fill from previous; then fill start from next
            df[col] = s.astype(df[col].dtype)

        except Exception as e:
            logger.error(f"Error in selection: {e}")

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


def get_scores(df, selected_photos_df, people_ids, tags_features, logger):
    try:
        images_scores = {}
        class_scores, similarity_scores, person_scores, tags_scores = [], [], [], []
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

        # Inline normalization function
        def normalize(col):
            min_val, max_val = col.min(), col.max()
            return np.ones_like(col) * 0.5 if max_val - min_val < CONFIGS['ε'] else (col - min_val) / (max_val - min_val)

        # Normalize all relevant score columns
        df['class_score_norm'] = normalize(df['class_score'])
        df['similarity_score_norm'] = normalize(df['similarity_score'])
        df['person_score_norm'] = normalize(df['person_score'])
        df['tags_score_norm'] = normalize(df['tags_score'])

        # Invert and normalize image order
        df['image_order_score'] = 1 / (df['image_order'] + CONFIGS['ε'])
        df['image_order_score_norm'] = normalize(df['image_order_score'])

        #Ranking is better than order cause its normalized and higher is better
        total_weight = sum(CONFIGS['weights'].values())
        for key in CONFIGS['weights']:
            CONFIGS['weights'][key] /= total_weight

        # Compute final weighted total score
        df['total_score'] = (
                CONFIGS['weights']['class'] * df['class_score_norm'] +
                CONFIGS['weights']['similarity'] * df['similarity_score_norm'] +
                CONFIGS['weights']['person'] * df['person_score_norm'] +
                CONFIGS['weights']['tags'] * df['tags_score_norm'] +
                CONFIGS['weights']['rank'] * df['ranking']  # swap with image_order_score_norm if needed
        )

        sorted_df = df.sort_values(by='total_score', ascending=False)
        sorted_scores = list(zip(sorted_df['image_id'], sorted_df['total_score']))

    except Exception as e:
        logger.error(f"Error getting scores for images: {e}")
        return None, None

    return sorted_scores,sorted_df

def load_event_mapping(csv_path: str,logger):
    event_mapping = {}
    try:
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            # Clean sub event name
            raw_event = str(row['sub event'])
            # event = re.sub(r"^['\"]+|['\"]+$", '', raw_event).lower().strip()
            event = re.sub(r"^['\"]+|['\"]+$", '', raw_event).strip()

            for col in df.columns:
                if col.lower().strip() == 'sub event':
                    continue

                category = col
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
        logger.error(f"Error loading event mapping: {e}")
        return {}

def define_min_max_spreads(df,focus_table,n_actual_dict):
    total_images = sum(n_actual_dict.values())
    total_people =  df['persons_ids'].explode().nunique()

    actual_events_have = 0
    actual_events_needed = 0
    for event, config in focus_table.items():
        if isinstance(config, dict) and 'value' in config:
            if isinstance(config['value'], float):
                if event in n_actual_dict.keys() and config['value'] > 0.0:
                    actual_events_have  += 1
                else:
                    actual_events_needed += 1

    if total_images <= 600 and total_people <= 30 and actual_events_have / actual_events_needed <= 0.8:
        return 15,18
    elif total_images > 1000 and total_people > 70 and actual_events_have / actual_events_needed > 0.8:
        return 23,26
    else:
        return 19,22


def calculate_optimal_selection(
    n_actual_dict,
    image_lookup_table, #relation_table
    spreads_per_category_table, # LUT
    focus_table,
    density,
    df,
    logger
    ):
    try:


        min_total_spreads, max_total_spreads = define_min_max_spreads(df,focus_table,n_actual_dict)

        TARGET_SPREADS = max_total_spreads
        density_factors = CONFIGS['density_factors']
        density_factor = density_factors.get(density, 1.0)

        modified_lut = spreads_per_category_table.copy() # Create a copy to avoid modifying the original LUT

        for event,pair in modified_lut.items():
            modified_lut[event] = (min(24,max(1, pair[0]*density_factor)),pair[1])  # Ensure base spreads are at least 1 and not above 24

        # Calculate total numeric values
        total_value = sum(
            config['value']
            for config in focus_table.values()
            if isinstance(config, dict) and isinstance(config.get('value'), (int, float))
        )

        # Add spreads in one pass
        for event, config in focus_table.items():
            if isinstance(config, dict) and 'value' in config:
                if isinstance(config['value'], str):
                    config['spreads'] = 0
                    config['photos'] = 1
                    config['miss'] = max(0,config['photos'] - n_actual_dict.get(event, 0))
                    config['miss_spreads'] = 0
                    if config['miss'] > 0:
                        config['photos'] = 0
                    config['over_photos'] = max(0, n_actual_dict.get(event, 0) - config['photos'])
                    config['over_spreads'] = config['over_photos'] / modified_lut[event][0]
                else:
                    config['spreads'] = config['value'] / total_value * TARGET_SPREADS
                    config['photos'] = config['spreads']*modified_lut[event][0]
                    config['miss'] = max(0,config['photos'] - n_actual_dict.get(event, 0))
                    config['miss_spreads'] = round(config['miss'] / modified_lut[event][0])
                    if config['miss'] > 0:
                        config['photos'] = config['photos'] - config['miss']
                        config['spreads'] = round(config['photos']/ modified_lut[event][0])
                    config['over_photos'] = max(0,n_actual_dict.get(event, 0) - config['photos'])
                    config['over_spreads'] = config['over_photos'] / modified_lut[event][0]

        total_miss_spreads = sum(
            config['miss_spreads']
            for config in focus_table.values()
        )
        total_over_spreads = sum(
            config['over_spreads']
            for config in focus_table.values()
        )

        total_miss_spreads = max(0, total_miss_spreads)
        while total_over_spreads>1 and total_miss_spreads > 1:
            for event, config in focus_table.items():
                if config['over_spreads'] > 1 and total_miss_spreads > 1:
                    # Reduce over spreads
                    config['over_spreads'] -= 1
                    config['over_photos'] -= modified_lut[event][0]
                    config['photos'] += modified_lut[event][0]
                    config['spreads'] += 1
                    total_over_spreads -= 1
                    total_miss_spreads -= 1
            availble_over_events = 0
            for config in focus_table.values():
                availble_over_events += config['over_spreads']>1
            if availble_over_events == 0:
                break

        spreads = {}
        selections = {}
        for event in n_actual_dict:
            spreads[event] = focus_table[event]['spreads']
            selections[event] = round(focus_table[event]['photos'])

        if total_miss_spreads > 1:
            logger.warning(f"Unable to fill desired spreads. Total miss spreads: {total_miss_spreads}")


    except Exception as e:
        logger.error(f"Error calculate_optimal_selection: {e}")
        return None,None

    return selections,spreads

def smart_wedding_selection(df, user_selected_photos, people_ids, focus, tags_features,density,
                            logger):
    try:
        error_message = None
        ai_images_selected = []
        category_picked = {}

        orientation_time_categories = {
            'bride', 'groom', 'bride and groom', 'bride party', 'groom party',
            'full party', 'walking the aisle',
            'first dance', 'cake cutting', 'ceremony', 'dancing'
        }

        persons_categories = {'portrait', 'very large group','speech'}

        # Load configs and mapping
        event_mapping = load_event_mapping(CONFIGS['focus_csv_path'], logger)

        if len(focus)>0:
            focus_table = event_mapping.get(focus[0], event_mapping['brideAndGroom'])
            relation_table = relations.get(focus[0], relations['brideAndGroom'])
        else:
            focus_table = event_mapping['brideAndGroom']
            relation_table = relations['brideAndGroom']

        actual_number_images_dict = {
            cluster_name: len(cluster_df)
            for cluster_name, cluster_df in df.groupby('cluster_context')
        }

        if len(df) <= 200 and density >= 3:
            return df['image_id'].values.tolist(), {}, error_message

        images_allocation,spreads_allocation = calculate_optimal_selection(
        actual_number_images_dict,
        relation_table,
        wedding_lookup_table,
        focus_table,
        density,
        df,
        logger
        )

        if images_allocation is None:
            return None,None, "No images got selected!"

        user_selected_photos_df = df[df['image_id'].isin(user_selected_photos)]

        def get_candidate_images(cluster_df, cluster_name):
            """
            Returns a scored_df DataFrame and list of available image ids
            depending on whether scoring is required.
            """
            # CASE 1: no scoring, all images are candidates
            if len(people_ids) == 0 and len(user_selected_photos) == 0 and len(tags_features) == 0:
                scored_df = cluster_df.copy()
                scored_df["total_score"] = 1.0  # assign dummy uniform score
                scored_df = scored_df.sort_values(by='image_order', ascending=True)
                return scored_df, scored_df["image_id"].tolist(), True

            # CASE 2: scoring required
            scores, scored_df = get_scores(cluster_df, user_selected_photos_df, people_ids, tags_features, logger)
            if scores is None or all(score <= 0 for _, score in scores):
                return None, []
            candidates_images_scores = [
                (image_id, score) for image_id, score in scores
                if score > selection_threshold[cluster_name]
            ]
            if len(candidates_images_scores) < len(scored_df):
                # fallback: at least top N images
                candidates_images_scores = [
                    (row['image_id'], row['total_score'])
                    for _, row in scored_df.head(images_allocation[cluster_name]).iterrows()
                ]

            sorted_candidates = sorted(candidates_images_scores, key=lambda x: x[1], reverse=True)
            available_img_ids = [image_id for image_id, _ in sorted_candidates]
            return scored_df, available_img_ids,False

        for iteration, (cluster_name, cluster_df) in enumerate(df.groupby('cluster_context')):
            n_actual = len(cluster_df)
            category_picked.setdefault(cluster_name, {})
            category_picked[cluster_name]['actual'] = n_actual
            need = images_allocation[cluster_name]
            if need == 0:
                continue

            scored_df, available_img_ids,no_selection = get_candidate_images(cluster_df, cluster_name)

            if scored_df is None or len(available_img_ids) == 0:
                continue

            if cluster_name in ['accessories', 'wedding dress']:
                user_selected_ids = [
                    image_id for image_id in available_img_ids
                    if image_id in user_selected_photos_df['image_id'].values
                ]
                if len(user_selected_ids) != 0:
                    ai_images_selected.extend(user_selected_ids[:need])
                    category_picked[cluster_name]['selected'] = category_picked[cluster_name].get('selected', 0) + len(
                        user_selected_ids[:need])
                    continue
                else:
                    ai_images_selected.extend(available_img_ids[:need])
                    category_picked[cluster_name]['selected'] = category_picked[cluster_name].get('selected', 0) + len(
                        available_img_ids[:need])
                continue

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
                images_allocation[cluster_name] -= len(user_selected_ids)

            #remove the images that selected from the user
            scored_df = scored_df[scored_df['image_id'].isin(available_img_ids_wo_user)]
            scored_df['image_time_date'] = scored_df['image_time'].apply(lambda x: convert_to_timestamp(x))
            min_keep = int(np.ceil(need * 3))

            if len(scored_df) < min_keep:
                valid_images_df = scored_df.copy()
            else:
                valid_images_df = identify_temporal_clusters(scored_df, 'image_time_date', 20, 4, logger)

            if valid_images_df.empty:
                logger.info(f"There are no images to select for {cluster_name}")
                continue

            df_with_time_cluster = time_clusters_fixed_span(valid_images_df, logger)
            color_candidates_df = df_with_time_cluster[df_with_time_cluster['image_color'] != 0]
            grayscale_candidates_df = df_with_time_cluster[df_with_time_cluster['image_color'] == 0]
            has = len(df_with_time_cluster)

            if no_selection:
                image_order_dict = (
                    valid_images_df.set_index('image_id')['image_order']
                    .sort_values(ascending=True)
                    .to_dict()
                )
            else:
                image_order_dict = (
                    valid_images_df.set_index('image_id')['total_score']
                    .sort_values(ascending=False)
                    .to_dict()
                )

            # If enough remaining or too few to process more

            if has <= need:
                # images = valid_images_df['image_id'].values.tolist()
                # to_add = images[:need]
                # filter images based on people and query even if has less than need
                to_add =  valid_images_df.assign(_pid=valid_images_df["persons_ids"].apply(tuple)).sort_values("image_order", ascending=True).drop_duplicates(subset=["_pid", "image_subquery_content"], keep="first").head(need)["image_id"].tolist()

                ai_images_selected.extend(to_add)
                category_picked[cluster_name]['selected'] = category_picked[cluster_name].get('selected', 0) + len(to_add)
                logger.info(f"it has less than needed so we select them all {cluster_name} no filtering")
                continue


            # -------- Cluster-specific selection strategies --------
            elif cluster_name in ['bride getting dressed','getting hair-makeup']:
                    # Select only bride photos
                    filtered = color_candidates_df[color_candidates_df["image_subquery_content"].str.contains("bride", case=False, na=False)]
                    if len(filtered) > 0:
                        all_ids = [pid for sublist in filtered["persons_ids"] for pid in sublist]
                        most_common_id = Counter(all_ids).most_common(1)[0][0]
                        df = filtered[filtered["persons_ids"].apply(lambda ids: most_common_id in ids)]

                        if len(df) <= need or len(df) - need <= 1 :
                            logger.info(f"this cluster {cluster_name} has no enough images related to bride less than needed we select them all no filtering")
                            if no_selection:
                                preferred_color_ids  = df.sort_values(by='image_order', ascending=True)['image_id'].values.tolist()[:need]
                            else:
                                preferred_color_ids = df.sort_values(by='total_score', ascending=False)[
                                                          'image_id'].values.tolist()[:need]
                        else:
                            #preferred_color_ids  = filter_similarity(need, df.reset_index(), cluster_name)
                            # preferred_color_ids = filter_similarity_diverse(need=need,
                            #                                                 df=df.reset_index(),
                            #                                                 cluster_name=cluster_name, logger=logger,
                            #                                                 # df has: image_id, image_embedding, total_score, sub_group_time_cluster, image_oreintation (or image_orientation)
                            #                                                 target_group_size=10)

                            preferred_color_ids = select_remove_similar(need=need,
                                                         df=df.reset_index(),
                                                         cluster_name=cluster_name, logger=logger,
                                                         # df has: image_id, image_embedding, total_score, sub_group_time_cluster, image_oreintation (or image_orientation)
                                                         target_group_size=10)

                            # preferred_color_ids = filter_similarity_diverse_new(need=need,
                            #                                                 df=df.reset_index(),
                            #                                                 cluster_name=cluster_name, logger=logger,
                            #                                                 # df has: image_id, image_embedding, total_score, sub_group_time_cluster, image_oreintation (or image_orientation)
                            #                                                 target_group_size=10)

                            # set e.g. 3 to balance portrait/landscape)
                    else:
                        df = color_candidates_df
                        preferred_color_ids = df.sort_values(by='image_order', ascending=True)[
                            'image_id'].values.tolist()[:need]

            elif cluster_name in orientation_time_categories:
                # Cluster by time and find most solo person
                #df_with_time_cluster = cluster_by_time(valid_images_df, logger)
                original_pool = color_candidates_df.copy()
                filtered_df = color_candidates_df
                groom_id = int(filtered_df['groom_id'].to_list()[0])
                bride_id = int(filtered_df['bride_id'].to_list()[0])
                if cluster_name == 'bride':
                    filtered_df = filtered_df[
                        filtered_df['persons_ids'].apply(lambda x: x == [bride_id])
                    ]
                elif  cluster_name == 'groom':
                    if len(color_candidates_df) < need * 2:
                        filtered_df = filtered_df
                    else:
                        filtered_df = filtered_df[
                        filtered_df['persons_ids'].apply(lambda x: x == [groom_id])
                    ]

                elif cluster_name == "bride and groom":
                    # Filter images that contain both top two IDs or have 2 faces or 2 bodies
                    def has_both_ids_or_two_faces_bodies(row):
                        ids = {str(v) for v in row.get("persons_ids", [])}
                        target = {str(bride_id), str(groom_id)}
                        has_two_faces = row.get('n_faces', 0) == 2
                        has_two_bodies = row.get('number_bodies', 0) == 2
                        return  target.issubset(ids) and (has_two_faces or has_two_bodies)

                    filtered_df = filtered_df[
                        filtered_df.apply(has_both_ids_or_two_faces_bodies, axis=1)]

                elif cluster_name == 'bride party':
                     if bride_id:
                        def has_bride_in_photos(row):
                            return bride_id in row['persons_ids']

                        filtered_df = filtered_df[
                            filtered_df.apply(has_bride_in_photos, axis=1)]

                elif cluster_name == 'groom party':
                    if groom_id:
                        def has_groom_in_photos(row):
                            return groom_id in row['persons_ids']

                        filtered_df = filtered_df[
                            filtered_df.apply(has_groom_in_photos, axis=1)]

                remaining = len(filtered_df)

                if remaining == 0:
                    # Nothing left — bring back half of the *entire* original pool (rounded down).
                    bring_back = original_pool.head(len(original_pool) // 2)
                    filtered_df = bring_back.copy()
                else:
                    filtered_out_df = original_pool[~original_pool['image_id'].isin(filtered_df['image_id'])]
                    filtered_out = len(filtered_out_df)

                    # Condition: filtered out > 80% of what remains
                    if filtered_out > 0.8 * remaining:
                        n_to_add = remaining // 2  # "half of the filtering"
                        logger.info(
                            f"We took out more than 80% from this cluster {cluster_name} so we get {n_to_add} images back from filtering")
                        if n_to_add > 0:
                            # Choose which to bring back:
                            # Option A (deterministic, preserves original order)
                            add_back = filtered_out_df.head(n_to_add)
                            filtered_df = (
                                pd.concat([filtered_df, add_back], ignore_index=True)
                                .drop_duplicates(subset='image_id', keep='first')
                            )

                df_clustered  = filtered_df.set_index('image_id')

                if no_selection:
                    preferred_color_ids = df_clustered.sort_values(
                        'image_order', ascending=True
                    ).index.tolist()
                else:
                    preferred_color_ids = df_clustered.sort_values(
                        'total_score', ascending=False
                    ).index.tolist()

                if cluster_name == 'dancing' and need < len(preferred_color_ids):
                    df_clustered_reset = df_clustered.reset_index()
                    landscape_ids = df_clustered_reset[(df_clustered_reset['image_id'].isin(preferred_color_ids)) & (df_clustered_reset['image_orientation'] == 'landscape')]['image_id'].values.tolist()
                    if len(landscape_ids) < need:
                        non_landscape_df = df_clustered[~df_clustered.index.isin(landscape_ids)]
                        non_landscape_ids = non_landscape_df.index.tolist()

                        # Combine to reach the required number
                        selected_ids = landscape_ids + non_landscape_ids[:need - len(landscape_ids)]
                        preferred_color_ids = selected_ids
                        df_clustered = df_clustered.loc[preferred_color_ids]
                    else:
                        preferred_color_ids = landscape_ids
                        df_clustered = df_clustered.loc[preferred_color_ids]

                # Remove duplicates and finalize selection
                #preferred_color_ids  = filter_similarity(need, color_candidates_df.reset_index(),cluster_name)
                #
                preferred_color_ids = select_remove_similar(need=need,
                                        df=df_clustered.reset_index(),
                                        cluster_name=cluster_name,logger=logger ,# df has: image_id, image_embedding, total_score, sub_group_time_cluster, image_oreintation (or image_orientation)
                                        target_group_size=10)
                #
                # preferred_color_ids = filter_similarity_diverse_new(need=need,
                #                                                 df=df_clustered.reset_index(),
                #                                                 cluster_name=cluster_name, logger=logger,
                #                                                 # df has: image_id, image_embedding, total_score, sub_group_time_cluster, image_oreintation (or image_orientation)
                #                                                 target_group_size=10)


                # set e.g. 3 to balance portrait/landscape)

                if preferred_color_ids is None:
                    continue

            elif cluster_name in persons_categories:
                original_valid_df = color_candidates_df
                if cluster_name == 'portrait':
                    formal_query = ["formal studio-style wedding portrait, bride and groom centered, attendants standing still, bouquets held, symmetrical but there are no people behind them",
                                    "formal family portrait with bride in white dress and groom in suit AND these people are standing still AND these people are facing cameraAND these people are arranged in one or two rows"]

                    #color_candidates_df = color_candidates_df[color_candidates_df["image_subquery_content"].isin(formal_query)]
                    color_candidates_df = color_candidates_df[
                        (color_candidates_df["image_subquery_content"].isin(formal_query)) &
                        (color_candidates_df["persons_ids"].apply(
                            lambda x: isinstance(x, list) and bride_id in x and groom_id in x
                        ))]


                    if len(color_candidates_df) == 0:
                          color_candidates_df = original_valid_df
                    elif len(color_candidates_df) < need:
                         add_unformal = ["family with bride and groom group picture at night at the end of the wedding party","a group with people with bride and groom posing AND people behind them in background OR on the side of picture"]
                         #remaining_df = original_valid_df[~original_valid_df['image_subquery_content'].isin(formal_query)]
                         remaining_df = original_valid_df[
                        (original_valid_df["image_subquery_content"].isin(add_unformal)) &
                        (original_valid_df["persons_ids"].apply(
                            lambda x: isinstance(x, list) and bride_id in x and groom_id in x
                        ))]

                         if remaining_df.empty:
                             remaining_df = original_valid_df[
                                 (original_valid_df["persons_ids"].apply(
                                     lambda x: isinstance(x, list) and bride_id in x and groom_id in x
                                 ))]

                         color_candidates_df = pd.concat([color_candidates_df, remaining_df], ignore_index=True)

                color_candidates_df  = color_candidates_df.set_index('image_id')
                preferred_color_ids = color_candidates_df.sort_values(
                    'total_score', ascending=False
                ).index.tolist()

                preferred_color_ids = person_max_union_selection(
                    images_for_category=preferred_color_ids,
                    df=color_candidates_df.reset_index(),
                    needed_count=need,
                    image_cluster_dict=image_order_dict,
                    logger=logger
                )
            else:
                clusters_ids = get_clusters(color_candidates_df.reset_index())  # df was indexed earlier
                preferred_color_ids = select_non_similar_images(clusters_ids, image_order_dict, need)

            num_preferred_color = len(preferred_color_ids)
            final_color_selection = []
            final_grayscale_selection = []

            if num_preferred_color >= need:
                # Case A: Success with Color. We have enough.
                final_color_selection = preferred_color_ids[:need]  # Take the best 'need'
            else:
                # Case B: Insufficiency of Color. We need to fill the gap.
                final_color_selection = preferred_color_ids  # Take all preferred color images

                grayscale_need = need - num_preferred_color

                if grayscale_need > 0 and not grayscale_candidates_df.empty:
                    if len(grayscale_candidates_df) <= grayscale_need:
                        # Not enough grayscale to fill the need, take all available
                        final_grayscale_selection = grayscale_candidates_df.index.tolist()
                    elif grayscale_need == 1:
                        # Only need one, take the best
                        final_grayscale_selection = grayscale_candidates_df.sort_values(
                            'total_score', ascending=False
                        ).head(1).index.tolist()
                    else:
                        # Need multiple, so ensure they are diverse
                        # final_grayscale_selection = filter_similarity(
                        #     grayscale_need, grayscale_candidates_df.reset_index(), cluster_name
                        # )
                        # final_grayscale_selection = filter_similarity_diverse(need=grayscale_need,
                        #                                 df=grayscale_candidates_df.reset_index(),
                        #                                 cluster_name=cluster_name,logger=logger ,# df has: image_id, image_embedding, total_score, sub_group_time_cluster, image_oreintation (or image_orientation)
                        #                                 target_group_size=10)

                        final_grayscale_selection = select_remove_similar(need=grayscale_need,
                                                     df=grayscale_candidates_df.reset_index(),
                                                     cluster_name=cluster_name, logger=logger,
                                                     # df has: image_id, image_embedding, total_score, sub_group_time_cluster, image_oreintation (or image_orientation)
                                                     target_group_size=10)

                        # final_grayscale_selection = filter_similarity_diverse_new(need=grayscale_need,
                        #                                                       df=grayscale_candidates_df.reset_index(),
                        #                                                       cluster_name=cluster_name, logger=logger,
                        #                                                       # df has: image_id, image_embedding, total_score, sub_group_time_cluster, image_oreintation (or image_orientation)
                        #                                                       target_group_size=10)

            # --- Finalize selection for the cluster ---
            selected_ids = final_color_selection + final_grayscale_selection
            ai_images_selected.extend(selected_ids)
            category_picked[cluster_name]['selected'] = category_picked[cluster_name].get('selected', 0) + len(
                selected_ids)

        logger.info(f"Total images: {len(ai_images_selected)}")

    except Exception as e:
        logger.error(f"Error in smart_wedding_selection: {e}")
        return [] , f"Error in smart_wedding_selection: {e}"

    return ai_images_selected,spreads_allocation, error_message


# # Get scores for each image
# scores, scored_df = get_scores(cluster_df, user_selected_photos_df, people_ids, tags_features, logger)
#
# if scores is None:
#     continue
#
# # Skip cluster if all scores are zero or below threshold
# if all(score <= 0 for _, score in scores):
#     continue
#
# has = len(scored_df)
#
# candidates_images_scores = [(image_id, score) for image_id, score in scores
#                             if score > selection_threshold[cluster_name]]
#
# if len(candidates_images_scores) < need and len(candidates_images_scores) < has:
#     candidates_images_scores = []
#     for iter_idx, (index, row) in enumerate(scored_df.iterrows()):
#         if iter_idx < need:
#             candidates_images_scores.append((row['image_id'], row['total_score']))
#     # pass
# sorted_candidates_images_scores = sorted(candidates_images_scores, key=lambda x: x[1], reverse=True)
# available_img_ids = [image_id for image_id, _ in sorted_candidates_images_scores]

#
# grayscale_images = [img for img in images_filtered if df_clustered.at[img, 'image_color'] == 0]
#                 num_grayscale = len(grayscale_images)
#                 gray_needed = random.choice([1, 2]) if num_grayscale > CONFIGS['grays_scale_limit'] else (
#                     1 if grayscale_images else 0)
#
#                 # Select grayscale images
#                 selected_gray_image = []
#                 if gray_needed:
#                     gray_df = df_clustered.loc[grayscale_images].sort_values(by='total_score', ascending=False)
#                     selected_gray_image = gray_df.head(gray_needed).index.tolist()
#                     ai_images_selected.extend(selected_gray_image)
#                     need -= len(selected_gray_image)
#
#                 colored_images = [
#                     img for img in images_filtered
#                     if img not in grayscale_images and img not in selected_gray_image
#                 ]
#
#                 filtered_colored_df = df_clustered.loc[colored_images]
