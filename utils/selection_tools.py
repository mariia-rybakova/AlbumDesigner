import math
import random
import numpy as np

from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

from utils.parser import CONFIGS, limit_imgs, relations


def calculate_selection(category, n_actual, lookup_table):
    if category in lookup_table:
        n_target, std_target = lookup_table[category]

        # Scale selection proportionally with a weighted adjustment
        proportional_factor = min(1, n_target / n_actual)
        deviation_adjustment = (n_actual - n_target) / (std_target + 1e-6)
        selection = n_target + deviation_adjustment * proportional_factor

        # Ensure selection stays within reasonable bounds
        selection = max(4, min(selection, n_target * 1.5))
    else:
        # Default selection for unrecognized categories
        selection = 4

    # Selection can't exceed actual images
    return min(round(selection), n_actual)


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
    if 'persons_ids' in row_data.index:
        persons_in_image = row_data['persons_ids']
        person_exists = 0
        missed_person = 0
        for person in persons_in_image:
            if person in people_ids:
                person_exists += 1
            else:
                missed_person += 1
        person_score = person_exists / (len(persons_in_image) + 0.00000000001) * 2 + CONFIGS['person_score']
    else:
        person_score = CONFIGS['person_score']

    # 10 images similarity score
    if 'embedding' in row_data.index:
        similarity_scores = calculate_similarity_scores(row_data['embedding'].tolist(), selected_photos_df['embedding'].tolist())
        similarity_score = abs(similarity_scores.mean())
    else:
        similarity_score = CONFIGS['similarity_score']

    # class matching between 10 selected images and the intent image
    if 'image_class' in row_data.index:
        image_class = row_data['image_class']
        ten_photos_class = selected_photos_df['image_class'].values.tolist()
        class_match_counts = ten_photos_class.count(image_class)
        class_matching_score = class_match_counts / len(ten_photos_class) + CONFIGS['class_matching_zero_score']
    else:
        class_matching_score = CONFIGS['class_matching_penalty']

    if len(tags) !=0:
       tags_score = calcuate_tags_score(tags, row_data['embedding'])
    else:
        tags_score = 1

    total_score = class_matching_score * similarity_score * person_score * row_data[
        'image_order'] * tags_score

    return total_score



def images_scores_sorted(df, selected_photos_df, people_ids, tags_features):
    images_scores = {}
    for index,data in df.iterrows():
        image_id = data['image_id']
        image_score = calculate_scores(data,selected_photos_df, people_ids, tags_features)
        images_scores[image_id] = image_score

    # pick images based on relations percentage
    sorted_scores = sorted(images_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_scores

def calculate_proportional_allocation(group_sizes, needed_count, min_per_group=3):
    num_groups = len(group_sizes)
    minimums = [1 if size <= 3 else min_per_group for size in group_sizes]  # Apply the rule for small groups

    # Reduce the needed count by the mandatory minimums
    needed_count -= sum(minimums)

    # Adjust for cases where the needed count becomes zero or negative
    if needed_count <= 0:
        final_selection = [min(size, min_val) for size, min_val in zip(group_sizes, minimums)]
        return final_selection, sum(final_selection)

    # Calculate proportional allocation for the remaining slots
    remaining_slots = needed_count
    weights = [size / sum(group_sizes) for size in group_sizes]
    extra_images = [int(weight * remaining_slots) for weight in weights]

    # Distribute remaining slots to ensure the total matches
    while sum(extra_images) < remaining_slots:
        for i in sorted(range(num_groups), key=lambda x: -weights[x]):
            if sum(extra_images) < remaining_slots:
                extra_images[i] += 1
            else:
                break

    # Combine minimums and extra images, ensuring the final selection doesn't exceed group sizes
    final_selection = [
        min(size, min_val + extra)
        for size, min_val, extra in zip(group_sizes, minimums, extra_images)
    ]

    return final_selection, sum(final_selection)


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


def select_non_similar_images(event,clusters_ids,image_order_dict,needed_count):
    not_people_events = ['vehicle', 'settings', 'rings', 'entertainment', 'accessories', 'food',
                         'wedding dress', 'suit', 'pet','speech', 'cake cutting']
    result = []
    generators = {k: iter(v) for k, v in clusters_ids.items()}  # Create generators for each list

    while needed_count > 0:
        for key in generators:
            if needed_count <= 0:
                break
            if needed_count < len(clusters_ids):
                items_to_take = 1
            # Check list size to determine how many to take
            elif len(clusters_ids[key]) <= 5 or event in not_people_events:  # Small list: take at most 1
                items_to_take = min(1, needed_count)
            else:  # Large list: take up to the remaining needed count
                list_size = len(clusters_ids[key])
                items_to_take = round(list_size/ needed_count)

            images_ranked = sorted(clusters_ids[key], key=lambda img: image_order_dict.get(img, float('inf')), reverse=True)

            selected_items = images_ranked[:items_to_take]
            clusters_ids[key] = [item for item in clusters_ids[key] if item not in selected_items]

            # Add the selected items to the result and update remaining needed count
            result.extend(selected_items)
            needed_count -= len(selected_items)

    return result

def get_clusters(df):
    clusters_ids = {}
    # image class
    for index,data in df.iterrows():
        class_id = data['cluster_label']
        image_id = data['image_id']
        if class_id not in clusters_ids:
            clusters_ids[class_id] = []

        clusters_ids[class_id].append(image_id)
    return clusters_ids

def jaccard_distance(list1, list2):
    set1, set2 = set(list1), set(list2)
    return 1 - len(set1 & set2) / len(set1 | set2)

def select_by_cluster(clusters_ids,image_order_dict):
    final_selected_images = []

    for cluster_label, images_in_cluster in clusters_ids.items():

        # if i have 4 images i choose one of them if more then i choose more
        if len(images_in_cluster) <= 3:
            # Select image with highest 'order_score' in the cluster
            best_image = min(images_in_cluster, key=lambda img: image_order_dict.get(img, float('inf')))
            final_selected_images.append(best_image)
        else:
            n = round(len(images_in_cluster) / 4)

            images_order = sorted(images_in_cluster,
                                  key=lambda img: image_order_dict.get(img, float('inf')), reverse=True)
            final_selected_images.append(images_order[n])

    return final_selected_images


def select_by_person(clusters_ids,images_list,df,image_cluster_dict):
    result = []
    persons_ids = []

    for image_id in images_list:
        img_ifo_df = df[df['image_id'] == image_id]
        if 'persons_ids' in img_ifo_df.columns:
            persons_ids.extend(img_ifo_df['persons_ids'].values.tolist())
        else:
            persons_ids.extend([])

    if len(set(item for sublist in persons_ids for item in sublist)) > 1:
        mlb = MultiLabelBinarizer()
        binary_matrix = mlb.fit_transform(persons_ids)  # Binary matrix

        # Compute pairwise distances
        dist_matrix = squareform(pdist(binary_matrix, metric=jaccard_distance))

        if len(dist_matrix) == 0 or dist_matrix.shape[0] < 2:
            # if no people found then select using Clusters labels
            result = select_by_cluster(clusters_ids,image_cluster_dict)
        else:
            # Perform Agglomerative Clustering based on person similarity
            person_clustering = AgglomerativeClustering(
                n_clusters=None,
                metric='precomputed',
                linkage='complete',
                distance_threshold=0.3,  # Adjust this threshold as needed
            )
            person_clustering.fit(dist_matrix)

            # Dictionary to hold images in each person-based cluster
            person_clusters = {}
            for idx, label in enumerate(person_clustering.labels_):
                person_clusters.setdefault(label, []).append(images_list[idx])

            # Select the best image in each cluster based on 'order_score'
            for cluster_label, images_in_cluster in person_clusters.items():
                # if i have 4 images i choose one of them if more then i choose more
                if len(images_in_cluster) <= 4:
                    # Select image with highest 'order_score' in the cluster
                    best_image = min(images_in_cluster, key=lambda img:  df.set_index('image_id').loc[img, 'image_order'])
                    result.append(best_image)
                else:
                    n = round(len(images_in_cluster) / 4)
                    images_order = sorted(images_in_cluster, key=lambda img: df.set_index('image_id').loc[img, 'image_order'], reverse=True)
                    result.extend(images_order[:n])

    return result


def select_by_time(needed_count, selected_images, df):
    image_id_time_mapping = []
    for image_id in selected_images:
        time_integer =  df.set_index('image_id').loc[image_id, 'image_time']
        correct_time =  datetime.fromtimestamp(time_integer)
        image_id_time_mapping.append((image_id, correct_time))

    images_time = [time_image  for _, time_image in image_id_time_mapping]
    general_times = process_time(images_time)

    db = DBSCAN(eps=50, min_samples=2).fit(np.array(general_times).reshape(-1, 1))

    labels = db.labels_
    cluster_mapping = {image_id: label for (image_id, _), label in zip(image_id_time_mapping, labels)}

    n_clusters = len(set(labels) - {-1})
    clustered_images = {label: [] for label in range(n_clusters)}

    for image_id, label in cluster_mapping.items():
        if label == -1:
            continue
        clustered_images[label].append(image_id)

    clusters_time_id = {}
    for key in list(clustered_images.keys()):
        if key not in clusters_time_id:
            clusters_time_id[key] = {}

        for im_id in clustered_images[key]:
            cluster_id = df.set_index('image_id').loc[im_id, 'cluster_label']
            if cluster_id not in clusters_time_id[key]:
                clusters_time_id[key][cluster_id] = []
            clusters_time_id[key][cluster_id].append(im_id)

    group_sizes = [sum(len(images) for images in content_clusters.values()) for content_clusters in
                   clusters_time_id.values()]

    allocation, needed_count = calculate_proportional_allocation(group_sizes, needed_count)

    # Step 2: Ensure at least 3 images from each time cluster
    result = []
    for (time_key, content_clusters), max_take in zip(clusters_time_id.items(), allocation):
        selected_images = []

        # Select up to `max_take` images from this time cluster
        while len(selected_images) < max_take and len(result) < needed_count:
            for content_key, images in content_clusters.items():
                images_sorted = sorted(images, key=lambda img: df.set_index('image_id').loc[img, 'image_order'], reverse=True)
                if images_sorted:
                    selected_images.append(images_sorted.pop(0))
                    clusters_time_id[time_key][content_key] = images_sorted
                    if len(selected_images) == max_take:
                        break

        result.extend(selected_images)
        if len(result) == needed_count:
            break

    return result


def remove_similar_images(category, selected_images, df, user_relation,image_cluster_dict):
    persons_categories = ['portrait', 'very large group', 'speech', 'walking the aisle']
    time_categories = ['bride', 'groom', 'bride and groom', 'bride party', 'groom party']

    if len(selected_images) == 1:
        return selected_images

    final_selected_images = []
    clusters_ids = get_clusters(df)

    # Identify grayscale images in the filtered list
    grayscale_images = [img for img in selected_images if
                        df.set_index('image_id').loc[img, 'image_color'] == 0]
    num_grayscale = len(grayscale_images)

    if num_grayscale > CONFIGS['grays_scale_limit']:
        selected_gray_image = random.choice(grayscale_images)
        selected_images = [image for image in selected_images if image not in grayscale_images]
        # Remove other grayscale images except the selected one
        final_selected_images.append(selected_gray_image)

    needed_count = calculate_selection(category, len(selected_images), relations[user_relation])

    if needed_count == len(selected_images):
        chosen_images = selected_images
    elif category in persons_categories:
        # Select images using persons clustering
        chosen_images = select_by_person(clusters_ids, selected_images, df,image_cluster_dict)
    elif category in time_categories:
        # Select based on time clustering, then cluster label
        chosen_images = select_by_time(needed_count, selected_images, df)
    else:
        # Select by Clusters label
        chosen_images = select_non_similar_images(category, clusters_ids, df, needed_count)

    final_selected_images.extend(chosen_images)
    # best_image = max(images_in_cluster, key=lambda img: gallery_photos_info[img]['image_order'])

    return final_selected_images


def select_random_image(images):
    if not images:
        return None
    options = [0, len(images) // 2, -1]  # Indices for top, middle, and last
    selected_index = random.choice(options)
    return images.pop(selected_index if selected_index != -1 else len(images) - 1)


def calculate_centroid(embeddings):
    if not embeddings:
        return None
    return np.mean(embeddings, axis=0)


def select_n_images(images, n, gallery_photos_info):
    """
    Select N images from the list based on cluster label diversity.
    Prioritize images from the beginning, middle, and end of the cluster label list.
    """
    if len(images) <= n:
        return images  # Return all if fewer images than required

    # Create a dictionary of cluster labels to image indices
    cluster_dict = {}
    for idx, image in enumerate(images):
        if 'cluster_label' in gallery_photos_info[image]:
            cluster_label = gallery_photos_info[image]['cluster_label']
            cluster_dict.setdefault(cluster_label, []).append(idx)

    # Flatten the cluster dictionary into a list of indices, grouped by cluster labels
    cluster_indices = [idx for indices in cluster_dict.values() for idx in indices]

    # Initialize selected indices and clusters
    selected_indices = set()
    selected_clusters = set()
    total_indices = len(cluster_indices)

    # Helper function to check cluster membership
    def is_unique_cluster(idx):
        cluster_label = gallery_photos_info[images[idx]]['cluster_label']
        return cluster_label not in selected_clusters

    # Select first, middle, and last indices if they belong to unique clusters
    if total_indices >= 1:
        idx = cluster_indices[0]  # First index
        if is_unique_cluster(idx):
            selected_indices.add(idx)
            selected_clusters.add(gallery_photos_info[images[idx]]['cluster_label'])

    if total_indices >= 2:
        idx = cluster_indices[total_indices // 2]  # Middle index
        if is_unique_cluster(idx):
            selected_indices.add(idx)
            selected_clusters.add(gallery_photos_info[images[idx]]['cluster_label'])

    if total_indices >= 3:
        idx = cluster_indices[-1]  # Last index
        if is_unique_cluster(idx):
            selected_indices.add(idx)
            selected_clusters.add(gallery_photos_info[images[idx]]['cluster_label'])

    # Add more indices if needed to reach the required count
    for idx in cluster_indices:
        if len(selected_indices) >= n:
            break
        if is_unique_cluster(idx):  # Ensure cluster label uniqueness
            selected_indices.add(idx)
            selected_clusters.add(gallery_photos_info[images[idx]]['cluster_label'])

    # Select images based on the chosen indices
    selected_images = [images[idx] for idx in selected_indices]
    return selected_images


def calculate_required_images(total_images, min_limit=10, max_limit=30, max_total_images=150):
    # Ensure required images are between min_limit and max_limit
    required_images = min_limit + ((max_limit - min_limit) / max_total_images) * total_images
    return max(min_limit, min(max_limit, round(required_images)))


def proportional_selection_with_calculation(person_images, clusters_class_imgs, gallery_photos_info,logger,max_total_images=150, min_limit=10, max_limit=30):
    # Calculate the total number of images
    total_images = sum(len(images) for images in person_images.values())

    # Calculate the required images
    required_images = calculate_required_images(total_images, min_limit, max_limit, max_total_images)

    # Main logic with the updated function
    # Proportional allocation
    quotas = {
        cls: max(1, math.ceil((len(images) / total_images) * required_images))
        for cls, images in person_images.items()
    }

    selected_images_dict = {}
    selected_images = []
    fallback_clusters = set()  # Track clusters for fallback

    while len(selected_images) < required_images:
        progress_made = False  # Track if any image was selected in this round

        for cls, images in person_images.items():
            if quotas[cls] > 0 and images:
                # Select N images using the updated function
                n = quotas[cls]
                n_images = select_n_images(images, n, gallery_photos_info)

                for selected_image in n_images:
                    selected_image_class = gallery_photos_info[selected_image]['cluster_label']
                    image_color = gallery_photos_info[selected_image]['image_color']

                    # Avoid grayscale duplicates of the same color
                    if image_color == 0 and any(
                            gallery_photos_info[img]['image_color'] == 1 and
                            gallery_photos_info[img]['cluster_label'] == selected_image_class
                            for img in selected_images
                    ):
                        continue

                    # Try to add the image if the class isn't already selected
                    if selected_image_class not in selected_images_dict:
                        selected_images_dict[selected_image_class] = []
                        selected_images_dict[selected_image_class].append(selected_image)
                        selected_images.append(selected_image)
                        quotas[cls] -= 1
                        progress_made = True

                    # Check if required images count is reached
                    if len(selected_images) >= required_images:
                        break

        # Exit if no progress is made and there are no more images to select
        if not progress_made:
            for fallback_class in fallback_clusters:
                cluster_images = clusters_class_imgs.get(fallback_class, [])
                if not cluster_images:
                    continue

                selected_embeddings = [
                    gallery_photos_info[img]['embedding']
                    for img in selected_images_dict.get(fallback_class, [])
                ]
                centroid = calculate_centroid(selected_embeddings)

                farthest_image = None
                max_distance = -1
                for candidate_image in cluster_images:
                    if candidate_image in selected_images:
                        continue
                    if 'embedding' not in gallery_photos_info[candidate_image]:
                        continue

                    candidate_embedding = gallery_photos_info[candidate_image]['embedding']
                    distance = cosine(candidate_embedding, centroid)
                    if distance > max_distance:
                        max_distance = distance
                        farthest_image = candidate_image

                if farthest_image:
                    selected_images.append(farthest_image)
                    selected_images_dict.setdefault(fallback_class, []).append(farthest_image)

                    if 'persons_ids' in gallery_photos_info[farthest_image]:
                        n_person = len(gallery_photos_info[farthest_image]['persons_ids'])
                    else:
                        continue

                    person_class = (
                        f'person_1_{gallery_photos_info[farthest_image]["persons_ids"][0]}'
                        if n_person == 1 else n_person
                    )
                    quotas[person_class] -= 1
                    cluster_images.remove(farthest_image)
                    progress_made = True

                if len(selected_images) >= required_images:
                    break

        if not progress_made:
            logger.warning("Could not reach the required number of images.")
            break

    return selected_images


def is_subset(smaller, larger):
    return all(x in larger for x in smaller)


def get_appearance_percentage(persons_dict, total_images):
    percentage_dict = {}
    for person in persons_dict:
        percentage = len(persons_dict[person]) / total_images
        percentage_dict[person] = percentage

    return percentage_dict


def select_images_of_group(people_clustering_dict, photos_dict, clusters_class_imgs,logger):
    # remove one image group which has portrait image since we don't have a layout for this
    pple_images_related = {
        key: v for key, v in people_clustering_dict.items()
        if not (len(v) == 1 and photos_dict[v[0]]['image_orientation'] == 'portrait')
    }

    selected = proportional_selection_with_calculation(pple_images_related, clusters_class_imgs,
                                                       photos_dict,logger)
    return selected


def select_images_of_one_person(related_images, photos_dict,logger):
    # Calculate the total number of images
    total_images = len(related_images)

    # Calculate the required images
    required_images = calculate_required_images(total_images, min_limit=10, max_limit=30, max_total_images=150)

    selected_images_dict = dict()
    selected_images = []
    fallback_clusters = set()

    while len(selected_images) < required_images:
        if not related_images:  # Break if there are no more images to select
            logger("No more images to select.")
            break

        # Select a random image
        selected_image = select_random_image(related_images)
        if not selected_image:
            continue

        selected_image_class = photos_dict[selected_image]['cluster_label']
        image_color = photos_dict[selected_image]['image_color']

        # Avoid selecting grayscale images of the same color image already selected
        if image_color == 0 and any(
                photos_dict[img]['image_color'] == 1 and
                photos_dict[img]['cluster_label'] == selected_image_class
                for img in selected_images
        ):
            continue

        # Try to add the image if the class isn't already selected
        if selected_image_class not in selected_images_dict:
            selected_images_dict[selected_image_class] = []
            selected_images_dict[selected_image_class].append(selected_image)
            selected_images.append(selected_image)
        else:
            # If we can't use this cluster now, mark it as fallback
            fallback_clusters.add(selected_image_class)
            continue

        # Check if required images count is reached
        if len(selected_images) >= required_images:
            break

    # Handle fallback case: Select the farthest images if not enough are selected
    if len(selected_images) < required_images:
        for fallback_class in fallback_clusters:
            fallback_images = [
                img for img in related_images
                if photos_dict[img]['cluster_label'] == fallback_class
            ]

            if not fallback_images:
                continue

            # Calculate the centroid of already selected images for this class
            selected_embeddings = [
                photos_dict[img]['embedding']
                for img in selected_images_dict.get(fallback_class, [])
                if 'embedding' in photos_dict[img]
            ]
            if not selected_embeddings:
                continue

            centroid = calculate_centroid(selected_embeddings)

            # Find the farthest image from the centroid
            farthest_image = None
            max_distance = -1
            for candidate_image in fallback_images:
                if candidate_image in selected_images:
                    continue
                if 'embedding' not in photos_dict[candidate_image]:
                    continue

                candidate_embedding = photos_dict[candidate_image]['embedding']
                distance = cosine(candidate_embedding, centroid)
                if distance > max_distance:
                    max_distance = distance
                    farthest_image = candidate_image

            # Select the farthest image
            if farthest_image:
                selected_images.append(farthest_image)
                selected_images_dict[fallback_class].append(farthest_image)
                related_images.remove(farthest_image)  # Remove from the pool

            # Break if we've selected enough images
            if len(selected_images) >= required_images:
                break

    # Final validation
    if len(selected_images) < required_images:
        logger.warning("Could not reach the required number of images.")

    return selected_images


def generate_dict_key(numbers, n_bodies):
    if numbers == 0 and n_bodies == 0 or not numbers:
        return 'No PEOPLE'

    # Convert the string of numbers into a list
    try:
        id_list = eval(numbers) if isinstance(numbers, str) else numbers
    except:
        return "Invalid_numbers"

    # Calculate the count based on the list length or n_bodies
    count = max(len(id_list), n_bodies) if isinstance(id_list, list) else n_bodies

    # Determine the suffix
    suffix = "person" if count == 1 else "pple"

    # Combine count, suffix, and the numbers joined by underscores
    key = f"{count}_{suffix}_" + "_".join(map(str, id_list))
    return key


def smart_non_wedding_selection(df, logger):
    # convert df to dict
    if df is None:
        logger.info("the dataframe is empty! for non wedding selection")
        return None, "Error fetching the data"

    total_images = len(df)

    if total_images <= CONFIGS['small_gallery_number']:
        selected_images = df['image_id'].values.tolist()
        return selected_images, None

    df['people_cluster'] = df.apply(lambda row: generate_dict_key(row['persons_ids'], row['number_bodies']), axis=1)

    photos_info_dict = df.set_index('image_id').to_dict(orient='index')
    images_ids = list(photos_info_dict.keys())
    clusters_class_imgs = df.groupby("cluster_label")["image_id"].apply(list).to_dict()
    persons_images_clustering = df.groupby('people_cluster')['image_id'].apply(list).to_dict()

    persons_percentage = get_appearance_percentage(persons_images_clustering, total_images)

    # cover image selection for non wedding gallery
    count_percentage = 0

    for percent_person in persons_percentage.keys():
        if persons_percentage[percent_person] >= CONFIGS['person_count_percentage']:
            count_percentage += 1

    if count_percentage == 1:
        # one person gallery
        auto_selected_images = select_images_of_one_person(images_ids, photos_info_dict,logger)
    else:
        # more than one person gallery
        auto_selected_images = select_images_of_group(persons_images_clustering, photos_info_dict, clusters_class_imgs,logger)

    return auto_selected_images, None


def calculate_selection(category, n_actual, lookup_table):
    if category in lookup_table:
        n_target, std_target = lookup_table[category]

        # Scale selection proportionally with a weighted adjustment
        proportional_factor = min(1, n_target / n_actual)
        deviation_adjustment = (n_actual - n_target) / (std_target + 1e-6)
        selection = n_target + deviation_adjustment * proportional_factor

        # Ensure selection stays within reasonable bounds
        selection = max(4, min(selection, n_target * 1.5))
    else:
        # Default selection for unrecognized categories
        selection = 4

    # Selection can't exceed actual images
    return min(round(selection), n_actual)


def smart_wedding_selection(df, selected_photos, people_ids, user_relation, tags_features,
                            logger):

    logger.info("====================================")
    logger.info("Starting Image selection Process....")

    error_message = None
    ai_images_selected = []
    category_picked = {}

    selected_photos_df = df[df['image_id'].isin(selected_photos)]

    for iteration, (cluster_name, cluster_df) in enumerate(df.groupby('cluster_context')):
        n_actual = len(cluster_df)

        logger.info("====================================")
        logger.info(f"Starting with {cluster_name} and actual number  of images {n_actual}")

        if cluster_name not in category_picked:
            category_picked[cluster_name] = []

        image_order_dict = cluster_df.set_index('image_id')['image_order'].to_dict()

        # we don't select images from None, other and couples we ignore couples cause its before wedding
        if cluster_name == 'None' or cluster_name == 'other' or cluster_name == 'couple':
            continue
        # if we have 4 images for event we choose them all
        elif n_actual < CONFIGS['small_groups'] and cluster_name not in CONFIGS['events_disallowing_small_images']:
            continue
        elif cluster_name == 'accessories':
            # we select none similar accessories through clusters ids with no need to calculate the scores
            clusters_ids = get_clusters(cluster_df)
            chosen_images = select_non_similar_images(cluster_name, clusters_ids, cluster_df, 2)
            images_ranked = sorted(chosen_images, key=lambda img: image_order_dict.get(img, float('inf')), reverse=True)

            ai_images_selected.extend(images_ranked)
            category_picked[cluster_name].extend(images_ranked)

        else:
            # Get scores for each image
            scores = images_scores_sorted(cluster_df, selected_photos_df, people_ids, tags_features)

            # if all scores are zeros don't select anything
            if all(t[1] <= 0 for t in scores):
                continue

            # Get images that have people we want and ignore images with 0 score
            available_images_scores = [score for score in scores if score[1] > 0]
            available_img_ids = [image_id for image_id, score in available_images_scores]

            # remove similar before choosing from them
            available_img_ids = remove_similar_images(cluster_name, available_img_ids, cluster_df, user_relation,
                                                      image_order_dict)

            if cluster_name == 'wedding dress' or cluster_name == 'rings':
                ai_images_selected.extend(available_img_ids[:1])
                category_picked[cluster_name].extend(available_img_ids[:1])
            elif len(available_img_ids) < CONFIGS['small_groups_not_to_select']:
                # No less than 3 images for any event
                continue
            else:
                images_ranked = sorted(available_img_ids, key=lambda img: image_order_dict.get(img, float('inf')),
                                       reverse=True)
                ai_images_selected.extend(images_ranked)
                category_picked[cluster_name].extend(images_ranked)

    # limit total number of images
    LIMIT = CONFIGS['max_number_images']
    if len(ai_images_selected) == 0:
        error_message = 'No images were selected.'
        logger.error("No images were selected.")
    elif len(ai_images_selected) > LIMIT:
        deleted_images = []
        total_to_reduce = len(ai_images_selected) - LIMIT

        for group, images in category_picked.items():
            if len(images) == 0:
                continue
            if len(images) > limit_imgs[group]:
                to_reduce = len(images) - limit_imgs[group]
                category_picked[group] = images[:-to_reduce]
                deleted_images.extend(images[-to_reduce:])
                total_to_reduce -= to_reduce

        if total_to_reduce != 0:
            # if we still need to cut more images then we take the largest group and cut from it
            largest_group = max(category_picked.keys(), key=lambda x: len(category_picked[x]))
            if len(category_picked[largest_group]) - total_to_reduce > 5:
                category_picked[largest_group] = category_picked[largest_group][:-total_to_reduce]
                deleted_images.extend(category_picked[largest_group][-total_to_reduce:])

        # remove the images from chosen images
        ai_images_selected = list(filter(lambda img: img not in deleted_images, ai_images_selected))

    logger.info(f"Total images: {len(ai_images_selected)}")
    logger.info("*******************************************************")
    return ai_images_selected, error_message



