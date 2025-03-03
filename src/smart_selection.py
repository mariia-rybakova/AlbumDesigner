import os
import time
import shutil
import random
import numpy as np

from datetime import datetime

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform

from utils import image_clustering
from utils.image_selection_scores import map_cluster_label, calculate_scores
from utils.read_files_types import read_pkl_file
from utils.user_relation_percentage import relations_2
from utils.selection_limintation import limit_imgs



def generate_event_pdfs(events_dict, gal_path, output_folder):
    """
    Generate a PDF for each event, organizing its images in a grid format.

    Args:
        events_dict (dict): A dictionary where keys are event names, and values are lists of image IDs.
        gal_path (str): Path to the folder containing the images.
        output_folder (str): The folder to save the generated PDFs.
    """
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

    for event, image_ids in events_dict.items():
        if len(image_ids) == 0:
            continue

        target_folder = os.path.join(output_folder,event)
        os.makedirs(target_folder, exist_ok=True)

        for i, img_id in enumerate(image_ids):
            img_path = os.path.join(gal_path, f"{img_id}.jpg")
            destination_path = os.path.join(target_folder , f"{img_id}.jpg")
            shutil.copy(img_path, destination_path)


def jaccard_distance(list1, list2):
    set1, set2 = set(list1), set(list2)
    return 1 - len(set1 & set2) / len(set1 | set2)

def get_clusters(selected_images,gallery_photos_info):
    clusters_ids = {}
    # image class
    for image_id in selected_images:
        class_id = gallery_photos_info[image_id]['cluster_label']
        if class_id not in clusters_ids:
            clusters_ids[class_id] = []

        clusters_ids[class_id].append(image_id)
    return clusters_ids


def select_by_cluster(clusters_ids,gallery_photos_info):
    final_selected_images = []

    for cluster_label, images_in_cluster in clusters_ids.items():
        # if i have 4 images i choose one of them if more then i choose more
        if len(images_in_cluster) <= 3:
            # Select image with highest 'order_score' in the cluster
            best_image = min(images_in_cluster, key=lambda img: gallery_photos_info[img]['image_order'])
            final_selected_images.append(best_image)
        else:
            n = round(len(images_in_cluster) / 4)
            images_order = sorted(images_in_cluster,
                                  key=lambda img: gallery_photos_info[img]['image_order'], reverse=True)
            final_selected_images.append(images_order[n])

    return final_selected_images


def select_non_similar_images(event,clusters_ids,gallery_photos_info,needed_count):
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

            images_ranked = sorted(clusters_ids[key], key=lambda img: gallery_photos_info[img]['image_order'], reverse=True)

            selected_items = images_ranked[:items_to_take]
            clusters_ids[key] = [item for item in clusters_ids[key] if item not in selected_items]

            # Add the selected items to the result and update remaining needed count
            result.extend(selected_items)
            needed_count -= len(selected_items)

    return result

def select_by_person(clusters_ids,images_list,gallery_photos_info):
    result = []
    persons_ids = []

    for image_id in images_list:
        if 'persons_ids' in gallery_photos_info[image_id]:
            persons_ids.append(gallery_photos_info[image_id]['persons_ids'])
        else:
            persons_ids.append([])

    if len(set(item for sublist in persons_ids for item in sublist)) > 1:
        mlb = MultiLabelBinarizer()
        binary_matrix = mlb.fit_transform(persons_ids)  # Binary matrix

        # Compute pairwise distances
        dist_matrix = squareform(pdist(binary_matrix, metric=jaccard_distance))

        if len(dist_matrix) == 0 or dist_matrix.shape[0] < 2:
            # if no people found then select using Clusters labels
            result = select_by_cluster(clusters_ids,gallery_photos_info)
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
                    best_image = min(images_in_cluster, key=lambda img: gallery_photos_info[img]['image_order'])
                    result.append(best_image)
                else:
                    n = round(len(images_in_cluster) / 4)
                    images_order = sorted(images_in_cluster, key=lambda img: gallery_photos_info[img]['image_order'], reverse=True)
                    result.extend(images_order[:n])

    return result

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


def select_by_time(needed_count, selected_images, gallery_photos_info,DEBUG):
    image_id_time_mapping = []
    for image_id in selected_images:
        time_integer = gallery_photos_info[image_id]['image_time']
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

    if DEBUG:
        gallery_id = 41661791
        gal_path = fr'dataset\newest_wedding_galleries\{gallery_id}'
        for cluster_id, images in clustered_images.items():
            os.makedirs(rf'\time_grouping\{cluster_id}', exist_ok=True)
            for image in images:
                image_path = os.path.join(gal_path,  f'{image}.jpg')
                shutil.copy(image_path, os.path.join(rf'\time_grouping\{str(cluster_id)}', str(image) + '.jpg'))

    clusters_time_id = {}
    for key in list(clustered_images.keys()):
        if key not in clusters_time_id:
            clusters_time_id[key] = {}

        for im_id in clustered_images[key]:
            cluster_id = gallery_photos_info[im_id]['cluster_label']
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
                images_sorted = sorted(images, key=lambda img: gallery_photos_info[img]['image_order'], reverse=True)
                if images_sorted:
                    selected_images.append(images_sorted.pop(0))
                    clusters_time_id[time_key][content_key] = images_sorted
                    if len(selected_images) == max_take:
                        break

        result.extend(selected_images)
        if len(result) == needed_count:
            break

    return result


def remove_similar_images(category, selected_images, gallery_photos_info,user_relation, DEBUG):
        persons_categories = ['portrait','very large group','speech', 'walking the aisle' ]
        time_categories = ['bride', 'groom', 'bride and groom', 'bride party', 'groom party']

        if len(selected_images) == 1:
            return selected_images

        final_selected_images = []
        clusters_ids = get_clusters(selected_images, gallery_photos_info)

        # Identify grayscale images in the filtered list
        grayscale_images = [img for img in selected_images if
                            gallery_photos_info[img]['image_color'] == 0]
        num_grayscale = len(grayscale_images)

        if num_grayscale > 1:
            selected_gray_image = random.choice(grayscale_images)
            selected_images = [image for image in selected_images if image not in grayscale_images]
            # Remove other grayscale images except the selected one
            final_selected_images.append(selected_gray_image)

        needed_count =  calculate_selection(category, len(selected_images), relations_2[user_relation])

        if needed_count == len(selected_images):
            chosen_images = selected_images
        elif category in persons_categories :
            # Select images using persons clustering
            chosen_images = select_by_person(clusters_ids,selected_images,gallery_photos_info)
        elif category in time_categories:
            # Select based on time clustering, then cluster label
            chosen_images = select_by_time(needed_count, selected_images, gallery_photos_info,DEBUG)
        else:
            # Select by Clusters label
            chosen_images = select_non_similar_images(category,clusters_ids,gallery_photos_info, needed_count)

        final_selected_images.extend(chosen_images)
        #best_image = max(images_in_cluster, key=lambda img: gallery_photos_info[img]['image_order'])

        return final_selected_images

def images_scores_sorted(imges,gallery_photos_info, ten_photos, people_ids, tags_features):
    images_scores = {}
    for image in imges:
        image_score = calculate_scores(image, gallery_photos_info, ten_photos, people_ids, tags_features)
        images_scores[image] = image_score

    # pick images based on relations percentage
    sorted_scores = sorted(images_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_scores


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


def select_images(clusters_class_imgs, gallery_photos_info, ten_photos, people_ids, tags_features, user_relation,DEBUG,
                  logger=None):
    if logger is not None:
        logger.info("====================================")
        logger.info("Starting Image selection Process....")

    error_message = None
    ai_images_selected = []
    category_picked = {}

    for iteration, (category, imges) in enumerate(clusters_class_imgs.items()):
        n_actual = len(imges)
        not_allowed_small_events = ['settings','vehicle','rings','food', 'accessories', 'entertainment', 'dancing', 'wedding dress', 'kiss']

        if logger is not None:
            logger.info("====================================")
            logger.info(f"Starting with {category} and actual number  of images {n_actual}")
        else:
            print(f"Starting with {category} and actual number of images  {n_actual}")


        if category not in category_picked:
            category_picked[category] = []

        # we don't select images from them
        if category == 'None' or category == 'other' or category == 'couple':
                    continue
        # if we have 4 images for event we choose them all
        elif n_actual < 3 and category not in not_allowed_small_events:
            continue
        elif category == 'accessories':
            # we select none similar accessories through clusters ids with no need to calculate the scores
            clusters_ids = get_clusters(imges, gallery_photos_info)
            chosen_images = select_non_similar_images(category, clusters_ids, gallery_photos_info, 2)
            images_ranked = sorted(chosen_images, key=lambda img: gallery_photos_info[img]['image_order'], reverse=True)

            ai_images_selected.extend(images_ranked)
            category_picked[category].extend(images_ranked)

        else:
            # Get scores for each image
            scores = images_scores_sorted(imges,gallery_photos_info, ten_photos, people_ids, tags_features)

            # if all scores are zeros don't select anything
            if all(t[1] <= 0 for t in scores):
                continue
            else:
                # Get images that have people we want and ignore images with 0 score
                available_images_scores = [score for score in scores if score[1] > 0]
                available_img_ids = [image_id for image_id, score in available_images_scores]

                # remove similar before choosing from them
                available_img_ids = remove_similar_images(category,available_img_ids , gallery_photos_info,user_relation,DEBUG)

                if category == 'wedding dress' or category == 'rings':
                    ai_images_selected.extend(available_img_ids[:1])
                    category_picked[category].extend(available_img_ids[:1])
                elif len(available_img_ids) < 3:
                    # No less than 3 images for any event
                    continue
                else:
                    images_ranked = sorted(available_img_ids, key=lambda img: gallery_photos_info[img]['image_order'], reverse=True)
                    ai_images_selected.extend(images_ranked)
                    category_picked[category].extend(images_ranked)

    # limit total number of images
    LIMIT = 130
    if len(ai_images_selected) == 0:
        error_message = 'No images were selected.'
        if logger is not None:
           logger.error("No images were selected.")
    elif len(ai_images_selected) > LIMIT :
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
             if len(category_picked[largest_group]) - total_to_reduce > 5 :
                 category_picked[largest_group] = category_picked[largest_group][:-total_to_reduce]
                 deleted_images.extend(category_picked[largest_group][-total_to_reduce:])

        # remove the images from chosen images
        ai_images_selected = list(filter(lambda img: img not in deleted_images, ai_images_selected))

    if logger is not None:
        logger.info(f"Total images: {len(ai_images_selected)}")
        logger.info("*******************************************************")
        for category, images in category_picked.items():
            logger.info("category", category, 'Number of selected images', len(images))

    else:
        print(f"Total images: {len(ai_images_selected)}")
        print("*******************************************************")
        for category, images in category_picked.items():
            print("category", category, 'Number of selected images', len(images))

    return ai_images_selected, gallery_photos_info, error_message


def auto_selection(project_base_url, ten_photos, tags_selected, people_ids, relation, queries_file,tags_features_file,DEBUG, logger):
    if gallery_photos_info is None:
        if logger is not None:
            logger.error('the gallery images dict is empty ')
        return None, None

    # Group images by cluster class labels
    clusters_class_imgs = {}
    # Get images group clusters class labels
    for im_id, img_info in gallery_photos_info.items():
        if 'cluster_class' not in img_info:
            if logger is not None:
                logger.warning("image id {} has no cluster_class we will ignore it! in auto selection".format(im_id))
            continue
        cluster_class = img_info['cluster_class']
        cluster_class_label = map_cluster_label(cluster_class)
        if cluster_class_label not in clusters_class_imgs:
            clusters_class_imgs[cluster_class_label] = []
        clusters_class_imgs[cluster_class_label].append(im_id)

    tags_features = read_pkl_file(tags_features_file)

    selected_tags_features = {}
    for tag in tags_selected:
        selected_tags_features[tag] = tags_features[tag]

    return select_images(clusters_class_imgs, gallery_photos_info, ten_photos, people_ids, selected_tags_features, relation,DEBUG,
                         logger)



# if __name__ == '__main__':
#     ten_photos = [8442389670,8442389689,8442389693,8442389725,8442389764,8442390760,8442390772,8442391947,8442392083,8442393338,8442393343,8442393913]
#     people_ids = [1,9, 20, 10, 15, 14, 2, 6, 7, 16]
#     user_relation = 'bride and groom'
#     tags = ['ceremony', 'dancing', 'bride and groom', 'walking the aisle', 'parents', 'first dance', 'kiss']
#     gallery_id = 32900972
#     project_base_url = 'ptstorage_18://pictures/32/900/32900972/1teshu0uhg8u'
#     tags_features_file = r'C:\Users\karmel\Desktop\AlbumDesigner\files\tags.pkl'
#     gal_path = fr'C:\Users\karmel\Desktop\AlbumDesigner\dataset\newest_wedding_galleries\{gallery_id}'
#     start = time.time()
#     ai_images_selected, gallery_photos_info, error_message = auto_selection(project_base_url, ten_photos, tags,
#                                                                             people_ids, user_relation,
#                                                                             r'C:\Users\karmel\Desktop\AlbumDesigner\files\queries_features.pkl',
#                                                                             tags_features_file,DEBUG=True, logger=None)
#
#     end = time.time()
#     elapsed_time = (end - start) / 60
#     print(f"Elapsed time: {elapsed_time:.2f} minutes")
