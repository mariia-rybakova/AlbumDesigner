import math
import random
import numpy as np

from scipy.spatial.distance import cosine
from utils.configs import CONFIGS, limit_imgs, relations

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



def calculate_required_images(total_images, min_limit=10, max_limit=30, max_total_images=150):
    # Ensure required images are between min_limit and max_limit
    required_images = min_limit + ((max_limit - min_limit) / max_total_images) * total_images
    return max(min_limit, min(max_limit, round(required_images)))



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
