import os
import time
import shutil
import random
import numpy as np
from itertools import islice

from fpdf import FPDF
from PIL import Image
from utils import image_meta, image_faces, image_persons, image_embeddings, image_clustering
from utils.image_queries import generate_query
from utils.image_selection_scores import map_cluster_label, calculate_scores
from utils.read_files_types import read_pkl_file
#from utils.user_relation_percentage import relations
from utils.user_relation_percentage import relations_2
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from math import ceil

"""We will get 10 Photos examples
relation to the bride and groom
People selection
Tag cloud flowers hugs, food"""

"""Selection will be baised twoards the tags but not strict """

"""We dont know if the user will select average number of photos for each event, or even total number of images for album"""
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


def select_by_cluster(clusters_ids,):
    final_selected_images = []
    for cluster_label, images_in_cluster in clusters_ids.items():
        # if i have 4 images i choose one of them if more then i choose more
        if len(images_in_cluster) <= 3:
            # Select image with highest 'order_score' in the cluster
            best_image = max(images_in_cluster, key=lambda img: gallery_photos_info[img]['image_order'])
            final_selected_images.append(best_image)
        else:
            n = round(len(images_in_cluster) / 4)
            images_order = sorted(images_in_cluster,
                                  key=lambda img: gallery_photos_info[img]['image_order'])
            final_selected_images.append(images_order[n])

    return final_selected_images


def select_items(event,clusters_ids,gallery_photos_info,needed_count):
    not_people_events = ['vehicle', 'settings', 'rings', 'entertainment', 'accessories', 'food',
                         'wedding dress', 'suit', 'pet','speech', 'cake cutting']
    result = []
    generators = {k: iter(v) for k, v in clusters_ids.items()}  # Create generators for each list

    while needed_count > 0:
        for key in generators:
            if needed_count == 0:
                break

            # Check list size to determine how many to take
            if len(clusters_ids[key]) <= 5 or event in not_people_events:  # Small list: take at most 1
                items_to_take = min(1, needed_count)
            else:  # Large list: take up to the remaining needed count
                list_size = len(clusters_ids[key])
                items_to_take = round(list_size/ needed_count)

            images_ranked = sorted(clusters_ids[key], key=lambda img: gallery_photos_info[img]['image_order'])

            selected_items = images_ranked[:items_to_take]

            # Add the selected items to the result and update remaining needed count
            result.extend(selected_items)
            needed_count -= len(selected_items)

    return result

def remove_similar_images(category, selected_images, gallery_photos_info,user_relation, threshold=0.90):
        if len(selected_images) == 1:
            return selected_images

        final_selected_images = []
        clusters_ids = get_clusters(selected_images, gallery_photos_info)

        # Identify grayscale images in the filtered list
        grayscale_images = [img for img in selected_images if
                            gallery_photos_info[img]['image_color'] == 0]
        num_grayscale = len(grayscale_images)

        if num_grayscale > 1:
            selected_image = random.choice(grayscale_images)
            selected_images = [image for image in selected_images if image not in grayscale_images]
            selected_images.append(selected_image)
            # Remove other grayscale images except the selected one
            final_selected_images.append(selected_image)

        needed_count =  calculate_selection(category, len(selected_images), relations_2[user_relation])
        chosen_images = select_items(category,clusters_ids,gallery_photos_info, needed_count)
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


def calculate_selection(category,n_actual,lookup_table):
    if category in lookup_table:
        n_target, std_target = lookup_table[category]

        # Calculate score using a Gaussian weighting function
        # weight = np.exp(-((n_actual - n_target) ** 2) / (2 * std_target ** 2))
        #
        # number_selected = int(weight * n_target)
        selection = n_actual - n_target / std_target
        selection = max(4, round(selection))

    else:
        # If category not in lookup, select 0 images
        selection = 0

    return min(selection, n_actual)


def select_images(clusters_class_imgs, gallery_photos_info, ten_photos, people_ids, tags_features, user_relation,
                  logger=None):
    if logger is not None:
        logger.info("====================================")
        logger.info("Starting Image selection Process....")

    error_message = None
    ai_images_selected = []
    category_picked = {}

    for iteration, (category, imges) in enumerate(clusters_class_imgs.items()):
        n_actual = len(imges)
        not_allowed_small_events = ['settings','vehicle','rings','food', 'accessories', 'entertainment', 'dancing']

        if category not in category_picked:
            category_picked[category] = []

        # we don't select images from them
        if category == 'None' or category == 'other':
                    continue
        # if we have 4 images for event we choose them all
        elif n_actual < 3 and n_actual not in not_allowed_small_events:
            continue
        else:
            # Get scores for each image
            scores = images_scores_sorted(imges,gallery_photos_info, ten_photos, people_ids, tags_features)

            # if all scores are zeros don't select anything
            if all(t[1] <= 0 for t in scores):
                continue
            else:
                # Get images that have people we want
                available_images_scores = [score for score in scores if score[1] > 0]
                available_img_ids = [image_id for image_id, score in available_images_scores]

                # remove similar before choosing from them
                available_img_ids = remove_similar_images(category,available_img_ids , gallery_photos_info,user_relation, threshold=0.94)

                if len(available_img_ids) < 3:
                    continue
                else:
                    ai_images_selected.extend(available_img_ids)
                    category_picked[category].extend(available_img_ids)

    if len(ai_images_selected) == 0:
        error_message = 'No images were selected.'
        if logger is not None:
           logger.error("No images were selected.")
    elif len(ai_images_selected) > 140 :
        pass

    for category, images in category_picked.items():
        print("category", category, 'Number of selected images', len(images))

    if logger is not None:
        logger.info(f"Total images: {len(ai_images_selected)}")
        logger.info("*******************************************************")
    else:
        print(f"Total images: {len(ai_images_selected)}")
        print("*******************************************************")

    generate_event_pdfs(category_picked, gal_path, "event_pdfs")

    return ai_images_selected, gallery_photos_info, error_message


def auto_selection(project_base_url, ten_photos, tags_selected, people_ids, relation, queries_file,tags_features_file, logger):
    faces_file = os.path.join(project_base_url, 'ai_face_vectors.pb')
    cluster_file = os.path.join(project_base_url, 'content_cluster.pb')
    persons_file = os.path.join(project_base_url, 'persons_info.pb')
    image_file = os.path.join(project_base_url, 'ai_search_matrix.pai')
    segmentation_file = os.path.join(project_base_url, 'bg_segmentation.pb')

    # Get info from protobuf files server
    gallery_photos_info,errors = image_embeddings.get_image_embeddings(image_file,logger)
    if errors:
         if logger is not None:
             logger.error('Couldnt find embeddings for images file %s', image_file)
         return None,None, errors
    gallery_photos_info,errors = image_faces.get_faces_info(faces_file, gallery_photos_info,logger)
    if errors:
        if logger is not None:
            logger.error('Couldnt find faces info for images file %s', image_file)
        return None,None, errors
    gallery_photos_info,errors = image_persons.get_persons_ids(persons_file, gallery_photos_info,logger)

    if errors:
        if logger is not None:
             logger.error('Couldnt find persons info for images file %s', image_file)
        return None,None, errors
    gallery_photos_info,errors = image_clustering.get_clusters_info(cluster_file, gallery_photos_info,logger)
    if errors:
        if logger is not None:
            logger.error('Couldnt find clusters info for images file %s', image_file)
        return None,None, errors
    gallery_photos_info,errors = image_meta.get_photo_meta(segmentation_file, gallery_photos_info,logger)
    if errors:
        if logger is not None:
            logger.error('Couldnt find photo meta for images file %s', image_file)
        return None,None, errors

    # Get Query Content of each image
    gallery_photos_info = generate_query(queries_file, gallery_photos_info,logger)

    if gallery_photos_info is None:
        if logger is not None:
            logger.error('the gallery images dict is empty ')
        return None, None, 'Could not generate query'

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

    return select_images(clusters_class_imgs, gallery_photos_info, ten_photos, people_ids, selected_tags_features, relation,
                         logger)



if __name__ == '__main__':
    ten_photos = [9871230045,9871230067,9871231567,9871231577,9871231585,9871235650,9871253529,9871253582,9871253597,9871260706]
    people_ids = [1,4,9,13,13, 32, 31, 17, 20, 23,35, 8,5,6,7]
    user_relation = 'bride and groom'
    tags = ['ceremony', 'dancing', 'bride and groom', 'walking the aisle', 'parents', 'first dance', 'kiss']
    #project_base_url = "ptstorage_32://pictures/40/332/40332857/ag14z4rwh9dbeaz0wn"
    gallery_id = 37141824
    project_base_url = 'ptstorage_17://pictures/37/141/37141824/dmgb4onqc3hm'
    tags_features_file =r'C:\Users\karmel\Desktop\AlbumDesigner\files\tags.pkl'
    gal_path = fr'C:\Users\karmel\Desktop\AlbumDesigner\dataset\newest_wedding_galleries\{gallery_id}'
    start = time.time()
    ai_images_selected, gallery_photos_info, error_message = auto_selection(project_base_url, ten_photos, tags, people_ids, user_relation,r'C:\Users\karmel\Desktop\AlbumDesigner\files\queries_features.pkl',tags_features_file, logger=None)

    end = time.time()
    elapsed_time = (end - start) / 60
    print(f"Elapsed time: {elapsed_time:.2f} minutes")

