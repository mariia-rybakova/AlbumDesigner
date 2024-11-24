import os
import time
import shutil
import random
import numpy as np

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



"""We will get 10 Photos examples
relation to the bride and groom
People selection
Tag cloud flowers hugs, food"""

"""Selection will be baised twoards the tags but not strict """

"""We dont know if the user will select average number of photos for each event, or even total number of images for album"""

def plot_groups_to_pdf(event,groups, images_path, output_pdf='persons_clustering.pdf'):
    """
    Plots each group of images on a separate page in a PDF.

    Parameters:
    - groups (dict): A dictionary where keys are group names and values are lists of image IDs.
    - images_path (str): The path to the directory containing the images.
    - output_pdf (str): Output PDF file path.
    """
    output_pdf = f'{event}.pdf'
    pdf = FPDF()

    for group, image_ids in groups.items():
        pdf.add_page()

        for idx, img_id in enumerate(image_ids):
            img_path = os.path.join(images_path, str(img_id) + '.jpg')

            # Check if image file exists
            if not os.path.exists(img_path):
                print(f"Image path not found for ID {img_id}")
                continue

            # Open and resize image
            img = Image.open(img_path)
            width, height = img.size
            aspect_ratio = width / height

            # Define desired width (adjustable) and calculate height to maintain aspect ratio
            pdf_width = 60
            img_height = pdf_width / aspect_ratio

            # Set default (x, y) positions, arranging images vertically with an offset
            x = 10
            y = 10 + idx * (img_height + 10)

            # Save the resized image temporarily
            temp_path = f"temp_{img_id}.png"
            img.thumbnail((pdf_width, img_height))
            img.save(temp_path)

            # Add image to PDF
            pdf.image(temp_path, x=x, y=y, w=pdf_width, h=img_height)

            # Optionally remove the temporary file after adding it (optional if you want to save space)
            os.remove(temp_path)

    # Save the PDF
    pdf.output(output_pdf)
    print(f"PDF saved to {output_pdf}")


def remove_similar_images(event, selected_images, gallery_photos_info, threshold=0.90):
        if len(selected_images) == 1:
            return selected_images

        not_people_events = ['vehicle','settings','rings', 'bride and groom','entertainment', 'accessories','food','wedding dress','suit','pet']
        if event not in not_people_events:
            # Step 1: Extract and one-hot encode `persons_ids` for Jaccard-based clustering
            persons_ids = [gallery_photos_info[image_id]['persons_ids'] for image_id in selected_images if 'persons_ids' in gallery_photos_info[image_id] ]
            mlb = MultiLabelBinarizer()
            one_hot_persons_ids = mlb.fit_transform(persons_ids)

            # Calculate the Jaccard similarity for each pair of images based on `persons_ids`
            # We use Jaccard similarity for binary/categorical data, suited to one-hot encoded IDs
            person_similarity_matrix = np.zeros((len(one_hot_persons_ids), len(one_hot_persons_ids)))
            for i in range(len(one_hot_persons_ids)):
                for j in range(i, len(one_hot_persons_ids)):  # Only compute upper triangle
                    similarity = jaccard_score(one_hot_persons_ids[i], one_hot_persons_ids[j])
                    person_similarity_matrix[i, j] = person_similarity_matrix[j, i] = similarity

            # Convert similarity to distance for clustering
            person_distance_matrix = 1 - person_similarity_matrix

            if len(person_distance_matrix) == 0 or  person_distance_matrix.shape[0] < 2:
                # if no people found then select four
                """Select 4 images"""
                clusters_ids = {}
                # image class
                for image_id in selected_images:
                    class_id = gallery_photos_info[image_id]['cluster_label']
                    if class_id not in clusters_ids:
                        clusters_ids[class_id] = []

                    clusters_ids[class_id].append(image_id)

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
            else:
                # Perform Agglomerative Clustering based on person similarity
                person_clustering = AgglomerativeClustering(
                    n_clusters=None,
                    linkage='average',
                    distance_threshold=0.3,  # Adjust this threshold as needed
                )
                person_clustering.fit(person_distance_matrix)

                # Dictionary to hold images in each person-based cluster
                person_clusters = {}
                for idx, label in enumerate(person_clustering.labels_):
                    person_clusters.setdefault(label, []).append(selected_images[idx])

                #plot_groups_to_pdf(event, person_clusters, r'C:\Users\karmel\Desktop\AlbumDesigner\dataset\newest_wedding_galleries\myselection\40570951', output_pdf='persons_clustering.pdf')
                # Select the best image in each cluster based on 'order_score'
                final_selected_images = []
                for cluster_label, images_in_cluster in person_clusters.items():
                    # if i have 4 images i choose one of them if more then i choose more
                    if len(images_in_cluster) <= 4:
                        # Select image with highest 'order_score' in the cluster
                        best_image = max(images_in_cluster, key=lambda img: gallery_photos_info[img]['image_order'])
                        final_selected_images.append(best_image)
                    else:
                        n = round(len(images_in_cluster)/4)
                        images_order = sorted(images_in_cluster, key=lambda img: gallery_photos_info[img]['image_order'])
                        final_selected_images.append(images_order[n])

        elif event == 'bride and groom':
            """Select 4 images"""
            clusters_ids = {}
            # image class
            for image_id in selected_images:
                class_id = gallery_photos_info[image_id]['cluster_label']
                if class_id not in clusters_ids:
                    clusters_ids[class_id] = []

                clusters_ids[class_id].append(image_id)

            final_selected_images = []
            for cluster_label, images_in_cluster in clusters_ids.items():
                # if i have 4 images i choose one of them if more then i choose more
                if len(images_in_cluster) <= 4:
                    # Select image with highest 'order_score' in the cluster
                    best_image = max(images_in_cluster, key=lambda img: gallery_photos_info[img]['image_order'])
                    final_selected_images.append(best_image)
                else:
                    n = round(len(images_in_cluster) / 4)
                    images_order = sorted(images_in_cluster, key=lambda img: gallery_photos_info[img]['image_order'])
                    final_selected_images.append(images_order[n])

            print("Selected images for bride and groom", {len(final_selected_images)})

        else:
            """Select one best image"""

            clusters_ids = {}
            # image class
            for image_id in selected_images:
                class_id = gallery_photos_info[image_id]['cluster_label']
                if class_id not in clusters_ids:
                    clusters_ids[class_id] = []

                clusters_ids[class_id].append(image_id)


            # Select the best image in each cluster based on 'order_score'
            final_selected_images = []
            for cluster_label, images_in_cluster in clusters_ids.items():
                # Select image with highest 'order_score' in the cluster
                best_image = max(images_in_cluster, key=lambda img: gallery_photos_info[img]['image_order'])
                final_selected_images.append(best_image)

                # Print out similar images for reference
                if len(images_in_cluster) > 1:
                    similar_images = [img for img in images_in_cluster if img != best_image]
                    print(
                        f"Cluster {cluster_label}: Keeping '{best_image}', removing similar images: {similar_images}")

        # Identify grayscale images in the filtered list
        grayscale_images = [img for img in final_selected_images if
                            gallery_photos_info[img]['image_color'] == 0]
        num_grayscale = len(grayscale_images)

        if num_grayscale >= 1:
            selected_image = random.choice(grayscale_images)
            print(f"Randomly selected grayscale image: {selected_image}")

            # Remove other grayscale images except the selected one
            final_selected_images = [img for img in final_selected_images if
                                     (gallery_photos_info[img]['image_color'] != 0) or (
                                                 img == selected_image)]

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
        weight = np.exp(-((n_actual - n_target) ** 2) / (2 * std_target ** 2))

        number_selected = int(weight * n_target)
    else:
        # If category not in lookup, select 0 images
        number_selected = 0

    return number_selected


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
        print(f"Categroy {category}, acutal number of images {n_actual}")
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
                available_img_ids = remove_similar_images(category,available_img_ids , gallery_photos_info, threshold=0.94)

                # we dont select one image for event
                if len(available_img_ids) == 3:
                    ai_images_selected.extend(available_img_ids)
                    category_picked[category].extend(available_img_ids)
                elif len(available_img_ids) < 3:
                    continue
                else:
                    n_actual = len(available_img_ids)
                    n = calculate_selection(category, n_actual, relations_2[user_relation])
                    n = max(4,n)

                    selected_images = available_img_ids[:n]
                    ai_images_selected.extend(selected_images)
                    category_picked[category].extend(selected_images)

    if len(ai_images_selected) == 0:
        error_message = 'No images were selected.'
        if logger is not None:
           logger.error("No images were selected.")
    elif len(ai_images_selected) > 140 :
        pass

    if logger is not None:
        logger.info(f"Total images: {len(ai_images_selected)}")
        logger.info(f"Picked: {category_picked}")
        logger.info("*******************************************************")
    else:
        print(f"Total images: {len(ai_images_selected)}")
        print(f"Picked: {category_picked}")
        print("*******************************************************")

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

