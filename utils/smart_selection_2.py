import os
import time
import shutil
import random
import csv
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.functional import embedding
from sklearn.cluster import AgglomerativeClustering

from utils import image_meta, image_faces, image_persons, image_embeddings, image_clustering
from utils.image_queries import generate_query
from utils.image_selection_scores import map_cluster_label, calculate_scores
from utils.read_files_types import read_pkl_file
from utils.user_relation_percentage import relations
from utils.similairty_thresholds import similarity_threshold
import numpy as np
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
"""We will get 10 Photos examples
relation to the bride and groom
People selection
Tag cloud flowers hugs, food"""

"""Selection will be baised twoards the tags but not strict """

"""We dont know if the user will select average number of photos for each event, or even total number of images for album"""

import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image


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

def select_images(clusters_class_imgs, gallery_photos_info, ten_photos, people_ids, tags_features, user_relation,
                  logger=None):
    if logger is not None:
        logger.info("====================================")
        logger.info("Starting Image selection Process....")

    error_message = None
    ai_images_selected = []
    category_picked = {}

    for iteration, (event, imges) in enumerate(clusters_class_imgs.items()):
        print("Event in auto2 ", event)
        n_imges = len(imges)
        not_allowed_small_events = ['None','other','settings','vehicle','rings','food', 'accessories', 'entertainment', 'dancing']

        if event not in category_picked:
            category_picked[event] = 0

        # we don't select images from them
        if event == 'None' or event == 'other':
                    continue
        # if we have 4 images for event we choose them all
        elif 3 <= n_imges <=5 and event not in not_allowed_small_events:
            ai_images_selected.extend(imges)
            category_picked[event] += len(imges)
            if logger is not None:
                logger.info(f"Event: {event}, Available images: {len(imges)}")
                logger.info(f"Images selected this iteration without Scores: {len(imges)}")
                logger.info(f"Total images selected so far without Scores: {len(ai_images_selected)}")
                logger.info("*******************************************************")
            else:
                print(f"Event: {event}, Available images: {len(imges)}")
                print(f"Images selected this iteration: {len(imges)}")
                print(f"Total images selected so far without Scores: {len(ai_images_selected)}")
                print("*******************************************************")

        else:
            # Get scores for each image
            images_scores = {}
            for image in imges:
                image_score = calculate_scores(image, gallery_photos_info, ten_photos, people_ids, tags_features)
                images_scores[image] = image_score

            # pick images based on relations percentage
            sorted_scores = sorted(images_scores.items(), key=lambda item: item[1], reverse=True)

            # if all scores are zeros don't select anything
            if all(t[1] <= 0 for t in sorted_scores):
                continue
            else:
                # Get images that have people we want
                available_images_scores = [score for score in sorted_scores if score[1] > 0]
                available_img_ids = [image_id for image_id, score in available_images_scores]

                # remove similar before choosing from them
                if event != 'dancing':
                      available_img_ids = remove_similar_images(event,available_img_ids , gallery_photos_info, threshold=0.94)

                # we dont select one image for event
                if len(available_img_ids) < 3:
                    continue
                # we select all 3 images
                elif len(available_img_ids) == 3:
                    ai_images_selected.extend(available_img_ids)
                    category_picked[event] += len(available_img_ids)
                    if logger is not None:
                        logger.info(f"Event: {event}, Available images: {len(available_img_ids)}")
                        logger.info(f"Images selected after removing similar: 3")
                        logger.info(f"Total images selected : {len(ai_images_selected)}")
                        logger.info("*******************************************************")
                    else:
                        print(f"Event: {event}, Available images: {available_img_ids}")
                        print(f"Images selected after removing similar: 3")
                        print(f"Total images selected so far: {len(ai_images_selected)}")
                        print("*******************************************************")
                else:
                    # Select N number based on relations
                    if event not in relations[user_relation]:
                        select_percentage = 1
                    else:
                        select_percentage = relations[user_relation][event]

                    # selection_params = relations[user_relation][event]
                    # select_weight =  np.prod(np.exp(-0.5 * np.power(((len(available_img_ids)  - selection_params[0]) / selection_params[1]), 2)))

                    n = max(4,round(len(available_img_ids) * select_percentage))

                    selected_images = available_img_ids[:n]
                    ai_images_selected.extend(selected_images)
                    category_picked[event] += len(selected_images)

                    if logger is not None:
                        logger.info(f"Event: {event}, Available images: {len(available_img_ids)}")
                        logger.info(f"Images selected after removing similar with percent: {len(selected_images)}")
                        logger.info(f"Total images selected so far: {len(ai_images_selected)}")
                        logger.info("*******************************************************")
                    else:
                        print(f"Event: {event}, Available images: {len(available_img_ids)}")
                        print(f"Images selected after removing similar with percent: {len(selected_images)}")
                        print(f"Total images selected so far: {len(ai_images_selected)}")
                        print("*******************************************************")

    if len(ai_images_selected) == 0:
        error_message = 'No images were selected.'
        if logger is not None:
           logger.error("No images were selected.")

    print("category and thier images", category_picked)

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
    #gal_id = 40570951
    #directory_path = Path(fr'C:\Users\karmel\Desktop\AlbumDesigner\dataset\newest_wedding_galleries\myselection\{gal_id}')
    #directory_path = Path(fr'C:\Users\karmel\Desktop\AlbumDesigner\dataset\newest_wedding_galleries\{gal_id}')
    #images_selected = [int(f.stem) for f in directory_path.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.png'] and  '_' not in f.stem]

    #not_processed = [im for im in gallery_photos_info if len(gallery_photos_info[im]) < 19]
    # with open(f'{gal_id}_not_processed.csv', mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     for item in not_processed:
    #         writer.writerow([item])


    #ai_images_selected = [im for im in gallery_photos_info if len(gallery_photos_info[im]) >= 19]

    #ai_images_selected = [im for im in ai_images_selected if im in images_selected]
    # for im in ai_images_selected:
    #     im_path = os.path.join(r'C:\Users\karmel\Desktop\AlbumDesigner\dataset\newest_wedding_galleries\40570951',
    #                            str(im) + '.jpg')
    #     shutil.copy(im_path,
    #                 rf'C:\Users\karmel\Desktop\AlbumDesigner\dataset\newest_wedding_galleries\myselection\40570951\{im}.jpg')

    # error_message = None
    # return ai_images_selected, gallery_photos_info, error_message


def copy_selected_folder(ai_images_selected, gal_path):
    """
    Copies selected images from the source directory to a new folder.

    Parameters:
    - ai_images_selected (list): List of image filenames to copy.
    - gal_path (str): Source directory containing the images.

    The function creates a folder named 'selected_images' in the current directory
    and copies the specified images into it.
    """
    destination_folder = r'C:\Users\karmel\Desktop\AlbumDesigner\results\selected_images'

    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Iterate over the list of image IDs and copy each one
    for image_id in ai_images_selected:
        source_path = os.path.join(gal_path, str(image_id) +".jpg")
        dest_path = os.path.join(destination_folder, str(image_id)+".jpg")

        # Check if the source image exists
        if os.path.exists(source_path):
            shutil.copy2(source_path, dest_path)
            print(f"Copied '{image_id}' to '{destination_folder}'.")
        else:
            print(f"Image '{image_id}' not found in '{gal_path}'.")


if __name__ == '__main__':
    ten_photos = [10010769563,10010769568,10010784903,10010784940,10010784958,10010784973,10015553832,10015553856,10015553882,10015553960,10015553971,10015553998,10015554008,10015597235,10015597239]
    people_ids = [2,17,1,5,6,29,30,28,11,9,5,46,62,4,2, 1, 5, 6, 17, 29, 30, 28, 27,2, 4, 1, 10, 17, 36, 49]
    user_relation = 'parents'
    tags = ['settings','detail','very large group','food', 'bride and groom','parents','ceremony', 'portrait', 'group photos']
    #project_base_url = "ptstorage_32://pictures/40/332/40332857/ag14z4rwh9dbeaz0wn"
    gallery_id = 40570951
    project_base_url = 'ptstorage_12://pictures/40/570/40570951/cct1r97452948lcinn'
    tags_features_file =r'C:\Users\karmel\Desktop\AlbumDesigner\files\tags.pkl'
    gal_path = fr'C:\Users\karmel\Desktop\AlbumDesigner\dataset\newest_wedding_galleries\{gallery_id}'
    start = time.time()
    ai_images_selected, gallery_photos_info, error_message = auto_selection(project_base_url, ten_photos, tags, people_ids, user_relation,r'C:\Users\karmel\Desktop\AlbumDesigner\files\queries_features.pkl',tags_features_file, logger=None)

    copy_selected_folder(ai_images_selected,gal_path)

    end = time.time()
    elapsed_time = (end - start) / 60
    print(f"Elapsed time: {elapsed_time:.2f} minutes")


#
# elif event == 'settings':
#     # filter setting images where have same cluster id and cluster label
#     clusters_ids = [gallery_photos_info[image]['cluster_label'] for image in imges]
#     # get indices of similar cluster ids
#     clusters_ids_indices = {}
#     for index, cluster_id in enumerate(clusters_ids):
#         if cluster_id not in clusters_ids_indices:
#             clusters_ids_indices[cluster_id] = []
#         clusters_ids_indices[cluster_id].append(index)
#
#     # get one image with the highest rank from each cluster id
#     images_to_be_selected = []
#     for cluser_id, indices in clusters_ids_indices.items():
#         images_same_cluster = [imges[index] for index in indices]
#         images_ranking = [(i, gallery_photos_info[image]['image_order']) for i, image in
#                           enumerate(images_same_cluster)]
#         sorted_ranking = sorted(images_ranking, key=lambda item: item[1], reverse=True)
#         highest_ranking_index = sorted_ranking[0][0]
#         images_to_be_selected.append((images_same_cluster[highest_ranking_index], sorted_ranking[0][1]))
#
#     select_percentage = relations[user_relation].get(event, 0)
#     available_images = len(images_to_be_selected)
#     n = round(available_images * select_percentage)
#     if n == 0:
#         continue
#     selected_images = images_to_be_selected[:n]
#     selected_images = [image_id for image_id, score in selected_images]
#     filtered_images = remove_similar_images(event, selected_images, gallery_photos_info, threshold=0.98)
#     ai_images_selected.extend(filtered_images)