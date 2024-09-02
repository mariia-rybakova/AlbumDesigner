import os

from utils import get_image_embeddings, get_faces_info, get_persons_ids, get_clusters_info, get_photo_meta
from utils.image_queries import generate_query
from utils.image_selection_scores import map_cluster_label, calculate_scores
from utils.read_files_types import read_pkl_file
from utils.user_relation_percentage import relations

"""We will get 10 Photos examples
relation to the bride and groom
People selection
Tag cloud flowers hugs, food"""

"""Selection will be baised twoards the tags but not strict """

"""We dont know if the user will select average number of photos for each event, or even total number of images for album"""


def select_images(clusters_class_imgs, gallery_photos_info, ten_photos, people_ids, tags_features, user_relation,
                  logger=None):
    error_message = None
    ai_images_selected = []
    category_picked = {}
    for iteration, (event, imges) in enumerate(clusters_class_imgs.items()):
        cluster_images_scores = {}
        for image in imges:
            image_score = calculate_scores(image, gallery_photos_info, ten_photos, people_ids, tags_features)
            cluster_images_scores[image] = image_score

        # pick images based on relations percentage
        sorted_scores = sorted(cluster_images_scores.items(), key=lambda item: item[1], reverse=True)

        # if all scores are zeros don't select anything
        if all(t[1] == 0 for t in sorted_scores):
            continue
        # we don't want to select identical images of settings
        if event == 'settings':
            # filter setting images where have same cluster id and cluster label
            clusters_ids = [gallery_photos_info[image]['cluster_label'] for image in imges]
            # get indices of similar cluster ids
            clusters_ids_indices = {}
            for index, cluster_id in enumerate(clusters_ids):
                if cluster_id not in clusters_ids_indices:
                    clusters_ids_indices[cluster_id] = []
                clusters_ids_indices[cluster_id].append(index)

            # get one image with the highest rank from each cluster id
            images_to_be_selected = []
            for cluser_id, indices in clusters_ids_indices.items():
                images_same_cluster = [imges[index] for index in indices]
                images_ranking = [(i, gallery_photos_info[image]['ranking']) for i, image in
                                  enumerate(images_same_cluster)]
                sorted_ranking = sorted(images_ranking, key=lambda item: item[1], reverse=True)
                highest_ranking_index = sorted_ranking[0][0]
                images_to_be_selected.append((images_same_cluster[highest_ranking_index], sorted_ranking[0][1]))

            select_percentage = relations[user_relation].get(event, 0)
            available_images = len(images_to_be_selected)
            n = round(available_images * select_percentage)
            if n == 0:
                continue
            selected_images = images_to_be_selected[:n]
            ai_images_selected.extend(selected_images)
        else:
            if event == 'None':
                continue

            # Get images that have people we want
            filter_scores = [score for score in sorted_scores if score[1] > 0]

            # Select N number based on relations
            select_percentage = relations[user_relation][event]
            available_images = len(sorted_scores)
            n = round(available_images * select_percentage)
            if n == 0:
                continue

            selected_images = filter_scores[:n]
            ai_images_selected.extend(selected_images)

        if event not in category_picked:
            category_picked[event] = 0
        category_picked[event] += len(selected_images)

        total_selected_images = len(ai_images_selected)

        print(f"Iteration {iteration}:")
        print(f"Event: {event}, Available images: {available_images}")
        print(f"Images selected this iteration: {len(selected_images)}")
        print(f"Total images selected so far: {total_selected_images}")
        print("*******************************************************")

    if len(ai_images_selected) == 0:
        error_message = 'No images were selected.'

    return ai_images_selected, gallery_photos_info, error_message


def auto_selection(project_base_url, ten_photos, tags_file, people_ids, relation, queries_file, logger):
    faces_file = os.path.join(project_base_url, 'ai_face_vectors.pb')
    cluster_file = os.path.join(project_base_url, 'content_cluster.pb')
    persons_file = os.path.join(project_base_url, 'persons_info.pb')
    image_file = os.path.join(project_base_url, 'ai_search_matrix.pai')
    segmentation_file = os.path.join(project_base_url, 'ai_bgsegementation.pb')

    # Get info from protobuf files server
    gallery_photos_info,errors = get_image_embeddings(image_file,None)
    if errors:
        #logger.error('Couldnt find embeddings for images file %s', image_file)
        return None,None, errors
    gallery_photos_info,errors = get_faces_info(faces_file, gallery_photos_info)
    if errors:
        #logger.error('Couldnt find faces info for images file %s', image_file)
        return None,None, errors
    gallery_photos_info,errors = get_persons_ids(persons_file, gallery_photos_info)

    if errors:
        #logger.error('Couldnt find persons info for images file %s', image_file)
        return None,None, errors
    gallery_photos_info,errors = get_clusters_info(cluster_file, gallery_photos_info)
    if errors:
        #logger.error('Couldnt find clusters info for images file %s', image_file)
        return None,None, errors
    gallery_photos_info,errors = get_photo_meta(segmentation_file, gallery_photos_info)
    if errors:
        #logger.error('Couldnt find photo meta for images file %s', image_file)
        return None,None, errors

    # Get Query Content of each image
    gallery_photos_info = generate_query(queries_file, gallery_photos_info)

    if gallery_photos_info is None:
        # logger.error('the gallery images dict is empty ')
        return None, None, 'Could not generate query'
    # Group images by cluster class labels
    clusters_class_imgs = {}
    # Get images group clusters class labels
    for im_id, img_info in gallery_photos_info.items():
        cluster_class = img_info['cluster_class']
        cluster_class_label = map_cluster_label(cluster_class)
        if cluster_class_label not in clusters_class_imgs:
            clusters_class_imgs[cluster_class_label] = []
        clusters_class_imgs[cluster_class_label].append(im_id)

    tags_features = read_pkl_file(tags_file)

    return select_images(clusters_class_imgs, gallery_photos_info, ten_photos, people_ids, tags_features, relation,
                         logger)

# if __name__ == '__main__':
#     ten_photos = [9741256963, 9741256966, 9741256968, 9741256975, 9741256982, 9741256991, 9741257000, 9741257036,
#                   9741257062, 9741257105]
#     people_ids = [2]
#     user_relation = 'parents'
#     tags = ['ceremony', 'dancing', 'bride and groom']
#     project_base_url = "ptstorage_32://pictures/40/332/40332857/ag14z4rwh9dbeaz0wn"
#     start = time.time()
#     auto_selection(project_base_url, ten_photos, tags, people_ids, user_relation)
#     end = time.time()
#     elapsed_time = (end - start) / 60
#     print(f"Elapsed time: {elapsed_time:.2f} minutes")
