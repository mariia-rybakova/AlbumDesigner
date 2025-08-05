from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform



def jaccard_distance(u, v):
    """Computes Jaccard distance between two binary vectors."""
    intersection = sum(u * v)
    union = sum(u) + sum(v) - intersection
    if union == 0:
        return 0.0  # Or 1.0 depending on convention for identical empty sets
    return 1.0 - (intersection / union)

def select_by_cluster(clusters_ids, image_cluster_dict,need):
    """
    Fallback selection mechanism (simplified for this example).
    In a real scenario, this would be your actual robust fallback function.
    """
    selected = []
    if image_cluster_dict:
        for cluster_id_key in clusters_ids:  # Use a different var name to avoid conflict
            if cluster_id_key in image_cluster_dict and image_cluster_dict[cluster_id_key]:
                selected.append(image_cluster_dict[cluster_id_key][0])  # Select first image
    return list(set(selected))[:need]

def person_clustering_selection( # Renamed for clarity, or keep your name
        images_for_category,
        df,
        needed_count,
        image_cluster_dict,
        logger
):

    """
    Processes a single category and adds its report pages to the provided PDF canvas.
    This function NO LONGER creates or saves the PDF.
    """
    """
       Selects images based on person clustering and optionally plots details to a PDF.
       `track_performance_data` is a dictionary to store metadata about the process.
    """
    # Condition for attempting person clustering check if its less than 5 no need return
    if len(images_for_category) < 5:
        return list(set(select_by_cluster(images_for_category, image_cluster_dict, needed_count)))

    relevant_cluster_labels = []
    try:
        if 'cluster_label' in df.columns:
            relevant_cluster_labels = df[df['image_id'].isin(images_for_category)][
                'cluster_label'].unique().tolist()

        result = []
        persons_ids_per_image = []
        images_with_persons_data = []
        for image_id in images_for_category:
            img_info_row = df[df['image_id'] == image_id]
            if img_info_row.empty:
                persons_ids_per_image.append([])  # Image not in df, no person data
                continue
            current_persons_val = img_info_row['persons_ids'].values[0]
            images_with_persons_data.append((image_id, current_persons_val))

        images_for_mlb = [item[0] for item in images_with_persons_data]
        person_ids_for_mlb = [item[1] for item in images_with_persons_data]

        # Need at least two images with person data to compare
        if len(person_ids_for_mlb) < 2:
            return list(set(select_by_cluster(images_for_category, image_cluster_dict,needed_count)))

        mlb = MultiLabelBinarizer()
        binary_matrix = mlb.fit_transform(person_ids_for_mlb)
        # Distance matrix calculation
        if binary_matrix.shape[0] < 2:  # Should be caught earlier, but as a safeguard
            result = select_by_cluster(relevant_cluster_labels, image_cluster_dict,needed_count)
            return list(set(result))

        dist_matrix = squareform(pdist(binary_matrix, metric=jaccard_distance))

        if not (dist_matrix.shape[0] >= 2 and dist_matrix.shape[0] == dist_matrix.shape[1]):
            result = select_by_cluster(relevant_cluster_labels, image_cluster_dict, needed_count)
            return list(set(result))

        # Agglomerative Clustering
        person_clustering_model = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',  # For scikit-learn >= 1.2, use affinity='precomputed'
            # For older versions, metric='precomputed' is fine.
            linkage='complete',
            distance_threshold=0.3,
        )

        person_clustering_model.fit(dist_matrix)
        # Process clusters
        # person_clusters_dict maps: cluster_label -> list of original image_ids in that cluster
        person_clusters_dict = {}
        for idx, label in enumerate(person_clustering_model.labels_):
            original_image_id = images_for_mlb[idx]  # Get the image_id from the list used for MLB
            person_clusters_dict.setdefault(label, []).append(original_image_id)

        current_selected_images = []
        for cluster_label, images_in_cluster in person_clusters_dict.items():
            if needed_count == 0:
                break

            # Sort images in cluster by 'image_order' (lower is better)
            images_ordered = sorted(
                images_in_cluster,
                key=lambda img_id: df.set_index('image_id').loc[img_id, 'total_score'], reverse=True
            )

            selected_from_this_cluster = []
            if images_ordered:  # Ensure cluster is not empty
                if len(images_ordered) <= 4:
                    needed_count -= 1
                    selected_from_this_cluster.append(images_ordered[0])  # Select the one with the best (lowest) order
                else:
                    num_to_select = round(len(images_ordered) / 4)
                    if num_to_select == 0: num_to_select = 1  # Ensure at least one is selected
                    needed_count -= num_to_select
                    selected_from_this_cluster.extend(images_ordered[:num_to_select])

            current_selected_images.extend(selected_from_this_cluster)

        result.extend(list(set(current_selected_images)))  # Add selections from person clustering

    except ValueError as e:  # e.g., if distance_threshold results in too many/few clusters or matrix issues
        result = select_by_cluster(relevant_cluster_labels, image_cluster_dict,needed_count)
        logger.error(f"Couldn't remove similar image using person clustering trying with content clustering  {e}")
        return list(set(result))

    return list(set(result))  # Ensure unique images in the final list
