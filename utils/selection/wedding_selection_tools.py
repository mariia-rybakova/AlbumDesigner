from collections import deque


def get_clusters(df):
    clusters_ids = {}
    # image class
    for index, data in df.iterrows():
        class_id = data['cluster_label']
        image_id = data['image_id']
        if class_id not in clusters_ids:
            clusters_ids[class_id] = []

        clusters_ids[class_id].append(image_id)
    return clusters_ids

def select_non_similar_images(clusters_ids,image_order_dict,needed_count):
    sorted_clusters = {}
    for key, imgs in clusters_ids.items():
        sorted_clusters[key] = sorted(
            imgs, key=lambda img: image_order_dict.get(img, float('-inf')), reverse=True
        )

    # --- Step 2: rank clusters by their max score ---
    cluster_order = sorted(
        sorted_clusters.keys(),
        key=lambda k: image_order_dict.get(sorted_clusters[k][0], float('-inf'))
        if sorted_clusters[k] else float('-inf'),
        reverse=True
    )
    # --- Step 3: convert each cluster to deque for efficient pops ---
    cluster_deques = {k: deque(v) for k, v in sorted_clusters.items()}

    result = []
    seen = set()

    # --- Step 4: round-robin across clusters in priority order ---
    while needed_count > 0:
        progress = False
        for key in cluster_order:
            if needed_count <= 0:
                break
            if not cluster_deques[key]:
                continue

            candidate = cluster_deques[key].popleft()
            if candidate in seen:
                continue

            result.append(candidate)
            seen.add(candidate)
            needed_count -= 1
            progress = True

        # if no cluster contributed this round → stop (no more usable images)
        if not progress:
            break

    return result
