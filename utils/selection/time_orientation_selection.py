import random
import numpy as np
import pandas as pd
import networkx as nx
import math

from collections import defaultdict
from datetime import timedelta
from utils.configs import CONFIGS
from sklearn.cluster import MiniBatchKMeans
from itertools import combinations
from sklearn_extra.cluster import KMedoids
# from testing_code.plotting import plot_clusters_to_pdf,plot_time_clusters_to_pdf
from sklearn.preprocessing import StandardScaler



def select_photos_with_scoring(groups, needed_count, cluster_name=None):
    """
    Selects a specified number of photos from groups based on scoring and size,
    with special handling for specific cluster names.

    Args:
        groups (pd.core.groupby.generic.DataFrameGroupBy): A pandas DataFrame grouped by
            ['sub_group_time_cluster', 'image_orientation'].
        needed_count (int): The total number of photos to select.
        clustername (str, optional): The name of the cluster. If 'dancing', a special
            selection logic prioritizing landscape photos is applied. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - sorted_group_keys (list): The keys of the groups from which photos were selected.
            - final_selection (dict): A dictionary mapping group keys to the number of photos selected.
            - total_selected (int): The total number of photos selected.
            - eligible_groups (pd.DataFrame): The DataFrame of groups considered for selection.
    """
    if groups.ngroups == 0 or needed_count == 0:
        return [], {}, 0, pd.DataFrame()

    # --- Initial Setup (Common for both logic paths) ---
    group_sizes = groups.size().rename('size')
    avg_scores = groups['total_score'].mean().rename('avg_score')
    group_info = pd.concat([group_sizes, avg_scores], axis=1).reset_index()

    final_selection = {}
    photos_selected_so_far = 0
    group_keys_col = ['sub_group_time_cluster', 'image_orientation']

    # --- Special Logic for 'dancing' cluster ---
    if cluster_name == 'dancing':
        # For the 'dancing' cluster, we prioritize all landscape shots first,
        # then fill the remainder with portrait shots.

        # 1. Separate groups by orientation
        # Assuming orientation values are 'landscape' and 'portrait'.
        # This should be validated or made more robust if other values can exist.
        landscape_groups = group_info[group_info['image_orientation'] == 'landscape'].copy()
        portrait_groups = group_info[group_info['image_orientation'] == 'portrait'].copy()

        # 2. Sort each orientation group by score and size to pick the best ones first
        landscape_groups.sort_values(by=['avg_score', 'size'], ascending=[False, False], inplace=True)
        portrait_groups.sort_values(by=['avg_score', 'size'], ascending=[False, False], inplace=True)

        # 3. Phase 1: Select from Landscape groups first
        for _, group in landscape_groups.iterrows():
            if photos_selected_so_far >= needed_count:
                break

            group_id = tuple(group[group_keys_col])
            # Take as many as possible up to the group's size or until needed_count is met
            can_take = needed_count - photos_selected_so_far
            to_take = min(group['size'], can_take)

            if to_take > 0:
                final_selection[group_id] = to_take
                photos_selected_so_far += to_take

        # 4. Phase 2: Fill remaining spots with Portrait groups if needed
        remaining_needed = needed_count - photos_selected_so_far
        if remaining_needed > 0:
            for _, group in portrait_groups.iterrows():
                if photos_selected_so_far >= needed_count:
                    break

                group_id = tuple(group[group_keys_col])
                can_take = needed_count - photos_selected_so_far
                to_take = min(group['size'], can_take)

                if to_take > 0:
                    final_selection[group_id] = to_take
                    photos_selected_so_far += to_take

        # 5. Finalize and return
        total_selected = sum(final_selection.values())
        sorted_group_keys = list(final_selection.keys())
        # For this logic path, `eligible_groups` is simply all original groups
        return sorted_group_keys, final_selection, total_selected, group_info


    # --- Original Logic for all other clusters ---
    else:
        # 1. Filter out the bottom 25% based on score
        mask = (
                (group_info['avg_score'] > 0.30) |
                (group_info['avg_score'].nunique() == 1)
        )
        eligible_groups = group_info[mask].copy()
        if eligible_groups.empty:
            eligible_groups = group_info.copy()

        # 2. Sort by priority: high score, large size
        eligible_groups = eligible_groups.sort_values(
            by=['avg_score', 'size'], ascending=[False, False]
        )

        # 3. Assign minimums — ensuring not to exceed needed_count
        def get_minimums(size):
            return 1 if size <= 5 else random.randint(2, 3)

        eligible_groups['minimums'] = eligible_groups['size'].apply(get_minimums)
        eligible_groups['minimums'] = np.minimum(eligible_groups['minimums'], eligible_groups['size'])

        for idx, group in eligible_groups.iterrows():
            if photos_selected_so_far >= needed_count:
                break

            group_id = tuple(group[group_keys_col])
            min_needed = int(group['minimums'])

            max_possible = needed_count - photos_selected_so_far
            actual_take = min(min_needed, max_possible)

            if actual_take > 0:
                final_selection[group_id] = actual_take
                photos_selected_so_far += actual_take
            else:
                final_selection[group_id] = 0

        # 4. Fill remaining using remaining capacity
        remaining_to_select = needed_count - photos_selected_so_far
        if remaining_to_select > 0:
            eligible_groups['already_selected'] = eligible_groups.apply(
                lambda row: final_selection.get(tuple(row[group_keys_col]), 0), axis=1)
            eligible_groups['capacity_after_min'] = eligible_groups['size'] - eligible_groups['already_selected']

            priority_sorted_groups = eligible_groups.sort_values(
                by=['avg_score', 'capacity_after_min'], ascending=[False, False]
            )

            for _, group in priority_sorted_groups.iterrows():
                if remaining_to_select <= 0:
                    break

                group_id = tuple(group[group_keys_col])
                already_selected = final_selection.get(group_id, 0)
                remaining_capacity = group['size'] - already_selected

                to_take = min(remaining_to_select, remaining_capacity)
                if to_take > 0:
                    final_selection[group_id] = already_selected + to_take
                    remaining_to_select -= to_take

        # 5. Clean up and return
        final_selection = {k: v for k, v in final_selection.items() if v > 0}
        total_selected = sum(final_selection.values())
        sorted_group_keys = list(final_selection.keys())

        return sorted_group_keys, final_selection, total_selected, eligible_groups



def select_photos_with_scoring_old(groups, needed_count):
    if groups.ngroups == 0:
        return {}, 0

    # 1. Get group sizes and average scores
    group_sizes = groups.size().rename('size')
    avg_scores = groups['total_score'].mean().rename('avg_score')

    # Aggregate to preserve all columns
    group_info = pd.concat([group_sizes, avg_scores], axis=1).reset_index()

    # 2. Filter out the bottom 25% based on score
    # score_percentile_25 = group_info['avg_score'].quantile(0.18)
    mask = (
            (group_info['avg_score'] > 0.30) |
            (group_info['avg_score'].nunique() == 1)
    )

    eligible_groups = group_info[mask].copy()

    if eligible_groups.empty:
        eligible_groups = group_info.copy()

    # 3. Sort by priority: high score, large size
    eligible_groups = eligible_groups.sort_values(
        by=['avg_score', 'size'], ascending=[False, False]
    )

    # 4. Assign minimums — ensuring not to exceed needed_count
    def get_minimums(size):
        return 1 if size <= 5 else random.randint(2, 3)

    eligible_groups['minimums'] = eligible_groups['size'].apply(get_minimums)
    eligible_groups['minimums'] = np.minimum(eligible_groups['minimums'], eligible_groups['size'])

    group_keys = ['sub_group_time_cluster', 'image_orientation']

    final_selection = {}
    photos_selected_so_far = 0

    for idx, group in eligible_groups.iterrows():
        if photos_selected_so_far >= needed_count:
            break

        group_id = tuple(group[group_keys])
        min_needed = int(group['minimums'])

        max_possible = needed_count - photos_selected_so_far
        actual_take = min(min_needed, max_possible)

        if actual_take > 0:
            final_selection[group_id] = actual_take
            photos_selected_so_far += actual_take
        else:
            final_selection[group_id] = 0

    # 5. Fill remaining using remaining capacity
    remaining_to_select = needed_count - photos_selected_so_far

    if remaining_to_select > 0:
        eligible_groups['already_selected'] = eligible_groups.apply(
            lambda row: final_selection.get(tuple(row[group_keys]), 0), axis=1)
        eligible_groups['capacity_after_min'] = eligible_groups['size'] - eligible_groups['already_selected']

        priority_sorted_groups = eligible_groups.sort_values(
            by=['avg_score', 'capacity_after_min'], ascending=[False, False]
        )

        for _, group in priority_sorted_groups.iterrows():
            if remaining_to_select <= 0:
                break

            group_id = tuple(group[group_keys])
            already_selected = final_selection.get(group_id, 0)
            remaining_capacity = group['size'] - already_selected

            to_take = min(remaining_to_select, remaining_capacity)
            if to_take > 0:
                final_selection[group_id] = already_selected + to_take
                remaining_to_select -= to_take

    # Clean up
    final_selection = {k: v for k, v in final_selection.items() if v > 0}
    total_selected = sum(final_selection.values())
    sorted_group_keys = list(final_selection.keys())  # Preserves the sorted order

    return sorted_group_keys, final_selection, total_selected, eligible_groups

# --- Step 1: Helper function for robust temporal clustering ---
def identify_temporal_clusters(
        df: pd.DataFrame,
        time_col: str,
        threshold_minutes: int,
        min_group_size: int,
        logger,
) -> pd.DataFrame:
    """
    Identifies temporal clusters of images using a graph-based approach.

    Images are nodes, and an edge exists if their timestamps are within the threshold.
    Filters out any cluster smaller than min_group_size.

    Args:
        df: DataFrame containing image data with a timestamp column.
        time_col: The name of the timestamp column.
        threshold_minutes: The maximum time difference in minutes to connect two images.
        min_group_size: The minimum number of images required for a cluster to be kept.

    Returns:
        A DataFrame containing only the images from valid temporal clusters,
        with a new 'temporal_group_id' column.
    """
    try:
        if df.empty or time_col not in df.columns:
            return pd.DataFrame()

        df[time_col] = pd.to_datetime(df[time_col])

        df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
        threshold = timedelta(minutes=threshold_minutes)

        G = nx.Graph()
        # Add all images as nodes
        G.add_nodes_from(df_sorted.index)

        # Efficiently add edges between images within the time threshold
        for i in range(len(df_sorted)):
            j = i + 1
            # This subtraction now works correctly (datetime - datetime = timedelta)
            while j < len(df_sorted) and (df_sorted.at[j, time_col] - df_sorted.at[i, time_col]) <= threshold:
                G.add_edge(i, j)
                j += 1

        # Find connected components (the actual temporal clusters)
        connected_components = list(nx.connected_components(G))

        # Create a mapping from image index to a component ID
        node_to_component = {node: i for i, component in enumerate(connected_components) for node in component}
        df_sorted['temporal_group_id'] = df_sorted.index.map(node_to_component)

        # Filter out small, isolated groups (orphans)
        group_sizes = df_sorted['temporal_group_id'].value_counts()
        valid_groups = group_sizes[group_sizes >= min_group_size].index

    except Exception as e:
        logger.error(f"Error in identify_temporal_clusters {e}")
        return pd.DataFrame()

    return df_sorted[df_sorted['temporal_group_id'].isin(valid_groups)].copy()


# --- Step 2: Refactored helper for efficient shot style calculation ---
def _determine_shot_style(row: pd.Series) -> str:
    """
    Determines the shot style (close, medium, far) for a single image row.
    Designed to be used with df.apply().
    """
    if row.get('n_faces', 0) > 0:
        max_face_size = max(face.bbox.x2 * face.bbox.y2 for face in row["faces_info"])
        if max_face_size < CONFIGS['FACE_FAR_THRESHOLD']: return "far"
        if max_face_size < CONFIGS['FACE_M_THRESHOLD']: return "medium"
        return "close"

    if row.get('number_bodies', 0) > 0:
        max_body_size = max(body.bbox.x2 * body.bbox.y2 for body in row["bodies_info"])
        if max_body_size < CONFIGS['BODY_FAR_THRESHOLD']: return "far"
        if max_body_size < CONFIGS['BODY_MEDIUM_THRESHOLD']: return "medium"
        return "close"

    return "unknown"


# --- Step 3: Refactored and simplified main function ---
def select_images_by_time_and_style(needed_count: int, df: pd.DataFrame,cluster_name, logger) -> list:
    """
    Selects a diverse set of images that are grouped closely in time.

    This function first identifies and filters out temporally isolated images,
    then selects a diverse mix of shots from the remaining valid groups.
    """
    try:
        # 1. Pre-computation: Identify and filter out orphan images/small groups
        # This is the most critical improvement to solve the core problem.
        # 2. Vectorized Calculation: Determine shot style for all valid images at once
        df['shot_style'] = df.apply(_determine_shot_style, axis=1)

        # 3. Grouping for Selection: Group by the real temporal cluster and orientation
        groups = df.groupby(['sub_group_time_cluster', 'image_orientation'])

        # 4. Allocation: Determine how many images to take from each group
        # The external `select_photos_with_scoring` function remains a black box as in the original.
        # It should now operate on these filtered, valid groups.
        _, final_selection, _, _ = select_photos_with_scoring(groups, needed_count,cluster_name)

        result = []

        # 5. Unified Selection Logic: Loop and select with style diversity
        for group_key, max_take in final_selection.items():
            if max_take == 0:
                continue

            group_df = groups.get_group(group_key)

            # Organize images by style for round-robin selection
            style_buckets = defaultdict(list)
            # Sort once by score to prepare for selection
            sorted_group = group_df.sort_values('total_score', ascending=False)
            for _, row in sorted_group.iterrows():
                style_buckets[row['shot_style']].append(row['image_id'])

            # Round-robin selection for diversity
            selected_from_group = []
            style_indices = {style: 0 for style in style_buckets}
            # Prioritize styles with more high-scoring images, but can be simplified if not needed
            styles_order = list(style_buckets.keys())

            while len(selected_from_group) < max_take:
                added_this_round = False
                for style in styles_order:
                    if style_indices[style] < len(style_buckets[style]):
                        selected_from_group.append(style_buckets[style][style_indices[style]])
                        style_indices[style] += 1
                        added_this_round = True
                        if len(selected_from_group) == max_take:
                            break
                if not added_this_round:
                    break  # No more images to select

            result.extend(selected_from_group)

            if len(result) >= needed_count:
                break

    except ValueError as e:  # e.g., if distance_threshold results in too many/few clusters or matrix issues
        logger.error(f"Couldn't remove similar image using time and style {e}")
        raise e

    return result[:needed_count]


def filter_similarity(need, df,cluster_name, target_group_size=10,threshold = 0.9):
    if len(df) <= need:
        return df['image_id'].values.tolist()

    embeddings = np.vstack(df['embedding'].values).astype('float32')
    n_clusters = math.ceil(len(df) / target_group_size)

    # kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=64)
    # labels = kmeans.fit_predict(embeddings)

    kmedoids = KMedoids(n_clusters=n_clusters, random_state=42, method="pam")
    labels = kmedoids.fit_predict(embeddings)

    groups = {}
    for img_id, label in zip(df['image_id'], labels):
        groups.setdefault(label, []).append(img_id)

    embedding_dict = dict(zip(df['image_id'], df['embedding']))
    score_lookup = dict(zip(df['image_id'], df['total_score']))

    # Step 2: DSU (Union-Find) to merge similar images
    parent = {img: img for img in df['image_id']}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    # Step 3: Compare only within each k-means group
    for g_id, img_ids in groups.items():
        if len(img_ids) < 2:
            continue
        for a, b in combinations(img_ids, 2):
            sim = float(np.dot(embedding_dict[a], embedding_dict[b]))  # normalized vectors
            if sim >= threshold:
                union(a, b)

    # Step 4: Build clusters from DSU
    clusters = {}
    for img in df['image_id']:
        root = find(img)
        clusters.setdefault(root, []).append(img)

    total_images = sum(len(c) for c in clusters.values())
    cluster_allocation = {
        root: max(1, round((len(c) / total_images) * need))
        for root, c in clusters.items()
    }

    selected_images = []
    for root, imgs in clusters.items():
        sorted_imgs = sorted(imgs, key=lambda x: score_lookup[x], reverse=True)
        selected_images.extend(sorted_imgs[:cluster_allocation[root]])

    # If over need, cut down by global score
    if len(selected_images) > need:
        selected_images = sorted(selected_images, key=lambda x: score_lookup[x], reverse=True)[:need]

    # plot_clusters_to_pdf(df, selected_images, clusters, image_dir=r'C:\Users\karmel\Desktop\AlbumDesigner\dataset\newest_wedding_galleries/46881120', output_pdf=fr"C:\Users\karmel\Desktop\AlbumDesigner\output\46881120\{cluster_name}.pdf")
    #
    # plot_time_clusters_to_pdf(df, selected_images, r'C:\Users\karmel\Desktop\AlbumDesigner\dataset\newest_wedding_galleries/46881120', fr"C:\Users\karmel\Desktop\AlbumDesigner\output\46881120\{cluster_name}_time.pdf",
    #                           img_ext=".jpg", thumb_size=(200, 200), cols=4,
    #                           margin=40, padding=20)


    return selected_images


def filter_similarity_diverse(
    need: int,
    df: pd.DataFrame,
    cluster_name,
    logger,
    target_group_size: int = 10,
) -> list[str]:
    """
    Uses your existing time clusters (sub_group_time_cluster),
    your orientation column (image_oreintation/image_orientation),
    and your image embeddings (image_embedding).
    Returns a list of selected image_ids.
    """
    try:
        is_not_scored = all(x == 1 for x in df['total_score'].to_list())
        if is_not_scored:
            score_lookup = dict(zip(df['image_id'], df['image_order']))
        else:
            score_lookup = dict(zip(df['image_id'], df['total_score']))

        # Extract embeddings
        embeddings = np.vstack(df['embedding'].values).astype('float32')

        ori_ser = df["image_orientation"].astype(str).str.lower()

        orientation_bin = (
            ori_ser.map({"portrait": 0, "landscape": 1})
            .fillna(1)  # e.g., "square", "unknown", NaN -> 1
            .astype("int8")
            .to_numpy()
            .reshape(-1, 1)
        )

        # Time cluster as numeric (already categorical, can just scale)
        time_cluster = df["sub_group_time_cluster"].values.reshape(-1, 1)

        # Combine features
        features = np.hstack([embeddings, orientation_bin, time_cluster])

        # Scale combined features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Decide number of clusters
        n_clusters = max(2, int(np.ceil(len(df) / target_group_size)))

        # Run KMedoids
        kmedoids = KMedoids(n_clusters=n_clusters, random_state=42, method="pam")
        labels = kmedoids.fit_predict(features_scaled)

        df["joint_cluster"] = labels

        clusters = {}
        for cid, grp in df.groupby("joint_cluster"):
            clusters[cid] = grp["image_id"].tolist()

        reverse = not is_not_scored
        for cid in clusters:
            clusters[cid] = sorted(clusters[cid], key=lambda x: score_lookup[x], reverse=reverse)

        emb_lookup = dict(zip(df['image_id'], df['embedding']))

        # 3) allocation
        K = len(clusters)
        alloc = {cid: 0 for cid in clusters}

        # Per-cluster cap based on size (tune thresholds as you like)
        def cluster_cap(sz: int) -> int:
            if sz <= 7:     return 1  # tiny/small: only 1 (diversity)
            if sz <= 12:    return 2  # medium: up to 2
            if sz <= 20:    return 3  # large: up to 3
            return 4  # very large: up to 4

        caps = {cid: min(cluster_cap(len(clusters[cid])), len(clusters[cid])) for cid in clusters}

        if K <= need:
            # Baseline: one per cluster
            for cid in alloc:
                alloc[cid] = 1
            remaining = need - K

            # Greedy: keep giving extras to clusters with the most residual capacity, honoring caps
            while remaining > 0:
                residuals = [(cid, caps[cid] - alloc[cid]) for cid in clusters]
                residuals = [(cid, r) for cid, r in residuals if r > 0]
                if not residuals:
                    break
                # tie-break by residual, then by cluster size, then by top score
                best_cid = max(
                    residuals,
                    key=lambda x: (x[1], len(clusters[x[0]]), score_lookup[clusters[x[0]][0]])
                )[0]
                alloc[best_cid] += 1
                remaining -= 1
        else:
            # More clusters than needed: take the best 'need' clusters (top image score) — 1 each
            top = sorted(clusters.keys(), key=lambda c: score_lookup[clusters[c][0]], reverse=True)[:need]
            for cid in top:
                alloc[cid] = 1

        # ========= 4) selection (prefer different time bins + different subqueries; far by embedding) =========
        selected: list[str] = []

        # Lookups for time bin and subquery (optional)
        time_lookup = dict(
            zip(df["image_id"], df["sub_group_time_cluster"])) if "sub_group_time_cluster" in df.columns else {}
        subq_lookup = dict(
            zip(df["image_id"], df["image_subquery_content"])) if "image_subquery_content" in df.columns else {}

        def min_emb_distance_to_selected(img_id: str) -> float:
            if not selected:
                return float("inf")
            e = emb_lookup[img_id]
            # cosine distance on normalized vectors = 1 - dot
            return min(1.0 - float(np.dot(e, emb_lookup[sid])) for sid in selected)

        # First pass: one per allocated cluster (top score)
        for cid, k in alloc.items():
            if k <= 0:
                continue
            first = clusters[cid][0]
            selected.append(first)

        # Extra picks per cluster
        for cid, k in alloc.items():
            if k <= 1:
                continue

            # Already-used time bins and subqueries in THIS cluster (based on what's selected from this cluster)
            used_time_bins = set()
            used_subqs = set()
            for sid in selected:
                # limit to this cluster’s already picked (fast check via membership)
                if sid in clusters[cid]:
                    if time_lookup:
                        used_time_bins.add(time_lookup.get(sid))
                    if subq_lookup:
                        used_subqs.add(subq_lookup.get(sid))

            # Candidates are the remaining images in this cluster (already sorted by score)
            cands = [img for img in clusters[cid][1:] if img not in selected]

            # Step A: prefer candidates from NEW time bins and NEW subqueries within the cluster
            def candidate_priority(img: str):
                tb = time_lookup.get(img, None)
                sq = subq_lookup.get(img, None)
                new_time = 0 if (tb in used_time_bins) else 1
                new_subq = 0 if (sq in used_subqs) else 1
                # Higher is better: prefer new time bin, then new subquery, then higher score
                return (new_time, new_subq, score_lookup[img])

            # For each extra slot in this cluster:
            for _ in range(k - 1):
                if not cands:
                    break

                # Rank by (new time bin?, new subquery?, score), then pick the one farthest by embedding
                cands.sort(key=candidate_priority, reverse=True)
                # Take top few by priority to limit distance checks (optional micro-optim)
                shortlist = cands[: min(8, len(cands))]

                pick = max(shortlist, key=min_emb_distance_to_selected)
                selected.append(pick)

                # Update used sets for this cluster
                if time_lookup:
                    used_time_bins.add(time_lookup.get(pick))
                if subq_lookup:
                    used_subqs.add(subq_lookup.get(pick))

                # Remove from candidate pool
                cands.remove(pick)

        # Tidy to exact 'need'
        if len(selected) > need:
            selected = sorted(selected, key=lambda x: score_lookup[x], reverse=True)[:need]
        elif len(selected) < need:
            remaining = [img for cid in clusters for img in clusters[cid] if img not in selected]
            remaining.sort(key=lambda x: score_lookup[x], reverse=True)
            selected.extend(remaining[: need - len(selected)])

        # plot_clusters_to_pdf(df, selected, clusters, image_dir=r'C:\Users\karmel\Desktop\AlbumDesigner\dataset\newest_wedding_galleries/46881120', output_pdf=fr"C:\Users\karmel\Desktop\AlbumDesigner\output\46229128\{cluster_name}.pdf")
        #
        # plot_time_clusters_to_pdf(df, selected, r'C:\Users\karmel\Desktop\AlbumDesigner\dataset\newest_wedding_galleries/46881120', fr"C:\Users\karmel\Desktop\AlbumDesigner\output\46229128\{cluster_name}_time.pdf",
        #                           img_ext=".jpg", thumb_size=(200, 200), cols=4,
        #                           margin=40, padding=20)

    except Exception as e:
        logger.error(f"Error filter_similarity_diverse cluster name {cluster_name}: {e}")
        return []

    return selected



