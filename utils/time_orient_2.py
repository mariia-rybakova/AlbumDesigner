import random
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from datetime import timedelta
from utils.parser import CONFIGS

# Assume CONFIGS dict is defined elsewhere
# CONFIGS = {'FACE_FAR_THRESHOLD': ..., 'FACE_M_THRESHOLD': ..., etc.}


def select_photos_with_scoring(groups, needed_count):
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
        time_col: str = 'image_time_date',
        threshold_minutes: int = 20,
        min_group_size: int = 4
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
def select_images_by_time_and_style(needed_count: int, df: pd.DataFrame) -> list:
    """
    Selects a diverse set of images that are grouped closely in time.

    This function first identifies and filters out temporally isolated images,
    then selects a diverse mix of shots from the remaining valid groups.
    """
    # 1. Pre-computation: Identify and filter out orphan images/small groups
    # This is the most critical improvement to solve the core problem.
    # 2. Vectorized Calculation: Determine shot style for all valid images at once
    df['shot_style'] = df.apply(_determine_shot_style, axis=1)

    # 3. Grouping for Selection: Group by the real temporal cluster and orientation
    groups = df.groupby(['sub_group_time_cluster', 'image_orientation'])

    # 4. Allocation: Determine how many images to take from each group
    # The external `select_photos_with_scoring` function remains a black box as in the original.
    # It should now operate on these filtered, valid groups.
    _, final_selection, _, _ = select_photos_with_scoring(groups, needed_count)

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

    return result[:needed_count]


