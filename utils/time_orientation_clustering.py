import random
import numpy as np
import pandas as pd

from collections import defaultdict
from utils.parser import CONFIGS


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


def select_by_new_clustering_experiment(needed_count,df_all_data=None):
    # Group by time_cluster and image_orientation
    groups = df_all_data.groupby(['sub_group_time_cluster', 'image_orientation'])
    original_groups_names = list(groups.groups.keys())

    # Calculate how many images to take from each group
    # allocation, needed_count = calculate_proportional_allocation(group_sizes, needed_count)
    sorted_selected_group_keys, final_selection, total_selected, eligible_groups = select_photos_with_scoring(groups,needed_count)

    # Initialize results and tracking structures
    result = []
    group_clusters = {}

    # Iterate through each group and its allocation
    for group_key in original_groups_names:
        group_df = groups.get_group(group_key).copy()
        # Sort images in this group by image_order (descending)
        sorted_images = group_df.sort_values('total_score', ascending=False)['image_id'].tolist()

        max_take = final_selection.get(group_key, 0)

        # Initialize cluster tracking if not exists
        if group_key not in group_clusters:
            group_clusters[group_key] = []

        if max_take == 0:
            group_clusters[group_key].extend(sorted_images)
            continue

        if len(group_df) < 4 or max_take < 2:
            # Select up to max_take images from this group
            selected_from_group = sorted_images[:max_take]
            result.extend(selected_from_group)
            group_clusters[group_key].extend(sorted_images)
        else:
            # know the shot style
            group_df.loc[:, "shot_style"] = ""
              # check which one has a close shot and which one has a far shot

            for index, row in group_df.iterrows():
                # print("image id and category", row['image_id'], row['cluster_context'])
                shot_style = "unknown"
                # Case 1: Use face sizes if faces are detected
                if row['n_faces'] > 0:
                    faces_info = row["faces_info"]
                    face_sizes = [face.bbox.x2 * face.bbox.y2 for face in faces_info]
                    max_face_size = max(face_sizes)
                    # print("face size", max_face_size)
                    if max_face_size < CONFIGS['FACE_FAR_THRESHOLD']:
                        shot_style = "far"
                    elif max_face_size < CONFIGS['FACE_M_THRESHOLD']:
                        shot_style = "medium"
                    else:
                        shot_style = "close"

                # Case 2: No face, but body detected
                elif row['n_faces'] == 0 and row['number_bodies'] > 0:
                    bodies_info = row["bodies_info"]
                    body_sizes = [body.bbox.x2 * body.bbox.y2 for body in bodies_info]
                    max_body_size = max(body_sizes)
                    # print("body size", max_body_size)
                    if max_body_size < CONFIGS['BODY_FAR_THRESHOLD']:
                        shot_style = "far"
                    elif max_body_size < CONFIGS['BODY_MEDIUM_THRESHOLD']:
                        shot_style = "medium"
                    else:
                        shot_style = "close"

                # Save result in DataFrame
                group_df.at[index, "shot_style"] = shot_style

            #choose images not only by score but also with diverse of three styles.
            # Step 1: Sort group by score descending
            sorted_group = group_df.sort_values('total_score', ascending=False)

            # Step 2: Organize by shot style
            style_buckets = defaultdict(list)
            style_scores = defaultdict(list)
            for _, row in sorted_group.iterrows():
                style = row['shot_style']
                style_buckets[style].append(row['image_id'])
                style_scores[style].append(row['total_score'])

            # Step 3: Round-robin selection for diversity
            selected_from_group = []
            style_mean_scores = {
                style: sum(scores) / len(scores)
                for style, scores in style_scores.items()
            }

            styles_order = sorted(
                style_mean_scores.keys(),
                key=lambda s: style_mean_scores[s],
                reverse=True
            )

            style_indices = {style: 0 for style in styles_order}

            while len(selected_from_group) < max_take:
                added = False
                for style in styles_order:
                    index = style_indices[style]
                    if index < len(style_buckets[style]):
                        selected_from_group.append(style_buckets[style][index])
                        style_indices[style] += 1
                        added = True
                        if len(selected_from_group) >= max_take:
                            break
                if not added:
                    break  # No more images to pick from any style

            # Extend the result and time_clusters
            result.extend(selected_from_group)
            group_clusters[group_key].extend(sorted_images)


        # Early termination if we've reached needed count
        if len(result) >= needed_count:
            result = result[:needed_count]
            break

    for cluster_name in original_groups_names: group_clusters.setdefault(cluster_name, groups.get_group(cluster_name).copy()['image_id'].tolist())

    return result
