import pandas as pd

from utils.album_tools import get_images_per_groups
from src.groups_operations.groups_splitting_merging import merge_illegal_group_by_time, split_illegal_group_by_time
from utils.configs import CONFIGS


def update_groups(group, merged, merge_group_key, illegal_group_key):
    if merge_group_key == group.name:
        return merged
    elif illegal_group_key == group.name:
        return
    else:
        return group


def update_not_processed(group, not_processed, illegal_group_key):
    if group.name == illegal_group_key:
        return not_processed
    else:
        return group


def do_not_change_group(illegal_group, groups, group_key):
    """Wont change a group for first dance and cake cutting"""
    illegal_group.loc[:, 'cluster_context'] = illegal_group["cluster_context"] + "_cant_merge"
    groups = groups.apply(
        lambda x: update_not_processed(x, not_processed=illegal_group, illegal_group_key=group_key))
    groups = groups.reset_index(drop=True)
    groups = groups.groupby(['time_cluster', 'cluster_context'])
    return groups


def get_split_score(group_key, lookup_table, imgs_number, is_wedding):
    if is_wedding:
        content_key = group_key[1].split("_")[0] if "_" in group_key[1] else group_key[1]
        group_value = lookup_table.get(content_key, [10])[0]
    else:
        group_value = lookup_table.get(group_key[0].split("_")[0], [10])[0]

    limited_splitting = round(imgs_number / group_value) if group_value > 0 else 0

    return limited_splitting


def splitting_process(groups, group_key, illegal_group, count):
    updated_group, labels_count = split_illegal_group_by_time(illegal_group, count)

    if updated_group is None:
        # we can't split this group
        illegal_group["cluster_context"] = illegal_group["cluster_context"] + "_cant_split"
        updated_group = illegal_group

    # Construct a list of tuples containing the name and data for each group
    new_groups = [(name, group) for name, group in groups if name != group_key]
    sub_group_cluster = updated_group.groupby(['time_cluster', 'cluster_context'])
    for sub in sub_group_cluster:
        new_groups.append(sub)

    # Convert the list of groups to a DataFrameGroupBy object
    groups = pd.concat([group for _, group in new_groups], ignore_index=True)
    groups = groups.reset_index(drop=True)
    groups = groups.groupby(['time_cluster', 'cluster_context'])
    return groups


def merge_groups(groups, illegal_group, illegal_group_key, selected_cluster, merge_target_key):
    illegal_group.loc[:, 'cluster_context'] = merge_target_key[1]
    illegal_group.loc[:, 'cluster_context_2nd'] = 'merged'
    combined_group = pd.concat([selected_cluster, illegal_group], ignore_index=False)

    groups = groups.apply(lambda x: update_groups(x, merged=combined_group,
                                                  merge_group_key=merge_target_key,
                                                  illegal_group_key=illegal_group_key))
    groups = groups.reset_index(drop=True)
    groups = groups.groupby(['time_cluster', 'cluster_context'])
    return groups


def handle_wedding_dress(illegal_group, groups, group_key, logger):
    merge_candidates = [group for group_id, group in groups if
                        group_id[1] in ["bride getting dressed", "bride", "getting dressed", "getting hair-makeup"]]
    if len(merge_candidates) == 0:
        logger.warning("Couldnt find a good group for wedding dress. Take the first one.")
        selected_group = list(groups)[0][1]
    else:
        selected_group = merge_candidates[0]

    illegal_group.loc[:, 'cluster_context'] = selected_group['cluster_context'].iloc[0]
    value_to_assign = selected_group['time_cluster'].iloc[0]
    illegal_group.loc[:, 'time_cluster'] = value_to_assign
    updated_group = pd.concat([selected_group, illegal_group], ignore_index=False)
    groups = groups.apply(lambda x: update_groups(x, merged=updated_group, merge_group_key=(value_to_assign, "bride getting dressed"), illegal_group_key=group_key))
    groups = groups.reset_index(drop=True)
    groups = groups.groupby(['time_cluster', 'cluster_context'])
    return groups


def handle_illegal(group_key, change_tuple, imgs_number, groups, count, logger):
    illegal_group = groups.get_group(group_key)
    if illegal_group.empty:
        logger.info(f"Illegal group {group_key} is empty. Skipping.")
        return groups

    content_cluster_id = group_key[1] if '_' not in group_key[1] else group_key[1].split('_')[0]
    if ("first dance" in content_cluster_id or "cake cutting" in content_cluster_id) and imgs_number <= CONFIGS['wedding_merge_images_number']:
        return do_not_change_group(illegal_group, groups, group_key)
    if "wedding dress" in group_key and imgs_number <= CONFIGS['wedding_merge_images_number']:
        return handle_wedding_dress(illegal_group, groups, group_key, logger)
    if change_tuple[0] == 'split':
        score = change_tuple[1]
        if score >= CONFIGS['min_split_score']:
            return splitting_process(groups, group_key, illegal_group, count)
        else:
            return do_not_change_group(illegal_group, groups, group_key)

    return groups

def add_class_preference(merge_candidates):
    """Modifies time difference based on content class pairs"""
    for i, candidate in enumerate(merge_candidates):
        illegal_group, group_key, selected_cluster, time_diff, merge_target_key = candidate
        if not all([group_key, merge_target_key]):
            continue

        source_class = group_key[1].split('_')[0] if '_' in group_key[1] else group_key[1]
        target_class = merge_target_key[1].split('_')[0] if '_' in merge_target_key[1] else merge_target_key[1]

        similar_classes = [['bride','bride getting dressed','getting hair-makeup','accessories'],['bride','groom'],['ceremony', 'walking the aisle'],['food','settings']]

        # Prefer merging similar classes
        if source_class == target_class:
            time_diff *= 0.2
        # Prefer merging related classes
        for similar_list in similar_classes:
            if source_class in similar_list and target_class in similar_list:
                time_diff *= 0.5
                break
        merge_candidates[i] = (illegal_group, group_key, selected_cluster, time_diff, merge_target_key)
    return merge_candidates


def handle_merging(groups_to_change, groups, merged_targets, logger):
    merging_candidates = list()
    current_merges = set()

    for group_to_change_key, change_tuple in groups_to_change.items():
        if change_tuple[0] != 'merge':
            continue
        if group_to_change_key in merged_targets:

            none_or_other = 'None' in group_to_change_key or 'other' in group_to_change_key

            if (not none_or_other and merged_targets[group_to_change_key] >= CONFIGS['merge_limit_times']) or (none_or_other and merged_targets[group_to_change_key] >= CONFIGS['none_limit_times'] ):
                # logger.info(f"Skipping group {group_to_change_key} as it was already merged into.")
                continue

        # Check if group_to_change_key exists in groups
        if group_to_change_key not in groups.groups:
            illegal_group = None
        else:
            illegal_group = groups.get_group(group_to_change_key)
        if illegal_group is None or illegal_group.empty:
            continue

        time_cluster_id = group_to_change_key[0]
        main_groups = [group for cluster_key, group in groups if
                       time_cluster_id == cluster_key[0] and cluster_key != group_to_change_key and merged_targets.get(cluster_key,0)<CONFIGS['merge_limit_times'] and len(group) + len(illegal_group) <= CONFIGS['max_imges_per_spread']]

        if len(main_groups) > 0:
            selected_cluster, selected_time_difference = merge_illegal_group_by_time(main_groups, illegal_group)
            selected_cluster_content_index = selected_cluster['cluster_context'].iloc[0]
            merge_target_key = (time_cluster_id, selected_cluster_content_index)
        else:
            selected_cluster, selected_time_difference, merge_target_key = None, float("inf"), None
        merging_candidates.append((illegal_group, group_to_change_key, selected_cluster, selected_time_difference, merge_target_key))

    add_class_preference(merging_candidates)

    merging_candidates = add_class_preference(merging_candidates)
    merging_candidates = sorted(merging_candidates, key=lambda x: x[3])
    for illegal_group, group_to_change_key, selected_cluster, selected_time_difference, merge_target_key in merging_candidates:
        if selected_cluster is None:
            do_not_change_group(illegal_group, groups, group_to_change_key)
            continue
        if merge_target_key in current_merges:
            # logger.info(f"Skipping merge for {group_to_change_key} as target {merge_target_key} already merged into.")
            continue

        groups = merge_groups(groups, illegal_group, group_to_change_key, selected_cluster, merge_target_key)
        # logger.info(f"Group {group_to_change_key} was merger to {merge_target_key}.")

        current_merges.add(merge_target_key)
        current_merges.add(group_to_change_key)

        if merge_target_key in merged_targets:
            merged_targets[merge_target_key] += 1
        else:
            merged_targets[merge_target_key] = 1
        if group_to_change_key in merged_targets:
            merged_targets[group_to_change_key] += 1
        else:
            merged_targets[group_to_change_key] = 1

    return groups, merged_targets


def process_illegal_groups(group2images, groups, look_up_table, is_wedding, logger=None, max_iterations=500):
    count = 2
    iteration = 0
    merged_targets = dict()
    try:
        while True:
            # Build groups_to_change directly here
            groups_to_change = dict()
            for group_key, imgs_number in group2images.items():
                if imgs_number < CONFIGS['max_img_split'] and '_cant_merge' not in group_key[1]: #and 'None' not in group_key[1]
                    groups_to_change[group_key] = ('merge', 0)
                else:
                    splitting_score = get_split_score(group_key, look_up_table, imgs_number, is_wedding)
                    if splitting_score >= CONFIGS['min_split_score'] and 'cant_split' not in group_key[1]: #and 'None' not in group_key[1]
                        groups_to_change[group_key] = ('split', splitting_score)

            if not groups_to_change or len(groups_to_change.keys()) == 0:
                logger.info('No groups to change. Continue.')
                break

            if iteration >= max_iterations:
                logger.warning(f"Maximum iterations ({max_iterations}) reached in process_illegal_groups. Groups to change left: {groups_to_change}. Exiting to avoid infinite loop.")
                break

            if iteration==0:
                for key_to_change, change_tuple in groups_to_change.items():
                    if key_to_change in merged_targets:
                        # logger.info(f"Skipping group {key_to_change} as it was already merged into.")
                        continue
                    imgs_number = group2images.get(key_to_change, 0)
                    new_groups = handle_illegal(key_to_change, change_tuple, imgs_number, groups, count, logger)
                    if new_groups is not None:
                        group2images = get_images_per_groups(new_groups)
                        groups = new_groups
                    else:
                        logger.info(f"handle_illegal returned None for key {key_to_change}. Skipping. {groups.groups}")
                        continue
            new_groups, merged_targets = handle_merging(groups_to_change, groups, merged_targets, logger)
            if new_groups is not None:
                new_group2images = get_images_per_groups(new_groups)
                if len(new_group2images) == len(group2images):
                    logger.info(f"No changes in groups after merging at iteration {iteration}. Exiting.")
                    break
                else:
                    group2images = new_group2images
                groups = new_groups

            # logger.info("Iteration completed")
            count += 1
            iteration += 1
    except Exception as ex:
        logger.error(f"Unexpected error in process_illegal_groups: {str(ex)}")
        return None, None, None

    logger.info(f"Final number of groups for the album: {len(groups)}")
    return groups, group2images, look_up_table


# if __name__ == "__main__":
#     is_wedding= True
#     df  = pd.read_excel(r'C:\Users\karmel\Desktop\AlbumDesigner\rightWedding.xlsx')
#     #df = pd.read_excel(r'C:\Users\karmel\Desktop\AlbumDesigner\nonWeddingg.xlsx')
#     if is_wedding:
#         groups = df.groupby(['time_cluster', 'cluster_context'])
#     else:
#         df = generate_people_clustering(df)
#         groups = df.groupby(['people_cluster'])
#
#     group2images = get_images_per_groups(groups)
#     look_up_table = get_lookup_table(group2images, is_wedding)
#     process_illegal_groups(group2images, groups, look_up_table, is_wedding, None)