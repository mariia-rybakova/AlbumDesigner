import pandas as pd

from utils.album_tools import get_images_per_groups
from src.groups_operations.groups_splitting_merging import merge_illegal_group_by_time, split_illegal_group_by_time, split_illegal_group_in_certain_point
from utils.configs import CONFIGS


def update_groups(group, merged, merge_group_key, illegal_group_key):
    if merge_group_key == group.name:
        return merged
    if illegal_group_key == group.name:
        return None
    return group


def do_not_change_group(illegal_group, groups, group_key):
    """Wont change a group for first dance and cake cutting"""
    illegal_group.loc[:, 'cluster_context'] = illegal_group["cluster_context"] + "_cant_merge"
    groups = groups.apply(lambda group: illegal_group if group.name == group_key else group)
    groups = groups.reset_index(drop=True)
    groups = groups.groupby(['time_cluster', 'cluster_context'])
    return groups


def get_lut_value(group_key, look_up_table, is_wedding):
    if is_wedding:
        content_key = group_key[1].split("_")[0] if "_" in group_key[1] else group_key[1]
        group_value = look_up_table.get(content_key, [10])[0]
    else:
        group_value = look_up_table.get(group_key[0].split("_")[0], [10])[0]
    return group_value


def get_groups_time(groups):
    general_times_list = list()
    group_key2time_list = dict()
    for group_key, group in groups:
        group_times = group['general_time'].values
        general_times_list.extend(group_times)
        group_key2time_list[group_key] = sorted(group_times)
    return sorted(general_times_list), group_key2time_list


def check_time_based_split_needed(general_times_list, group_time_list, group_key):
    if len(group_time_list) < 2:
        return False, None
    if group_key not in ['walking the aisle', 'bride', 'groom', 'bride and groom', 'settings', 'food', 'detail', 'vehicle', 'inside vehicle', 'rings', 'suit']:
        return False, None

    split_points = list()
    for i in range(len(group_time_list) - 1):
        start_time = group_time_list[i]
        end_time = group_time_list[i + 1]

        count_between = sum(start_time < t < end_time for t in general_times_list)

        if count_between > 2:
            print(f"Splitting needed for {group_key} between {start_time} and {end_time}")
            split_points.append(start_time)

    if len(split_points) > 0:
        return True, split_points

    return False, None


def handle_splitting(groups, group2images, look_up_table, is_wedding):
    # handle splitting
    count = 0

    general_times_list, group_key2time_list = get_groups_time(groups)

    for group_key, imgs_number in group2images.items():
        if imgs_number < CONFIGS['max_img_split']:
            continue

        illegal_group = groups.get_group(group_key)
        if illegal_group.empty:
            continue

        group_spread_size = get_lut_value(group_key, look_up_table, is_wedding)
        splitting_score = round(imgs_number / group_spread_size) if group_spread_size > 0 else 0
        if ((splitting_score > CONFIGS['min_split_score']
            or splitting_score == CONFIGS['min_split_score'] and group_spread_size > 5
            or splitting_score == 2 and group_spread_size >= 12
            or group_spread_size >= 24)
                and 'cant_split' not in group_key[1]):
            updated_group, labels_count = split_illegal_group_by_time(illegal_group, group_spread_size, count)
            count += 1
            split_try = True

        else:
            if_split, split_points = check_time_based_split_needed(general_times_list, group_key2time_list[group_key],
                                                               group_key=group_key[1] if is_wedding else group_key[0].split("_")[0])
            if if_split:
                updated_group, labels_count = split_illegal_group_in_certain_point(illegal_group, split_points, count)
                count += 1
                split_try = True
            else:
                updated_group = None
                split_try = False

        if not split_try:
            continue

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


def process_merging(groups_to_change, groups, merged_targets, logger):
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
                       time_cluster_id == cluster_key[0] and cluster_key != group_to_change_key and
                       merged_targets.get(cluster_key,0) + merged_targets.get(group_to_change_key, 0) < CONFIGS['merge_limit_times'] and
                       len(group) + len(illegal_group) <= CONFIGS['max_imges_per_spread']]

        if len(main_groups) > 0:
            selected_cluster, selected_time_difference = merge_illegal_group_by_time(main_groups, illegal_group, max_images_per_spread=CONFIGS['max_imges_per_spread'])
            selected_cluster_content_index = selected_cluster['cluster_context'].iloc[0]
            merge_target_key = (time_cluster_id, selected_cluster_content_index)
        else:
            selected_cluster, selected_time_difference, merge_target_key = None, float("inf"), None
        merging_candidates.append((illegal_group, group_to_change_key, selected_cluster, selected_time_difference, merge_target_key))

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

        target_merges = 0 if merge_target_key not in merged_targets else merged_targets[merge_target_key]
        change_group_merges = 0 if group_to_change_key not in merged_targets else merged_targets[group_to_change_key]

        merged_targets[merge_target_key] = target_merges + change_group_merges + 1
        merged_targets[group_to_change_key] = target_merges + change_group_merges + 1

    return groups, merged_targets


def process_illegal_groups(group2images, groups, look_up_table, is_wedding, logger=None, max_iterations=500):
    count = 2
    iteration = 0
    merged_targets = dict()
    try:
        groups = handle_splitting(groups, group2images, look_up_table, is_wedding)
        group2images = get_images_per_groups(groups)

        # iteratively merging groups
        while True:
            # Build groups_to_change directly here
            groups_to_change = dict()
            for group_key, imgs_number in group2images.items():
                if imgs_number < CONFIGS['max_img_split'] and '_cant_merge' not in group_key[1]: #and 'None' not in group_key[1]
                    groups_to_change[group_key] = ('merge', 0)

            if not groups_to_change or len(groups_to_change.keys()) == 0:
                logger.info('No groups to change. Iteration: {}. Continue.'.format(iteration))
                break

            if iteration >= max_iterations:
                logger.warning(f"Maximum iterations ({max_iterations}) reached in process_illegal_groups. Groups to change left: {groups_to_change}. Exiting to avoid infinite loop.")
                break

            new_groups, merged_targets = process_merging(groups_to_change, groups, merged_targets, logger)
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