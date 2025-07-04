import pandas as pd

from utils.album_tools import get_images_per_groups
from utils.illegal_group_splitting_merging import merge_illegal_group, split_illegal_group
from utils.illegal_group_splitting_merging import merge_illegal_group_by_time, split_illegal_group_by_time
from utils.parser import CONFIGS


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


def update_split_groups(grouped_df, splitted_group, splitting_key):
    """Get all groups except the group that has been splitted then group the splitted group so it becomes 2 groups then add them to the others"""
    # Construct a list of tuples containing the name and data for each group
    new_groups = [(name, group) for name, group in grouped_df if name != splitting_key]
    sub_group_cluster = splitted_group.groupby(['time_cluster', 'cluster_context'])
    #sub_group_cluster = splitted_group.groupby(['scene_order', 'cluster_context'])
    # Add the splitted group to the list of groups
    for sub in sub_group_cluster:
        new_groups.append(sub)

    # Convert the list of groups to a DataFrameGroupBy object
    updated_df = pd.concat([group for _, group in new_groups], ignore_index=True)

    return updated_df


def do_not_change_group(illegal_group, groups,group_key, logger=None):
    """Wont change a group for first dance and cake cutting"""
    illegal_group.loc[:, 'cluster_context'] = illegal_group["cluster_context"] + "_cant_merge"
    groups = groups.apply(
        lambda x: update_not_processed(x, not_processed=illegal_group, illegal_group_key=group_key))
    groups = groups.reset_index(drop=True)
    groups = groups.groupby(['time_cluster', 'cluster_context'])
    return groups


def merging_process(group_key, groups, illegal_group, logger=None):
    time_cluster_id = group_key[0]
    # merge this one with the rest
    main_groups = [group for cluster_key, group in groups if
                   time_cluster_id == cluster_key[0] and cluster_key != group_key]

    if len(main_groups) == 0 and len(illegal_group) == 1:
        time_cluster_id = time_cluster_id - 1
        illegal_group.loc[:, 'time_cluster'] = time_cluster_id
        main_groups = [group for cluster_key, group in groups if
                       time_cluster_id == cluster_key[0] and cluster_key != group_key]
    elif len(main_groups) == 0 and len(illegal_group) > 1:
        return do_not_change_group(illegal_group, groups, group_key, logger)

    illegal_group, updated_group, selected_cluster_content_index = merge_illegal_group_by_time(main_groups,illegal_group)

    groups = groups.apply(lambda x: update_groups(x, merged=updated_group,
                                                  merge_group_key=(time_cluster_id, selected_cluster_content_index),
                                                  illegal_group_key=group_key))
    groups = groups.reset_index(drop=True)
    groups = groups.groupby(['time_cluster', 'cluster_context'])
    return groups


def splitting_process(groups,group_key,illegal_group,count, logger=None):
    updated_group, labels_count = split_illegal_group_by_time(illegal_group,count)

    if updated_group is None:
        # we cant split this group
        illegal_group["cluster_context"] = illegal_group["cluster_context"] + "_cant_split"
        updated_group = illegal_group

    groups = update_split_groups(groups, updated_group, group_key)
    groups = groups.reset_index(drop=True)
    groups = groups.groupby(['time_cluster', 'cluster_context'])
    return groups


def handle_wedding_dress(illegal_group, groups, group_key, logger=None):
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


def handle_illegal(group_key, change_tuple, imgs_number, groups, count, logger=None):
    illegal_group = groups.get_group(group_key)
    if illegal_group.empty:
        logger.info(f"Illegal group {group_key} is empty. Skipping.")
        return groups

    content_cluster_id = group_key[1] if '_' not in group_key[1] else group_key[1].split('_')[0]
    if imgs_number == 1 and illegal_group.iloc[0]['image_orientation'] == 'portrait':
        return merging_process(group_key, groups, illegal_group, logger)
    if ("first dance" in content_cluster_id or "cake cutting" in content_cluster_id) and imgs_number <= CONFIGS['wedding_merge_images_number']:
        return do_not_change_group(illegal_group, groups, group_key, logger)
    if "wedding dress" in group_key and imgs_number <= CONFIGS['wedding_merge_images_number']:
        return handle_wedding_dress(illegal_group, groups, group_key, logger)
    if change_tuple[0] == 'merge':
        return merging_process(group_key, groups, illegal_group, logger)
    if change_tuple[0] == 'split':
        score = change_tuple[1]
        if score >= CONFIGS['min_split_score']:
            return splitting_process(groups, group_key, illegal_group, count, logger)
        else:
            return do_not_change_group(illegal_group, groups, group_key, logger)

    return do_not_change_group(illegal_group, groups, group_key, logger)


def get_split_score(group_key, lookup_table, imgs_number, is_wedding):
    if is_wedding:
        content_key = group_key[1].split("_")[0] if "_" in group_key[1] else group_key[1]
        group_value = lookup_table.get(content_key, [10])[0]
    else:
        group_value = lookup_table.get(group_key[0].split("_")[0], [10])[0]

    limited_splitting = round(imgs_number / group_value) if group_value > 0 else 0

    return limited_splitting


def process_illegal_groups(group2images, groups, look_up_table, is_wedding, logger=None, max_iterations=20):
    count = 2
    iteration = 0
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
                logger.warning(f"Maximum iterations ({max_iterations}) reached in process_illegal_groups. Exiting to avoid infinite loop.")
                break

            for key_to_change, change_tuple in groups_to_change.items():
                imgs_number = group2images.get(key_to_change, 0)
                new_groups = handle_illegal(key_to_change, change_tuple, imgs_number, groups, count, logger)
                if new_groups is not None:
                    group2images = get_images_per_groups(new_groups)
                    groups = new_groups
                else:
                    logger.info(f"handle_illegal returned None for key {key_to_change}. Skipping. {groups.groups}")
                    continue

            logger.info("Iteration completed")
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