import pandas as pd

from utils.illegal_group_splitting_merging import merge_illegal_group, split_illegal_group
from utils.album_tools import get_images_per_groups
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

def merging_process(group_key,groups,illegal_group):
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
        return do_not_change_group(illegal_group, groups,group_key)

    if len(main_groups) == 1:
        selected_cluster = main_groups[0]
        selected_cluster_content_index = list(main_groups[0]['cluster_context'])[0]
        illegal_group.loc[:, 'cluster_context'] = selected_cluster_content_index
        illegal_group.loc[:, 'cluster_context_2nd'] = 'merged'
        updated_group = pd.concat([selected_cluster, illegal_group], ignore_index=False)
    else:
        illegal_group, updated_group, selected_cluster_content_index = merge_illegal_group(main_groups,illegal_group)


    groups = groups.apply(lambda x: update_groups(x, merged=updated_group, merge_group_key=(
        time_cluster_id, selected_cluster_content_index), illegal_group_key=group_key))
    groups = groups.reset_index(drop=True)
    groups = groups.groupby(['time_cluster', 'cluster_context'])
    return groups

def splitting_process(groups,group_key,illegal_group,count):
    updated_group, labels_count = split_illegal_group(illegal_group,count)

    if updated_group is None:
        # we cant split this group
        illegal_group["cluster_context"] = illegal_group["cluster_context"] + "_cant_split"
        updated_group = illegal_group

    groups = update_split_groups(groups, updated_group, group_key)
    groups = groups.reset_index(drop=True)
    groups = groups.groupby(['time_cluster', 'cluster_context'])
    return groups

def handle_wedding_dress(illegal_group,groups,group_key):
    selected_cluster = [group for group_id, group in groups if
                        "bride getting dressed" == group_id[1] or "bride" == group_id[1] or 'getting dressed' ==
                        group_id[1] or 'getting hair-makeup' == group_id[1]]
    if len(selected_cluster) == 0:
        print("Couldnt find a good group for wedding dress!")

        selected_cluster=list(groups)[0][1]

        # return groups
    else:
        selected_cluster = selected_cluster[0]

    illegal_group.loc[:, 'cluster_context'] = selected_cluster['cluster_context'].iloc[0]
    value_to_assign = selected_cluster['time_cluster'].iloc[0]
    illegal_group.loc[:, 'time_cluster'] = value_to_assign
    updated_group = pd.concat([selected_cluster, illegal_group], ignore_index=False)
    groups = groups.apply(lambda x: update_groups(x, merged=updated_group, merge_group_key=(
        value_to_assign, "bride getting dressed"), illegal_group_key=group_key))
    groups = groups.reset_index(drop=True)
    groups = groups.groupby(['time_cluster', 'cluster_context'])
    return groups

def do_not_change_group(illegal_group, groups,group_key):
    """Wont change a group for first dance and cake cutting"""
    illegal_group.loc[:, 'cluster_context'] = illegal_group["cluster_context"] + "_cant_merge"
    groups = groups.apply(
        lambda x: update_not_processed(x, not_processed=illegal_group, illegal_group_key=group_key))
    groups = groups.reset_index(drop=True)
    groups = groups.groupby(['time_cluster', 'cluster_context'])
    return groups

def handle_illegal(group_key,change_tuple,content_cluster_id,illegal_group,imgs_number,groups,look_up_table,is_wedding,count):
    if "first dance" in content_cluster_id or "cake cutting" in content_cluster_id and imgs_number <= CONFIGS['wedding_merge_images_number']:
        return do_not_change_group(illegal_group, groups,group_key)
    elif "wedding dress" in group_key and imgs_number <= CONFIGS['wedding_merge_images_number']:
        """Merge wedding dress into group related to bride"""
        return handle_wedding_dress(illegal_group,groups,group_key)
    elif change_tuple[0] == 'merge':
        return merging_process(group_key,groups,illegal_group)
    elif change_tuple[0] == 'split':
        score = get_merge_split_score(group_key, look_up_table, imgs_number, is_wedding)
        if score >= CONFIGS['min_split_score']:
            return splitting_process(groups,group_key,illegal_group,count)
        else:
            return do_not_change_group(illegal_group, groups, group_key)
    else:
        return do_not_change_group(illegal_group, groups,group_key)


def get_merge_split_score(group_key,lookup_table,imgs_number,is_wedding):
    if is_wedding:
        if "_" in  group_key[1]:
            content_key = group_key[1].split("_")[0]
        else:
            content_key = group_key[1]
        group_value = lookup_table.get(content_key, [10])[0]
    else:
        group_value = lookup_table.get(group_key[0].split("_")[0], [10])[0]

    if group_value == 0:
        # Handle the zero case, e.g., set limited_splitting to a default value or raise an error
        limited_splitting = 0  # or any appropriate value or action
    else:
        limited_splitting = round(imgs_number / group_value)

    return limited_splitting

def update_needed(groups,is_wedding,lookup_table):
    global groups_to_change

    groups_to_change = dict()
    # Check if there is any group with value 1 and its key prefix doesn't have more than one key
    for group_key, imgs_number in groups.items():
        if imgs_number < CONFIGS['max_img_split'] and '_cant_merge' not in group_key[1] and 'None' not in group_key[1]:
            if group_key not in groups_to_change:
                groups_to_change[group_key] = ('merge',0)
        else:
          splitting_score = get_merge_split_score(group_key,lookup_table,imgs_number,is_wedding)
          if splitting_score >= CONFIGS['min_split_score'] and 'cant_split' not in group_key[1] and 'None' not in group_key[1]:
             if group_key not in groups_to_change:
                groups_to_change[group_key] = ('split', splitting_score)
          else:
             continue

    if  len(groups_to_change) > 0:
        return True
    else:
        return False

def process_illegal_groups(group2images, groups, look_up_table, is_wedding, logger):
    count = 2
    try:
        while update_needed(group2images, is_wedding, look_up_table):
            if 'groups_to_change' not in globals():
                logger.error("Error: groups_to_change is not defined.")
                return None, None, None

            for key_to_change, change_tuple in groups_to_change.items():
                try:
                    # Extract content_cluster_id
                    content_cluster_id = key_to_change[1] if '_' not in key_to_change[1] else key_to_change[1].split('_')[0]

                    # Get the illegal group, handling missing keys
                    if key_to_change not in groups.groups:
                        logger.warning(f"Warning: Key {key_to_change} not found in groups.")
                        continue

                    illegal_group = groups.get_group(key_to_change)
                    imgs_number = group2images.get(key_to_change, 0)
                    # Handle illegal groups
                    new_groups = handle_illegal(key_to_change, change_tuple, content_cluster_id, illegal_group, imgs_number, groups,look_up_table,is_wedding, count)

                    if new_groups is not None:
                        group2images = get_images_per_groups(new_groups, logger)  # Ensure logger is passed
                        if isinstance(group2images, str):  # Check for error messages
                            logger.error(group2images)
                            return None, None, None
                        groups = new_groups
                    else:
                        logger.warning(f"Warning: handle_illegal returned None for key {key_to_change}.")
                        continue

                except Exception as e:
                    logger.error(f"Error processing key {key_to_change}: {str(e)}")
                    continue

            logger.info("Iteration completed")
            count += 1

        logger.info(f"Final number of groups for the album: {len(groups)}")
        return groups, group2images, look_up_table

    except Exception as e:
        logger.error(f"Unexpected error in process_illegal_groups: {str(e)}")
        return None, None, None


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