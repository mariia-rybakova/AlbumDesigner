import pandas as pd
from networkx.utils import groups

from utils.illegal_group_splitting_merging import merge_illegal_group, split_illegal_group
from utils.lookup_table_tools import get_lookup_table
from utils.protobufs_processing import generate_people_clustering


def update_groups(group, merged, merge_group_key, illegal_group_key):
    if merge_group_key == group.name:
        return merged
    elif illegal_group_key == group.name:
        return
    else:
        return group

def update_notprocessed(group, not_processed, illegal_group_key):
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


def get_images_per_group(groups):
    """
    Return: dict group_name to list of images data
    """
    group2images_data_list = dict()
    for key, group in groups:
        num_images = len(group)
        group2images_data_list[f'{key[0]}_{key[1]}'] = num_images
    return group2images_data_list

def merging_process(group_key,groups,illegal_group):
    time_cluster_id = group_key[0]
    # merge this one with the rest
    main_groups = [group for cluster_key, group in groups if
                   time_cluster_id == cluster_key[0] and cluster_key != group_key]

    if len(main_groups) == 1:
        illegal_group.loc[:, 'cluster_context'] = illegal_group["cluster_context"] + "_cant_merge"
        groups = groups.apply(
            lambda x: update_notprocessed(x, not_processed=illegal_group,
                                          illegal_group_key=group_key))
        groups = groups.reset_index(drop=True)
        groups = groups.groupby(['time_cluster', 'cluster_context'])

    else:
        illegal_group, updated_group, selected_cluster_content_index = merge_illegal_group(main_groups,
                                                                                           illegal_group)

    groups = groups.apply(lambda x: update_groups(x, merged=updated_group, merge_group_key=(
        time_cluster_id, selected_cluster_content_index), illegal_group_key=group_key))
    groups = groups.reset_index(drop=True)
    groups = groups.groupby(['time_cluster', 'cluster_context'])
    return groups

def splitting_process(groups,group_key,illegal_group):
    updated_group, labels_count = split_illegal_group(illegal_group)

    if updated_group is None:
        # we cant split this group
        illegal_group["cluster_context"] = illegal_group["cluster_context"] + "_cant_split"
        updated_group = illegal_group

    groups = update_split_groups(groups, updated_group, group_key)
    groups = groups.reset_index(drop=True)
    groups = groups.groupby(['time_cluster', 'cluster_context'])
    return groups


def handle_illegal(group_key,score,content_cluster_id,illegal_group,imgs_number,groups):
    if "first dance" in content_cluster_id or "cake cutting" in content_cluster_id and imgs_number <= 3:
        """Wont change a group for first dance and cake cutting"""
        illegal_group.loc[:, 'cluster_context'] = illegal_group["cluster_context"] + "_cant_merge"
        groups = groups.apply(
            lambda x: update_notprocessed(x, not_processed=illegal_group, illegal_group_key=group_key))
        groups = groups.reset_index(drop=True)
        groups = groups.groupby(['time_cluster', 'cluster_context'])
        return groups

    elif "wedding dress" in group_key and imgs_number <= 3:
        """Merge wedding dress into group related to bride"""
        selected_cluster = [group for group_id, group in groups if
                            "bride getting dressed" == group_id[1] or "bride" == group_id[1] or 'getting dressed' ==
                            group_id[1] or 'getting hair-makeup' == group_id[1]]
        if len(selected_cluster) == 0:
            print("Couldnt find a good group for wedding dress!")
            return groups
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

    elif imgs_number < 3:
        return merging_process(group_key,groups,illegal_group)

    elif score >= 4:
        return splitting_process(groups,group_key,illegal_group)



def update_needed(groups,lookup_table):
    global groups_to_change

    groups_to_change = dict()
    # Check if there is any group with value 1 and its key prefix doesn't have more than one key
    for group_key, imgs_number in groups.items():
        if is_wedding:
            group_value = lookup_table.get(group_key[1], [0])[0]
        else:
            group_value = lookup_table.get(group_key[0].split("_")[0])[0]

        if group_value == 0:
            # Handle the zero case, e.g., set limited_splitting to a default value or raise an error
            limited_splitting = 0  # or any appropriate value or action
        else:
            limited_splitting = round(imgs_number / group_value)

        if (imgs_number < 3 or (limited_splitting >= 4 and 'cant_split' not in group_key)) and 'cant_merge' not in group_key:
            if group_key not in groups_to_change:
                groups_to_change[group_key] = limited_splitting
        else:
            continue

    if  len(groups_to_change) > 0:
        return True
    else:
        return False



def process_illegal_groups(group2images,groups,look_up_table,is_wedding, logger=None):
    count = 2
    while update_needed(group2images, look_up_table):
        for key_to_change,score in groups_to_change.items():
            content_cluster_id = key_to_change[1] if '_' not in key_to_change[1] else key_to_change[1].split('_')[0]
            illegal_group = groups.get_group(key_to_change)
            imgs_number = group2images[key_to_change]
            groups = handle_illegal(key_to_change,score,content_cluster_id,illegal_group,imgs_number,groups)

        count += 1
        group2images = get_images_per_group(groups)
        look_up_table = get_lookup_table(group2images,is_wedding)

    print(f"Final number of groups for the album {len(groups)}")
    return groups, group2images

def get_images_per_groups(original_groups):
    group2images_data_list = dict()
    for name_group, group_df in original_groups:
        num_images = len(group_df)
        group2images_data_list[name_group] = num_images
    return group2images_data_list

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