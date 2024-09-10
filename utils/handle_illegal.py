import pandas as pd
from .illegal_group_splitting_merging import merge_illegal_group, split_illegal_group
from .lookup_table import genreate_look_up


def get_images_per_group(groups):
    """
    Return: dict group_name to list of images data
    """
    group2images_data_list = dict()
    for key, group in groups:
        num_images = len(group)
        group2images_data_list[f'{key[0]}_{key[1]}'] = num_images
    return group2images_data_list

def get_splitting_core(group_key,lookup_table,content_cluster_id,imgs_number):
    if group_key not in lookup_table.keys():
        if content_cluster_id in lookup_table:
            group_value = lookup_table.get(content_cluster_id)[0]
        else:
            group_value = 22
    else:
        group_value = lookup_table.get(group_key)[0]

    if group_value == 0:
        # Handle the zero case, e.g., set limited_splitting to a default value or raise an error
        limited_splitting = 0  # or any appropriate value or action
    else:
        limited_splitting = round(imgs_number / group_value)
    return limited_splitting

def process_illegal_groups(images_per_group, sub_grouped,logger=None):
    def update_needed(groups):
        # Collect all keys' prefixes
        prefixes = {}
        for group_key in groups:
            prefix = group_key.split("_")[0]
            if prefix not in prefixes:
                prefixes[prefix] = []
            prefixes[prefix].append(group_key)

        # Check if there is any group with value 1 and its key prefix doesn't have more than one key
        for group_key, imgs_number in groups.items():
            if group_key not in lookup_table:
                id = group_key.split("_")[1]
                group_value = lookup_table.get(id, [0])[0]
            else:
                group_value = lookup_table.get(group_key, [0])[0]

            if group_value == 0:
                # Handle the zero case, e.g., set limited_splitting to a default value or raise an error
                limited_splitting = 0  # or any appropriate value or action
            else:
                limited_splitting = round(imgs_number / group_value)

            # Check if the group has value 1 and its key's prefix has more than one key
            prefix = group_key.split("_")[0]

            if (imgs_number <= 3 or (limited_splitting >= 4 and 'cant_split' not in group_key)) and not len(prefixes.get(prefix, [])) == 1:
                return True

        return False

    def update_groups(group, merged, merge_group_key, illegal_group_key):
        if merge_group_key == group.name:
            return merged
        elif illegal_group_key == group.name:
            return
        else:
            return group

    def update_split_groups(grouped_df, splitted_group, splitting_key):
        """Get all groups except the group that has been splitted then group the splitted group so it becomes 2 groups then add them to the others"""
        # Construct a list of tuples containing the name and data for each group
        new_groups = [(name, group) for name, group in grouped_df if name != splitting_key]
        sub_group_cluster = splitted_group.groupby(['image_time', 'cluster_context'])
        # Add the splitted group to the list of groups
        for sub in sub_group_cluster:
            new_groups.append(sub)

        # Convert the list of groups to a DataFrameGroupBy object
        updated_df = pd.concat([group for _, group in new_groups], ignore_index=True)

        return updated_df

    lookup_table = genreate_look_up(images_per_group)

    count = 2
    while update_needed(images_per_group):
        keys_to_delete = []
        keys_to_update = {}

        for group_key, imgs_number in list(images_per_group.items()):
            parts = group_key.split('_')
            time_cluster_id_str = parts[0]
            time_cluster_id_float = float(time_cluster_id_str)
            content_cluster_id = '_'.join(parts[1:])

            if group_key in keys_to_update.keys():
                imgs_number = keys_to_update[group_key]
                splitting_score = get_splitting_core(parts[1],lookup_table,content_cluster_id,imgs_number)
            else:
                splitting_score = get_splitting_core(parts[1],lookup_table,content_cluster_id,imgs_number)

            intended_group_key = (time_cluster_id_float, content_cluster_id)

            if "wedding dress" in group_key and imgs_number <= 3:
                selected_cluster = [group for group_id, group in sub_grouped if "bride getting dressed" == group_id[1] or "bride" == group_id[1] or 'getting dressed' == group_id[1] or 'getting hair-makeup' == group_id[1]]
                if len(selected_cluster) == 0:
                    print("Couldnt find a good group for wedding dress!")
                    continue
                else:
                    selected_cluster = selected_cluster[0]


                illegal_group = sub_grouped.get_group((time_cluster_id_float, content_cluster_id))
                illegal_group.loc[:, 'cluster_context'] = selected_cluster['cluster_context'].iloc[0]
                value_to_assign = selected_cluster['general_time'].iloc[0]
                illegal_group.loc[:, 'general_time'] = value_to_assign
                updated_group = pd.concat([selected_cluster, illegal_group], ignore_index=False)


                keys_to_delete.append(f'{group_key}')
                keys_to_update[
                    f'{selected_cluster["general_time"].values[0]}_{selected_cluster["cluster_context"].iloc[0]}'] = len(
                    updated_group)

                sub_grouped = sub_grouped.apply(lambda x: update_groups(x, merged=updated_group, merge_group_key=(
                    value_to_assign, "bride getting dressed"), illegal_group_key=intended_group_key))
                sub_grouped = sub_grouped.reset_index(drop=True)
                sub_grouped = sub_grouped.groupby(['general_time', 'cluster_context'])

            elif imgs_number <= 3:
                # merge this one with the rest
                main_groups = [group for cluster_key, group in sub_grouped if
                               time_cluster_id_float == cluster_key[0]]


                if len(main_groups) == 1:
                    logger.info(f"main time group has one group {intended_group_key}_ we cant do further merging for this group!!!")
                    continue
                else:
                    intended_group_index = None
                    illegal_group = sub_grouped.get_group(intended_group_key)
                    if len(illegal_group) == 0 or illegal_group is None:
                        logger.warning(f"Could'nt find illegal group {intended_group_key} inside the sub grouped")
                        continue

                    # get the index of illegal group inside the main groups list
                    for group_index, group in enumerate(main_groups):
                        if group.equals(illegal_group):
                            intended_group_index = group_index
                            break

                    if intended_group_index is None:
                        logger.warning(f"Cant find illegal group inside the main groups so we cant process splitting or merging for group {intended_group_key}")
                        illegal_group["cluster_context"] = illegal_group["cluster_context"] + "_cant_split"
                        continue

                    illegal_group, updated_group, selected_cluster_content_index = merge_illegal_group(main_groups,
                                                                                                       illegal_group,
                                                                                                       intended_group_index)
                # update subgroup that has same time_cluster_id with updated group
                keys_to_delete.append(f'{time_cluster_id_str}_{content_cluster_id}')
                keys_to_update[f'{time_cluster_id_str}_{selected_cluster_content_index}'] = len(updated_group)

                sub_grouped = sub_grouped.apply(lambda x: update_groups(x, merged=updated_group, merge_group_key=(
                    time_cluster_id_float, selected_cluster_content_index), illegal_group_key=intended_group_key))
                sub_grouped = sub_grouped.reset_index(drop=True)
                sub_grouped = sub_grouped.groupby(['general_time', 'cluster_context'])

            elif splitting_score >= 4:
                # split it
                illegal_group = sub_grouped.get_group(intended_group_key)

                updated_group, labels_count = split_illegal_group(illegal_group,count,logger)

                if updated_group is not None:
                    # update images per group
                    keys_to_delete.append(f'{time_cluster_id_str}_{content_cluster_id}')
                    for k, val in labels_count.items():
                        keys_to_update[f'{time_cluster_id_str}_{k}'] = val
                else:
                    # we cant split this group
                    illegal_group["cluster_context"] = illegal_group["cluster_context"] + "_cant_split"
                    updated_group = illegal_group

                sub_grouped = update_split_groups(sub_grouped, updated_group, intended_group_key)
                sub_grouped = sub_grouped.reset_index(drop=True)
                sub_grouped = sub_grouped.groupby(['general_time', 'cluster_context'])

        count += 1
        images_per_group = get_images_per_group(sub_grouped)
        lookup_table = genreate_look_up(images_per_group)

    number_groups = len(sub_grouped)
    logger.info(f"Number of groups to create an album {number_groups}")
    return sub_grouped, images_per_group, lookup_table
