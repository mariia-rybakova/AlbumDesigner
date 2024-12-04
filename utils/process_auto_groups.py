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


def update_group(group):
    to_merge_groups = ["bride getting dressed", "getting hair-makeup", 'accessories']

    if group.name in  to_merge_groups:
        group.loc[:, 'cluster_context'] = "merged_bride getting dressed"
    elif group.name == 'food' or group.name == 'settings':
        group.loc[:, 'time_cluster'] = 1
        group.loc[:, 'cluster_context'] = "merged_" + group.loc[:, 'cluster_context'].values[0]

    return group

def process_auto_groups(sub_grouped):
    sub_grouped = sub_grouped.apply(
        lambda x: update_group(x))
    sub_grouped = sub_grouped.reset_index(drop=True)
    sub_grouped = sub_grouped.groupby(['time_cluster', 'cluster_context'])

    group2images = get_images_per_group(sub_grouped)

    lookup_table = genreate_look_up(group2images)

    return sub_grouped,group2images,lookup_table