import pandas as pd

from utils.album_tools import get_images_per_groups
from src.groups_operations.groups_splitting_merging import merge_illegal_group_by_time, split_illegal_group_by_time, split_illegal_group_in_certain_point
from utils.configs import CONFIGS

BRIDE_CENTRIC_CLASSES = ['bride', 'bride party', 'wedding dress', 'getting hair-makeup', 'bride getting dressed']
GROOM_CENTRIC_CLASSES = ['groom', 'groom party', 'suit']


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
            split_points.append(start_time)

    if len(split_points) > 0:
        return True, split_points

    return False, None


def handle_wedding_splitting(photos_df, look_up_table ,logger=None):
    # handle splitting
    split_df = photos_df[photos_df['group_size'] >= CONFIGS['max_img_split']]
    split_groups = split_df.groupby(['time_cluster', 'cluster_context', 'group_sub_index'])
    general_times_list, group_key2time_list = get_groups_time(split_groups)
    count=0
    for group_key, group in split_groups:
        group_spread_size = look_up_table.get(group_key[1], [10])[0]
        splitting_score = round(group['group_size'].iloc[0] / group_spread_size) if group_spread_size > 0 else 0
        if ((splitting_score > CONFIGS['min_split_score']
             or splitting_score == CONFIGS['min_split_score'] and group_spread_size > 5
             or splitting_score == 2 and group_spread_size >= 12
             or group_spread_size >= 24)
                and 'cant_split' not in group_key[1]):

            updated_group, labels_count = split_illegal_group_by_time(group, group_spread_size, count)
            count += 1

        else:
            if_split, split_points = check_time_based_split_needed(general_times_list, group_key2time_list[group_key],
                                                                   group_key=group_key[1])
            if if_split:
                updated_group, labels_count = split_illegal_group_in_certain_point(group, split_points, count)
                count += 1
            else:
                updated_group = None
        if updated_group is not None:
           for row_index in updated_group.index:
               sub_index = int(updated_group.loc[row_index, 'cluster_context'].split('_')[-2])
               photos_df.loc[row_index, 'group_sub_index'] = sub_index

    photo_groups = photos_df.groupby(['time_cluster', 'cluster_context', 'group_sub_index'])
    for group_key, group in photo_groups:
        group_size = len(group)
        photos_df.loc[group.index, 'group_size'] = group_size
    return photos_df


def handle_wedding_bride_groom_merge(photos_df, logger=None):
    merge_df = photos_df[(photos_df['group_size'] < CONFIGS['max_img_split']) & ((photos_df['cluster_context'].isin(BRIDE_CENTRIC_CLASSES)) | (photos_df['cluster_context'].isin(GROOM_CENTRIC_CLASSES)))]

    # mask = photos_df.apply(tuple, axis=1).isin(merge_df.apply(tuple, axis=1))
    # targets_df = photos_df[~mask]

    targets_df = photos_df.copy()

    merge_groups = merge_df.groupby(['time_cluster', 'cluster_context', 'group_sub_index'])

    general_times_list, _ = get_groups_time(photos_df.groupby(['time_cluster', 'cluster_context', 'group_sub_index']))

    merge_candidates = list()

    for group_key, group in merge_groups:
        merge_targets = targets_df[(targets_df['time_cluster'] == group_key[0]) & (targets_df['group_size'] + len(group) <= CONFIGS['max_imges_per_spread'])]
        merge_target_groups = merge_targets.groupby(['time_cluster', 'cluster_context', 'group_sub_index'])
        main_groups = [m_group for _, m_group in merge_target_groups]
        selected_cluster, selected_time_difference = merge_illegal_group_by_time(main_groups, group,
                                                                                 general_times_list,
                                                                                 max_images_per_spread=CONFIGS['max_imges_per_spread'])

        if selected_cluster is not None:
            merge_candidates.append((group_key,selected_cluster, selected_time_difference))

    merge_candidates = sorted(merge_candidates, key=lambda x: x[2])
    current_merges = set()
    for group_key, selected_cluster, selected_time_difference in merge_candidates:
        to_merge_group = merge_groups.get_group(group_key)
        selected_key = (selected_cluster['time_cluster'].iloc[0], selected_cluster['cluster_context'].iloc[0], selected_cluster['group_sub_index'].iloc[0])
        if group_key in current_merges or selected_key in current_merges:
            continue
        if (group_key[1] in BRIDE_CENTRIC_CLASSES and selected_cluster['cluster_context'].iloc[0] in GROOM_CENTRIC_CLASSES) or (group_key[1] in GROOM_CENTRIC_CLASSES and selected_cluster['cluster_context'].iloc[0] in BRIDE_CENTRIC_CLASSES):
            if abs(len(to_merge_group) - len(selected_cluster)) >= 2:
                min_len = min(len(to_merge_group), len(selected_cluster))
                merged_group = pd.concat([to_merge_group.head(min_len), selected_cluster.head(min_len)])
                reminder_group = pd.concat([to_merge_group.tail(len(to_merge_group)-min_len), selected_cluster.tail(len(selected_cluster)-min_len)])
                for row_index in reminder_group.index:
                    photos_df.loc[row_index, 'group_size'] = len(reminder_group)
                new_sub_index = photos_df['group_sub_index'].max() + 1
                for row_index in merged_group.index:
                    photos_df.loc[row_index, 'cluster_context'] = selected_cluster['cluster_context'].iloc[0]
                    photos_df.loc[row_index, 'groups_merged'] = to_merge_group['groups_merged'].iloc[0] + selected_cluster['groups_merged'].iloc[0]
                    photos_df.loc[row_index, 'group_size'] = len(merged_group)
                    photos_df.loc[row_index, 'group_sub_index'] = new_sub_index
                    photos_df.loc[row_index, 'merge_allowed'] = False
                current_merges.add(group_key)
                current_merges.add(selected_key)

    return photos_df


def process_wedding_merging(photos_df, logger=None):
    mask_special = photos_df['cluster_context'].isin(['None', 'other'])
    df_special = photos_df[mask_special].copy()
    df_regular = photos_df[~mask_special].copy()

    merge_special_df = df_special[(df_special['group_size'] < CONFIGS['max_img_split']) &
                         (df_special['merge_allowed'] == True) &
                         (df_special['groups_merged'] < CONFIGS['none_limit_times'])]
    merge_regular_df = df_regular[(df_regular['group_size'] < CONFIGS['max_img_split']) &
                                 (df_regular['merge_allowed'] == True) &
                                 (df_regular['groups_merged'] < CONFIGS['merge_limit_times'])]
    merge_df = pd.concat([merge_special_df, merge_regular_df])
    merge_groups = merge_df.groupby(['time_cluster', 'cluster_context', 'group_sub_index'])
    if merge_groups.ngroups == 0:
        return photos_df, False

    # mask = photos_df.apply(tuple, axis=1).isin(merge_df.apply(tuple, axis=1))
    # targets_df = photos_df[~mask]

    targets_df = photos_df.copy()

    targets_df = targets_df[(targets_df['merge_allowed'] == True) & (targets_df['groups_merged'] < CONFIGS['merge_limit_times'])]

    general_times_list, _ = get_groups_time(photos_df.groupby(['time_cluster', 'cluster_context', 'group_sub_index']))

    merge_candidates = list()

    for group_key, group in merge_groups:
        merge_targets = targets_df[(targets_df['time_cluster'] == group_key[0]) &
                                   (targets_df['group_size'] + len(group) <= CONFIGS['max_imges_per_spread'])]
        merge_target_groups = merge_targets.groupby(['time_cluster', 'cluster_context', 'group_sub_index'])
        main_groups = [m_group for _, m_group in merge_target_groups]
        selected_cluster, selected_time_difference = merge_illegal_group_by_time(main_groups, group,
                                                                                 general_times_list,
                                                                                 max_images_per_spread=CONFIGS['max_imges_per_spread'])

        if selected_cluster is not None:
            merge_candidates.append((group_key, selected_cluster, selected_time_difference))

    if len(merge_candidates) == 0:
        return photos_df, False

    merge_candidates = sorted(merge_candidates, key=lambda x: x[2])
    current_merges = set()
    for group_key, selected_cluster, selected_time_difference in merge_candidates:
        to_merge_group = merge_groups.get_group(group_key)
        selected_key = (selected_cluster['time_cluster'].iloc[0], selected_cluster['cluster_context'].iloc[0],
                        selected_cluster['group_sub_index'].iloc[0])
        if group_key in current_merges or selected_key in current_merges:
            continue

        merged_group = pd.concat([to_merge_group, selected_cluster])

        # new_sub_index = photos_df['group_sub_index'].max() + 1
        for row_index in merged_group.index:
            photos_df.loc[row_index, 'cluster_context'] = selected_cluster['cluster_context'].iloc[0]
            photos_df.loc[row_index, 'groups_merged'] = to_merge_group['groups_merged'].iloc[0] + selected_cluster['groups_merged'].iloc[0]
            photos_df.loc[row_index, 'group_size'] = len(merged_group)
            photos_df.loc[row_index, 'group_sub_index'] = selected_cluster['group_sub_index'].iloc[0]
            if photos_df.loc[row_index, 'groups_merged'] >= CONFIGS['merge_limit_times']:
                photos_df.loc[row_index, 'merge_allowed'] = False
        current_merges.add(group_key)
        current_merges.add(selected_key)
    return photos_df, True


def process_wedding_illegal_groups(photos_df, look_up_table, logger=None, max_iterations=500):
    required_columns = {'time_cluster', 'cluster_context', 'cluster_label'}

    # Check if required columns exist
    if not required_columns.issubset(photos_df.columns):
        missing = required_columns - set(photos_df.columns)
        logger.error(f"Missing required columns: {missing}")
        raise ValueError(f"Missing required columns: {missing}")

    photos_df['group_sub_index'] = -1
    photos_df['group_size'] = -1
    mask_special = photos_df['cluster_context'].isin(['None', 'other'])
    df_special = photos_df[mask_special].copy()
    df_regular = photos_df[~mask_special].copy()
    groups_special = df_special.groupby(['time_cluster', 'cluster_context', 'cluster_label'])
    for idx, (key, group_df) in enumerate(groups_special):
        group_size = len(group_df)
        df_special.loc[group_df.index, 'group_sub_index'] = idx
        df_special.loc[group_df.index, 'group_size'] = group_size
    groups_regular = df_regular.groupby(['time_cluster', 'cluster_context'])
    for idx, (key, group_df) in enumerate(groups_regular):
        group_size = len(group_df)
        df_regular.loc[group_df.index, 'group_size'] = group_size

    photos_df = pd.concat([df_special, df_regular], ignore_index=True)

    iteration = 0
    try:
        photos_df = handle_wedding_splitting(photos_df,look_up_table, logger)

        photos_df['merge_allowed'] = True
        photos_df['original_context'] = photos_df['cluster_context'].copy()
        photos_df['groups_merged'] = 1
        photos_df = handle_wedding_bride_groom_merge(photos_df, logger)

        while True:
            # Build groups_to_change directly here
            if iteration >= max_iterations:
                logger.warning(f"Maximum iterations ({max_iterations}) reached in process_illegal_groups. Exiting to avoid infinite loop.")
                break

            photos_df, was_merge = process_wedding_merging(photos_df, logger)
            if not was_merge:
                break

            iteration += 1
    except Exception as ex:
        logger.error(f"Unexpected error in process_illegal_groups: {str(ex)}")
        return None, None, None

    groups = photos_df.groupby(['time_cluster', 'cluster_context', 'group_sub_index'])
    group2images = get_images_per_groups(groups)
    logger.info(f"Final number of groups for the album: {len(groups)}")
    logger.info(f"Final groups after illegal handling: {group2images}")
    return groups, group2images, look_up_table
