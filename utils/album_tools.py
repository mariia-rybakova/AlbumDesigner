import random
from datetime import datetime
from itertools import chain
import pandas as pd


def get_images_per_groups(original_groups):
    group2images_data_list = dict()

    for name_group, group_df in original_groups:
        num_images = len(group_df)
        group2images_data_list[name_group] = num_images

    return group2images_data_list


def get_important_imgs(data_df, top=5):
    selection_q = ['bride and groom in a great moment together','bride and groom ONLY','bride and groom ONLY with beautiful background ',' intimate moment in a serene setting between bride and groom ONLY','bride and groom Only in the picture  holding hands','bride and groom Only kissing each other in a romantic way',   'bride and groom Only in a gorgeous standing ','bride and groom doing a great photosession together',' bride and groom with a fantastic standing looking to each other with beautiful scene','bride and groom kissing each other in a photoshot','bride and groom holding hands','bride and groom half hugged for a speical photo moment','groom and brides dancing together solo', 'bride and groom cutting cake', ]
    # Step 1: Filter based on the conditions
    filtered_df = data_df[
        (data_df["cluster_context"] == "bride and groom") &
        (data_df["image_subquery_content"].isin(selection_q))
        ]

    # Step 2: Take the top N rows based on the 'top' variable
    top_filtered_df = filtered_df.head(top)

    # Step 3: Extract the image_ids into a list
    image_id_list = top_filtered_df["image_id"].tolist()

    if len(image_id_list) == 0:
        # let's pick another images
        image_id_list = data_df[
            (data_df["image_query_content"] == "bride")].head(top)['image_id'].tolist()

    return image_id_list


def get_cover_img(data_df, important_imgs):
    cover_img_id = random.choice(important_imgs)
    cover_image = data_df[data_df['image_id'] == cover_img_id]
    df = data_df[data_df['image_id'] != cover_img_id]
    return df, cover_image


def get_cover_layout(layout_df):
    one_img_layouts = [key for key, layout in layout_df.iterrows() if layout["number of boxes"] == 1]
    chosen_layout = random.choice(one_img_layouts)
    return chosen_layout


def read_timestamp(timestamp_str):
    try:
        # Try parsing with milliseconds
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        # If it fails, try parsing without milliseconds
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

def get_general_times(data_db):
    timestamps = data_db['image_time']
    timestamps = [read_timestamp(timestamp_str) for timestamp_str in timestamps]
    image_ids = data_db['image_id']
    image_ids2timestamps = [(image_id, timestamp) for image_id, timestamp in zip(image_ids, timestamps)]
    image_ids2timestamps = sorted(image_ids2timestamps, key=lambda x: x[1])
    if len(image_ids2timestamps) == 0:
        return dict()

    image_id2general_time = dict()
    first_image_time = image_ids2timestamps[0][1]
    for image_id, cur_timestamp in image_ids2timestamps:
        general_time = cur_timestamp.hour * 60 + cur_timestamp.minute + cur_timestamp.second / 60

        diff_from_first = cur_timestamp - first_image_time
        general_time += diff_from_first.days * 1440

        image_id2general_time[image_id] = general_time
    return image_id2general_time


def get_wedding_groups(df, manual_selection, logger):
    required_columns = {'time_cluster', 'cluster_context', 'cluster_label'}

    # Check if required columns exist
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        logger.error(f"Missing required columns: {missing}")
        return None

    # Split DataFrame based on cluster_context being 'None' or 'other' (as strings)
    if not manual_selection:
        mask_special = df['cluster_context'].isin(['None', 'other'])
        df_special = df[mask_special].copy()
        df_regular = df[~mask_special].copy()

        # Group df_special and update cluster_context for each group
        groups_special = df_special.groupby(['time_cluster', 'cluster_context', 'cluster_label'])
        for idx, (key, group_df) in enumerate(groups_special):
            group_size = len(group_df)
            new_context = f"{key[1]}_{idx}_{group_size}"
            df_special.loc[group_df.index, 'cluster_context'] = new_context

        # Merge modified df_special with df_regular
        merged_df = pd.concat([df_special, df_regular], ignore_index=True)
        # Group the merged DataFrame by ['time_cluster', 'cluster_context']
        groups_final = merged_df.groupby(['time_cluster', 'cluster_context'])
    else:
        groups_final = df.groupby(['time_cluster', 'cluster_context'])
    return groups_final


def get_none_wedding_groups(df, logger):
    # Check if required column exists
    required_column = 'people_cluster'

    if required_column not in df.columns:
        logger.error(f"Missing required column: {required_column}")
        return None

    return df.groupby(['people_cluster'])


def sort_groups_by_name(data_list):
    priority_list = ["bride getting dressed", "getting hair-makeup", "groom getting dress", "bride", "groom",
                     "bride party", "groom party",
                     "kiss", "portrait", "bride and groom", "walking the aisle", "ceremony", "settings", 'speech',
                     "first dance", 'food', "cake cutting", "dancing"]

    priority_dict = {name: i for i, name in enumerate(priority_list)}

    sorted_data_list = sorted(
    [group_data for group_data in data_list if group_data],  # Remove empty dicts
    key=lambda group_data: priority_dict.get(
        list(group_data.keys())[0].split("*")[0].split('_')[1],
        float('inf')
    )) # sort by priority

    return sorted_data_list
