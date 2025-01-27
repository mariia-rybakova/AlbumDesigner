import random
import statistics
from datetime import datetime


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


def  get_wedding_groups(df):
    return df.groupby(['time_cluster', 'cluster_context'])


def get_none_wedding_groups(df):
    return df.groupby(['cluster_people'])


def get_images_per_groups(original_groups):
    group2images_data_list = dict()
    for name_group, group_df in original_groups:
        num_images = len(group_df)
        group2images_data_list[name_group] = num_images
    return group2images_data_list


def sort_groups_by_photo_time(data_list):
    result = {}
    def get_mean_time(sublist):
        # Extract all general_time values from the photos in the sublist
        total_spread_time = []
        for page in sublist[1:3]:
            time = [photo.general_time for photo in page]
            total_spread_time.extend(time)

        if total_spread_time:
            #logger.info(f"layout id {sublist[0]}, average time, {statistics.mean(total_spread_time)}")
            return statistics.median(total_spread_time)
        else:
            return float('inf')

    # Iterate over each group in the dictionary
    for data_dict in data_list:
        for group_key, group_data in data_dict.items():
            if group_key not in result:
                result[group_key] = []

            # Ensure group_data is a list and contains elements
            if isinstance(group_data, list) and len(group_data) > 0:
                # Sort the group data by get_mean_time and extend the result
                sorted_group = sorted(group_data[0], key=get_mean_time)
                result[group_key].extend(sorted_group)

    return result

def sort_groups(list_of_groups,group2images):
    group2images
    for list_of_group in list_of_groups:
        group_name = list_of_group.keys()[0]




def sort_sub_groups(sub_grouped, group_names):
    # Define the priority list for secondary sorting
    priority_list = ["bride getting dressed", "getting hair-makeup", "bride", "groom getting dress", "groom",
                     "portrait",
                     "bride and groom", "walking the aisle", "ceremony", "settings", "dancing"]

    # Create a dictionary to map group names to their priority
    if priority_list:
         priority_dict = {name: i for i, name in enumerate(priority_list)}

    time_group_dict = {}
    for group_name in group_names:
        orig_group_name = group_name.split('*')
        group_id =orig_group_name[0].split('_')
        group = sub_grouped.get_group(group_id)

        # Calculate the median instead of the mean
        group_time_median = group["general_time"].median()
        time_group_dict[group_name] = group_time_median

    sorted_time_groups = dict(
        sorted(time_group_dict.items(),
               key=lambda item: (item[1],
                                 priority_dict.get(item[0].split('_')[1].split('*')[0], float('inf'))))
    )

    return sorted_time_groups
