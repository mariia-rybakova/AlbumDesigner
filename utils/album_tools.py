import random
from datetime import datetime


def get_important_imgs(data_df, top=5):
    selection_q = ['bride and groom in a great moment together','bride and groom ONLY','bride and groom ONLY with beautiful background ',' intimate moment in a serene setting between bride and groom ONLY','bride and groom Only in the picture  holding hands','bride and groom Only kissing each other in a romantic way',   'bride and groom Only in a gorgeous standing ','bride and groom doing a great photosession together',' bride and groom with a fantastic standing looking to each other with beautiful scene','bride and groom kissing each other in a photoshot','bride and groom holding hands','bride and groom half hugged for a speical photo moment']
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
            (data_df["image_query_content"] == "groom")].head(top)['image_id'].tolist()

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
