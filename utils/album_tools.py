import random
import pandas as pd
import ast
import statistics
from datetime import datetime

from src.smart_cropping import process_cropping
from .result_format_template import result_template


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


def get_wedding_groups(df,logger):
    # Ensure df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        return "Error: Input must be a Pandas DataFrame."

    # Required columns
    required_columns = {'time_cluster', 'cluster_context'}

    # Check if required columns exist
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        logger.error(f"Missing required columns: {missing}")
        return f"Error: Missing required columns: {missing}"

    # Handle empty DataFrame case
    if df.empty:
        logger.error('empty dataframe cant get the groups!')
        return "Error: DataFrame is empty."

    try:
        return df.groupby(['time_cluster', 'cluster_context'])
    except Exception as e:
        logger.error(f"Unexpected error during grouping: {e}")
        return f"Error: Unexpected error during grouping: {str(e)}"


def get_none_wedding_groups(df, logger=None):
    # Ensure df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        logger.error("Input must be a Pandas DataFrame.")
        return "Error: Input must be a Pandas DataFrame."

    # Check if required column exists
    required_column = 'people_cluster'

    if required_column not in df.columns:
        logger.error(f"Missing required column: {required_column}")
        return f"Error: Missing required column: {required_column}"

    # Handle empty DataFrame case
    if df.empty:
        logger.error('empty dataframe cant get the groups!')
        return "Error: DataFrame is empty."

    try:
        return df.groupby(['people_cluster'])
    except Exception as e:
        logger.error(f"Unexpected error during grouping: {e}")
        return f"Error: Unexpected error during grouping: {str(e)}"



def calculate_median_time(spread):
    all_times = []
    for page in spread[1:3]:  # Left and right pages
        for photo in page:
            if hasattr(photo, 'general_time') and photo.general_time is not None:
                all_times.append(photo.general_time)

    if all_times:
        return statistics.median(all_times)
    else:
        return float('inf')  # Handle spreads with no valid times


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

def organize_groups(data_list,layouts_df,groups_df, is_wedding,logger):  # Add smart_cropping function as argument
    organized_groups = {}
    for group_data in data_list:
        group_name = list(group_data.keys())[0]
        spreads_and_other_info = group_data[group_name]
        spreads = spreads_and_other_info[0]

        spreads.sort(key=lambda spread: calculate_median_time(spread))  # Use median time
        organized_groups[group_name] = {}  # Initialize list to store spreads with cropping info

        for spread_index, spread in enumerate(spreads): #added spread index
            if spread_index not in organized_groups[group_name]:
                organized_groups[group_name][spread_index] = {}

            layout_index = spread[0]
            layout_id = layouts_df.loc[layout_index]['id']
            cur_layout_info = ast.literal_eval(layouts_df.loc[layout_index]['boxes_info'])
            left_box_ids = ast.literal_eval(layouts_df.loc[layout_index]['left_box_ids'])
            right_box_ids = ast.literal_eval(layouts_df.loc[layout_index]['right_box_ids'])

            left_page_photos = list(spread[1])
            right_page_photos = list(spread[2])

            all_box_ids = left_box_ids + right_box_ids
            all_photos = left_page_photos + right_page_photos

            orig_group_name = group_name.split('*')
            if is_wedding:
                parts = orig_group_name[0].split('_')
                group_id = (int(parts[0]), '_'.join(parts[1:]))
            else:
                group_id = orig_group_name[0]

            c_group = groups_df.get_group(group_id)

            spread_with_cropping_info = []  # List to hold spread data with cropping info
            for i, box in enumerate(cur_layout_info):
                box_id = box['id']
                if box_id not in all_box_ids:
                    logger.error('Some error, cant find box with id: {}'.format(box_id))
                    continue  # Skip to the next box if there's an error

                element_index = all_box_ids.index(box_id)
                cur_photo = all_photos[element_index]
                c_image_id = cur_photo.id

                c_image_info = c_group[c_group['image_id'] == c_image_id]

                x, y, w, h = box['x'], box['y'], box['width'], box['height']
                box_aspect_ratio = w / h #calculate box aspect ratio

                try:
                    cropped_x, cropped_y, cropped_w, cropped_h = process_cropping(
                        float(c_image_info['image_as'].iloc[0]),
                        c_image_info['faces_info'],
                        c_image_info['background_centroid'].values[0],
                        float(c_image_info['diameter'].iloc[0]),
                        box_aspect_ratio
                    )

                    spread_with_cropping_info.append({
                        "box_number": i, #add box number
                        "image_id": c_image_id,
                        "image_x": cropped_x,
                        "image_y": cropped_y,
                        "image_w": cropped_w,
                        "image_h": cropped_h,
                    })

                except Exception as e:  # Catch and handle exceptions during cropping
                    print(f"Error during cropping for image {c_image_id}: {e}")

            organized_groups[group_name][spread_index]['id'] = layout_id
            organized_groups[group_name][spread_index]['images'] =spread_with_cropping_info  #append the list of dicts

    return organized_groups



def assembly_output(output_list,message,layouts_df,images_df,cover_images_ids, covers_images_df, covers_layouts_df):
    output = result_template
    # adding the Album Cover
    if message.cover:
        output['compositions'].append({"compositionId": 1,
                                       "designId":  message.cover_designs[0] ,
                                       "styleId": message.defaultPackageStyleId,
                                       "revisionCounter": 0,
                                       "copies": 1,
                                       "boxes": None,
                                       "logicalSelectionsState": None})

    # adding the first spread image
    if message.first:
        left_box_ids = layouts_df.loc[covers_layouts_df[0]]['left_box_ids']
        right_box_ids = layouts_df.loc[covers_layouts_df[0]]['right_box_ids']
        all_box_ids = left_box_ids + right_box_ids

        first_design_choose = message.first_designs # check which one has one image then choose randomly and its box id
        output['compositions'].append({"compositionId": 2,
                                       "designId": message.cover_designs[0],
                                       "styleId": message.defaultPackageStyleId,
                                       "revisionCounter": 0,
                                       "copies": 1,
                                       "boxes": None,
                                       "logicalSelectionsState": None})

        output['placementsImg'].append({"placementImgId": 1,
                                        "compositionId": 2,
                                        "boxId": all_box_ids[0],
                                        "photoId": cover_images_ids[0],
                                        "cropX": covers_images_df.iloc[0]['cropped_x'],
                                        "cropY": covers_images_df.iloc[0]['cropped_y'],
                                        "cropWidth": covers_images_df.iloc[0]['cropped_w'],
                                        "cropHeight": covers_images_df.iloc[0]['cropped_h'],
                                        "rotate": 0,
                                        "projectId": message.content['projectId'],
                                        "photoFilter": 0,
                                        "photo": None})

    # layouts_df.loc[covers_layouts_df[0]]['id']


    # Add images
    counter_comp_id = 1
    counter_image_id = 1
    for number_groups,group_dict in enumerate(output_list):

        for group_name in group_dict.keys():
            group_result = group_dict[group_name]
            total_spreads = len(group_result)
            for i in range(total_spreads):
                counter_comp_id += 1
                group_data = group_result[i]
                if isinstance(group_data, float):
                    continue
                if isinstance(group_data, list):
                    number_of_spreads = len(group_data)

                    for spread_index in range(number_of_spreads):
                        layout_id = group_data[spread_index][0]

                        output['compositions'].append({"compositionId": counter_comp_id,
                                                       "designId": layouts_df.loc[layout_id]['id'],
                                                       "styleId": 0,#message.content['style_id'],
                                                       "revisionCounter": 0,
                                                       "copies": 1,
                                                       "boxes": None,
                                                       "logicalSelectionsState": None})

                        cur_layout_info = layouts_df.loc[layout_id]['boxes_info']
                        left_box_ids = layouts_df.loc[layout_id]['left_box_ids']
                        right_box_ids = layouts_df.loc[layout_id]['right_box_ids']

                        left_page_photos = list(group_data[spread_index][1])
                        right_page_photos = list(group_data[spread_index][2])

                        all_box_ids = left_box_ids + right_box_ids
                        all_photos = left_page_photos + right_page_photos

                        # Loop over boxes and plot images
                        for j, box in enumerate(cur_layout_info):
                            counter_image_id = counter_image_id + 1
                            box_id = box['id']
                            if box_id not in all_box_ids:
                                print('Some error, cant find box with id: {}'.format(box_id))

                            element_index = all_box_ids.index(box_id)
                            cur_photo = all_photos[element_index]
                            image_id = cur_photo.id

                            image_info = images_df[images_df["image_id"] == image_id]
                            x = image_info['cropped_x']
                            y = image_info['cropped_y']
                            w = image_info['cropped_w']
                            h = image_info['cropped_h']

                            output['placementsImg'].append({"placementImgId" : counter_image_id,
                                                            "compositionId" : counter_comp_id,
                                                            "boxId" : box_id,
                                                            "photoId" : image_id,
                                                            "cropX" : x,
                                                            "cropY" : y,
                                                            "cropWidth" : w,
                                                            "cropHeight" : h,
                                                            "rotate" : 0,
                                                            "projectId" : message.content['projectId'],
                                                            "photoFilter" : 0,
                                                            "photo" : None})


    # adding the last page
    if message.last:
        left_box_ids = layouts_df.loc[covers_layouts_df[1]]['left_box_ids']
        right_box_ids = layouts_df.loc[covers_layouts_df[1]]['right_box_ids']
        all_box_ids = left_box_ids + right_box_ids

        last_design_choose = message.last_designs  # check which one has one image then choose randomly and its box id
        output['compositions'].append({"compositionId": output['compositions'][-1]['compositionId'] + 1,
                                       "designId": last_design_choose,
                                       "styleId": message.defaultPackageStyleId,
                                       "revisionCounter": 0,
                                       "copies": 1,
                                       "boxes": None,
                                       "logicalSelectionsState": None})

        output['placementsImg'].append({"placementImgId":  len(output['placementsImg']) +1,
                                        "compositionId": output['compositions'][-1]['compositionId'],
                                        "boxId": all_box_ids[0],
                                        "photoId": cover_images_ids[0],
                                        "cropX": covers_images_df.iloc[0]['cropped_x'],
                                        "cropY": covers_images_df.iloc[0]['cropped_y'],
                                        "cropWidth": covers_images_df.iloc[0]['cropped_w'],
                                        "cropHeight": covers_images_df.iloc[0]['cropped_h'],
                                        "rotate": 0,
                                        "projectId": message.content['projectId'],
                                        "photoFilter": 0,
                                        "photo": None})


    return output






