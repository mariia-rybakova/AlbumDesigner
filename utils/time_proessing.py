import copy
import pandas as pd

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

def read_timestamp(timestamp_str):
    try:
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

def convert_to_timestamp(time_integer):
    return datetime.fromtimestamp(time_integer)

def process_image_time_row(args):
    """Processes a single row to compute the general time."""
    row_time, first_image_time = args
    row_time = copy.deepcopy(row_time)
    cur_timestamp = convert_to_timestamp(row_time['image_time'])

    if 0 <= cur_timestamp.hour <= 4:
        general_time = int((cur_timestamp.hour + 24) * 60 + cur_timestamp.minute)
    else:
        general_time = int(cur_timestamp.hour * 60 + cur_timestamp.minute)

    diff_from_first = cur_timestamp - first_image_time
    general_time += diff_from_first.days * 1440

    row_time['general_time'] = int(general_time)
    return row_time

def process_image_time(data_df):
    # Convert image_time to timestamp
    data_df['image_time_date'] = data_df['image_time'].apply(lambda x: convert_to_timestamp(x))

    # Sort by timestamp to find the first image time
    data_df = data_df.sort_values(by='image_time_date')
    first_image_time = data_df.iloc[0]['image_time_date']

    time_data_dict = data_df[['image_id', 'image_time']].to_dict('records')

    args_list = [(row, first_image_time) for row in time_data_dict]

    # Using Pool to process each row in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        processed_rows = list(executor.map(process_image_time_row, args_list))

    # Create a new DataFrame from processed rows
    processed_df = pd.DataFrame(processed_rows)

    processed_df = data_df.merge(processed_df[['image_id', 'general_time']], how='left', on='image_id')

    # Sort the DataFrame by general time
    sorted_by_time_df = processed_df.sort_values(by="general_time", ascending=True)

    # Create a dictionary for image_id to general time
    image_id2general_time = dict(zip(processed_df['image_id'], processed_df['general_time']))

    return sorted_by_time_df, image_id2general_time
