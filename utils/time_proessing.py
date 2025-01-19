import pandas as pd
from datetime import datetime
from multiprocessing import Pool


def read_timestamp(timestamp_str):
    try:
        # Try parsing with milliseconds
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        # If it fails, try parsing without milliseconds
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

def convert_to_timestamp(time_integer):
   return datetime.fromtimestamp(time_integer)

def process_image_time_row(row):
    """Processes a single row to compute the general time."""
    cur_timestamp = convert_to_timestamp(row['image_time'])

    if 0 <= cur_timestamp.hour <= 4:
        general_time = int((cur_timestamp.hour + 24) * 60 + cur_timestamp.minute)
    else:
        general_time = int(cur_timestamp.hour * 60 + cur_timestamp.minute)

    # Assuming `first_image_time` is passed or globally available
    global first_image_time
    diff_from_first = cur_timestamp - first_image_time
    general_time += diff_from_first.days * 1440

    row['general_time'] = int(general_time)
    return row

def process_image_time(data_df):
    # Convert image_time to timestamp
    data_df['image_time_date'] = data_df['image_time'].apply(lambda x: convert_to_timestamp(x))

    # Sort by timestamp to find the first image time
    data_df = data_df.sort_values(by='image_time_date')
    global first_image_time
    first_image_time = data_df.iloc[0]['image_time_date']

    # Using Pool to process each row in parallel
    with Pool(processes=4) as pool:
        processed_rows = pool.map(process_image_time_row, [row for _, row in data_df.iterrows()])

    # Create a new DataFrame from processed rows
    processed_df = pd.DataFrame(processed_rows)

    # Sort the DataFrame by general time
    sorted_by_time_df = processed_df.sort_values(by="general_time", ascending=True)

    # Create a dictionary for image_id to general time
    image_id2general_time = dict(zip(processed_df['image_id'], processed_df['general_time']))

    return sorted_by_time_df, image_id2general_time