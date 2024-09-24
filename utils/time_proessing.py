from datetime import datetime

def read_timestamp(timestamp_str):
    try:
        # Try parsing with milliseconds
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        # If it fails, try parsing without milliseconds
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

def convert_to_timestamp(time_integer):
   return datetime.fromtimestamp(time_integer)

def process_image_time(data_df):
        timestamps = data_df['image_time'].apply(lambda x: convert_to_timestamp(x))

        #timestamps = [read_timestamp(timestamp_str) for timestamp_str in timestamps]
        image_ids = data_df['image_id']

        image_ids2timestamps = [(image_id, timestamp) for image_id, timestamp in zip(image_ids, timestamps)]
        image_ids2timestamps = sorted(image_ids2timestamps, key=lambda x: x[1])
        if len(image_ids2timestamps) == 0:
            return None, dict()

        image_id2general_time = dict()
        first_image_time = image_ids2timestamps[0][1]
        for image_id, cur_timestamp in image_ids2timestamps:
            #general_time = cur_timestamp.hour * 60 + cur_timestamp.minute + cur_timestamp.second / 60
            general_time = int(cur_timestamp.hour * 60 + cur_timestamp.minute)
            diff_from_first = cur_timestamp - first_image_time
            general_time += diff_from_first.days * 1440

            image_id2general_time[image_id] = int(general_time)
            data_df.loc[data_df['image_id'] == image_id, 'general_time'] = int(general_time)

        sorted_by_time_df = data_df.sort_values(by="general_time", ascending=False)

        return sorted_by_time_df, image_id2general_time