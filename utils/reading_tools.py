import copy
from math import isnan

import numpy as np
import pandas as pd

from utils.configs import label_list


def generate_dict_key(numbers, n_bodies):
    if (numbers == 0 and n_bodies == 0) or (not numbers):
        return 'No PEOPLE'
    if isinstance(numbers, float):
        if isnan(numbers):
            return 'No PEOPLE'

    # Convert the string of numbers into a list
    try:
        id_list = eval(numbers) if isinstance(numbers, str) else numbers
    except:
        return "Invalid_numbers"

    # Calculate the count based on the list length or n_bodies
    count = max(len(id_list), n_bodies) if isinstance(id_list, list) else n_bodies

    # Determine the suffix
    suffix = "person" if count == 1 else "pple"

    # Combine count, suffix, and the numbers joined by underscores
    key = f"{count}_{suffix}_" + "_".join(map(str, id_list))
    return key


def check_gallery_type(df):
    count = 0
    for idx, row in df.iterrows():  # Unpack the tuple into idx (index) and row (data)
        content_class = row['image_class']
        if pd.isna(content_class):
            continue
        if content_class == -1:
            count += 1

    number_images = len(df)

    if number_images > 0 and count / number_images > 0.6:  # Ensure no division by zero
        return False
    else:
        return True


def map_cluster_label(cluster_label):
    if type(cluster_label) is not int or cluster_label >= len(label_list):
        return "None"
    if cluster_label == -1:
        return "None"
    elif cluster_label >= 0 and cluster_label < len(label_list):
        context = label_list[cluster_label]
        if context in ['two brides', 'two grooms']:
            return 'bride and groom'
        return label_list[cluster_label]
    else:
        return "Unknown"


def process_content(row_dict):
    row_dict = copy.deepcopy(row_dict)
    cluster_class = row_dict.get('cluster_class')
    cluster_class_label = map_cluster_label(cluster_class)
    row_dict['cluster_context'] = cluster_class_label
    return row_dict


def _flatten(iterables):
    for x in iterables:
        if isinstance(x, (list, tuple, set)):
            for y in x:
                yield y
        elif pd.notna(x):
            yield x


def pick_from_set(candidates, allowed_set):
    if not isinstance(candidates, (list, tuple, set)):
        return np.nan
    for c in candidates:
        if c in allowed_set:
            return c
    return np.nan
