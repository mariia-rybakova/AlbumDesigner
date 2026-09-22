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


#: Share of *classified* photos that must be unclassified content (-1) before a
#: gallery counts as non-wedding.
UNCLASSIFIED_SHARE_FOR_NON_WEDDING = 0.6


def check_gallery_type(df):
    """Wedding or not, decided over the photos the content model has answered for.

    The share is taken over photos with a known `image_class`, not over the
    whole frame. A NaN there means the content model has not finished that
    photo, and counting it in the denominator while skipping it in the
    numerator pushes the ratio toward zero -- and `> 0.6` is the only route to
    non-wedding, so a gallery that arrived incomplete could only ever come out
    a *wedding*, the more certainly the less of it had been read.

    That is not hypothetical: 53753700 reached this with 61 rows of which 8
    carried content data, was called a wedding on 8/61 = 0.13, and went down
    the wedding path with 8 photos. Re-read once the gallery was complete, the
    same 85 photos answer non-wedding, which is what its projectCategory says.
    """
    if 'image_class' not in df.columns:
        return True

    known = df['image_class'].notna()
    number_images = int(known.sum())
    count = int((df.loc[known, 'image_class'] == -1).sum())

    if number_images > 0 and count / number_images > UNCLASSIFIED_SHARE_FOR_NON_WEDDING:
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
