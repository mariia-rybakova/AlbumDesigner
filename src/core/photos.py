import os
from glob import glob
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Set
from itertools import groupby

import cv2
import numpy as np


@dataclass(frozen=True)
class Photo:
    # class definition to hold all photo information required to calculate the layout score
    id: Any
    ar: float
    color: bool
    rank: float
    photo_class: Optional[str]
    cluster_label: int
    general_time: float
    original_context: Optional[str] = None

    @classmethod
    def from_array(cls, array):
        return cls(array[0], array[1], array[2], array[3], array[4], array[5], array[6])


def get_int_photo_id(photo_id):
    if isinstance(photo_id, int):
        return photo_id

    photo_id = photo_id.split('_')[0]
    photo_id = photo_id.split('.')[0]
    return int(photo_id)


def get_photos_from_db(data_db, is_wedding):
    photos = list()
    for index, row in data_db.iterrows():
        image_id = row['image_id']
        class_contex = row['cluster_context'] if is_wedding else None
        cluster_label = row['cluster_label']
        color = False if row['image_color'] == 0 else True
        aspect_ratio = row['image_as']
        rank_score = row['image_order']
        original_context = row['original_context'] if 'original_context' in row else None

        photos.append(Photo(id=image_id, ar=aspect_ratio, color=color, rank=rank_score,
                            photo_class=class_contex, cluster_label=cluster_label,
                            general_time=row['general_time'], original_context=original_context))


    # photos = sorted(photos, key=lambda photo: photo.id)

    return photos


def update_photos_ranks(data_db, chosen_photos):
    if data_db is None or chosen_photos is None or len(chosen_photos) == 0:
        return data_db
    for photo_id in chosen_photos:
        data_db.loc[data_db['image_id'] == photo_id, 'image_order'] = 0
    return data_db


# Photo time/context grouping utilities

PhotoTimeEntry = Tuple[int, float, Tuple[Optional[str], bool]]
"""A tuple of (photo_index, general_time, (original_context, color))."""


def get_time_sequences(spread_photos: List[int], photos: List[Photo]) -> List[PhotoTimeEntry]:
    """
    Build a list of (photo_id, time, (context, color)) tuples for the given spread.

    Args:
        spread_photos: List of photo indices within this spread.
        photos: Full list of Photo objects.

    Returns:
        List of PhotoTimeEntry tuples, each containing the photo index,
        its general_time, and a (original_context, color) grouping key.
    """
    return [
        (
            photo_id,
            photos[photo_id].general_time,
            (photos[photo_id].original_context, photos[photo_id].color)
        )
        for photo_id in spread_photos
    ]


def group_photos(spread_photos: List[int], photos: List[Photo]) -> List[List[PhotoTimeEntry]]:
    """
    Group spread photos by (context, color), sorted by time.

    Sorts photos by general_time, then groups consecutive photos sharing the
    same (original_context, color) pair using itertools.groupby.

    Args:
        spread_photos: List of photo indices within this spread.
        photos: Full list of Photo objects.

    Returns:
        A list of groups, where each group is a list of PhotoTimeEntry tuples.
    """
    time_sequences = get_time_sequences(spread_photos, photos)
    # sort by time
    time_sequences = sorted(time_sequences, key=lambda x: x[1])
    # group by (context, color)
    grouped = groupby(time_sequences, key=lambda x: x[2])
    return [list(group) for _, group in grouped]


def get_portraits_landscapes(subset_photo_idxs: List[int] | Set[int], all_photos: List[Photo]) -> Tuple[Set[int], Set[int]]:
    """
    Separate photo indices into portrait and landscape sets by aspect ratio.

    Args:
        subset_photo_idxs: Indices into all_photos to classify.
        all_photos: Full list of Photo objects.

    Returns:
        Tuple of (portrait_idxs, landscape_idxs) as sets. Photos with
        ar < 1 are portraits, the rest are landscapes.
    """
    photo_idxs = list(subset_photo_idxs)
    landscape_idxs = set()
    portrait_idxs = set()

    for i in range(len(photo_idxs)):
        if all_photos[photo_idxs[i]].ar < 1:
            portrait_idxs.add(photo_idxs[i])
        else:
            landscape_idxs.add(photo_idxs[i])

    return portrait_idxs, landscape_idxs


def count_photo_times_per_class(photos: List[Photo]) -> dict[Optional[str], List[float]]:
    """
    Group photo timestamps by their original_context class.

    Args:
        photos: List of Photo objects.

    Returns:
        Dict mapping each original_context value to a list of general_time
        values for photos in that context.
    """
    times_for_classes = {}

    for photo in photos:
        class_name = photo.original_context

        if class_name not in times_for_classes:
            times_for_classes[class_name] = []

        times_for_classes[class_name].append(photo.general_time)

    return times_for_classes