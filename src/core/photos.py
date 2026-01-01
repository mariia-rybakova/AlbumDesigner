import cv2
import os
import numpy as np
from glob import glob


from dataclasses import dataclass
from typing import Any, Optional

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
