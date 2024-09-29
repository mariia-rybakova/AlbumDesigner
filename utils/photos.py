import cv2
import os
import numpy as np
from glob import glob

from utils.album_tools import get_general_times


class Photo:
    # class definition to hold all photo information required to calculate the layout score
    def __init__(self, id, ar, color, rank, photo_class, cluster_label, general_time):
        self.id = id
        self.ar = ar
        self.color = color
        self.rank = rank
        self.photo_class = photo_class
        self.cluster_label = cluster_label
        self.general_time = general_time

    @classmethod
    def from_array(cls, array):
        return cls(array[0], array[1], array[2], array[3], array[4], array[5], array[6])


def get_int_photo_id(photo_id):
    if isinstance(photo_id, int):
        return photo_id

    photo_id = photo_id.split('_')[0]
    photo_id = photo_id.split('.')[0]
    return int(photo_id)


def get_photos_from_db(data_db):
    photos = list()
    for index, row in data_db.iterrows():
        image_id = row['image_id']
        class_contex = row['cluster_context']
        cluster_label = row['cluster_label']
        color = False if row['image_color'] == 0 else True
        aspect_ratio = row['image_as']
        rank_score = row['ranking']

        photos.append(Photo(id=image_id, ar=aspect_ratio, color=color, rank=rank_score,
                            photo_class=class_contex, cluster_label=cluster_label,
                            general_time=row['edited_general_time']))

    photos = sorted(photos, key=lambda photo: photo.id)

    return photos

