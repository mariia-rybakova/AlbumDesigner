from __future__ import annotations

import random
from itertools import combinations, product
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Iterable, Callable, Any, Optional

import numpy as np
import pandas as pd

from src.spreads_layout.layouts_tools import (filter_layouts, count_squares, is_large_spread_with_squares,
                                              update_with_page_capacities, apply_layouts_mask)
from src.spreads_layout.combinations import Combination
from src.spreads_layout.math_tools import limit_sample_size
from src.core.models import SpreadSearchParams
from src.core.photos import Photo, group_photos, get_portraits_landscapes


@dataclass
class Penalties:
    """
    Multiplicative penalty factors applied during spread layout evaluation.

    Each factor is multiplied into the score when its condition is detected.
    Values closer to 0 impose harsher penalties.

    Attributes:
        crop_penalty: Per-square-box penalty for cropping photos to fit.
        color_mix: Penalty when photos on a page have mixed color modes.
        class_mix: Penalty when photos on a page have different photo_class values.
        orientation_mix: Penalty when a page mixes portrait and landscape orientations.
        score_threshold: Minimum ratio of a spread's score to the max score in its
            group; spreads below this ratio are filtered out.
        double_mix_color: Penalty when both left and right pages have mixed colors.
        context_mix_penalty: Per-extra-context penalty (exponential) for multiple
            original_context values on a page.
        time_order_penalty: Per-inversion penalty for photos not in time order.
    """
    crop_penalty: float = 0.5
    color_mix: float = 0.000000001
    class_mix: float = 0.01
    orientation_mix: float = 0.1
    score_threshold: float = 0.01
    double_mix_color: float = 0.000000000000000001
    context_mix_penalty: float = 0.00001
    time_order_penalty: float = 0.005


@dataclass
class SingleSpreadLayout:
    layout_idx: int
    left_page_photo_idxs: Set[int]      # photo index in local list of photos per group
    right_page_photo_idxs: Set[int]     # (not photo ID)
    number_of_squares: int
    score: Optional[float] = None
    weight: Optional[float] = None
    left_page_photos: Optional[Set[Photo] | List[Photo]] = None   # set of Photos after resolve_photos,
    right_page_photos: Optional[Set[Photo] | List[Photo]] = None  # list after set_photos_order

    def __str__(self) -> str:
        return (f'Layout_idx: {self.layout_idx}; '
                f'Photos: [left page: {self.left_page_photo_idxs}, right page: {self.right_page_photo_idxs}]; '
                f'Square boxes in spread: {self.number_of_squares}')

    def resolve_photos(self, photos: List[Photo]) -> None:
        """Map photo indices to Photo objects for both pages."""
        self.left_page_photos = {photos[idx] for idx in self.left_page_photo_idxs}
        self.right_page_photos = {photos[idx] for idx in self.right_page_photo_idxs}

    def set_photos_order(self, left_ordered: List[Photo], right_ordered: List[Photo]) -> None:
        """Set the final ordered photo lists for both pages."""
        self.left_page_photos = left_ordered
        self.right_page_photos = right_ordered

    @dataclass
    class PageProperties:
        """
        Result of checking photo consistency within a single page.

        Attributes:
            is_same_color: Whether all photos on the page share the same color mode.
            is_same_class: Whether all photos share the same photo_class.
            is_bride_groom_mix: Whether bride-centric and groom-centric classes are
                mixed on the same page (only checked when is_same_class is True).
            number_of_unique_contexts: Count of distinct original_context values.
        """
        is_same_color: bool
        is_same_class: bool
        is_bride_groom_mix: bool
        number_of_unique_contexts: int

    @classmethod
    def check_page_properties(cls, photo_set: Set[int], photos: List[Photo]) -> PageProperties:
        """
        Analyze color, class, and context consistency for photos on a single page.

        Builds a DataFrame from the photo subset and checks whether all photos share
        the same color mode, same photo_class, and whether bride/groom classes are mixed.

        Args:
            photo_set: Set of photo indices (into the group's local photo list).
            photos: Full list of Photo objects for the group.

        Returns:
            PageProperties with consistency flags for the page.
        """
        bride_centric_classes = ['bride', 'bride party', 'wedding dress', 'getting hair-makeup','bride getting dressed']
        groom_centric_classes = ['groom','groom party','suit']

        if len(photo_set) == 1:
            return cls.PageProperties(True, True, False, 1)

        # Collect attributes
        colors = [photos[pid].color for pid in photo_set]
        photo_classes = [photos[pid].photo_class for pid in photo_set]
        contexts = [photos[pid].original_context for pid in photo_set]

        # Uniqueness checks
        is_same_color = len(set(colors)) == 1
        is_same_class = len(set(photo_classes)) == 1
        number_of_unique_contexts = len(set(contexts))

        def calculate_bride_groom_mix():
            bride_centric = any(cls_name in bride_centric_classes for cls_name in photo_classes)
            groom_centric = any(cls_name in groom_centric_classes for cls_name in photo_classes)
            return bride_centric and groom_centric

        return cls.PageProperties(
            is_same_color=is_same_color,
            is_same_class=is_same_class,
            is_bride_groom_mix=calculate_bride_groom_mix() if not is_same_class else False,
            number_of_unique_contexts=number_of_unique_contexts
        )

    def apply_page_penalties(self, page_check_result: PageProperties, penalty: Penalties) -> None:
        """
        Apply multiplicative penalties to self.score based on page consistency.

        Penalizes mixed colors, mixed classes, bride/groom mixing, and multiple
        original_context values on a single page.

        Args:
            page_check_result: PageProperties for the page being evaluated.
            penalty: Penalty configuration.
        """
        if not page_check_result.is_same_color:
            self.score *= penalty.color_mix
        if not page_check_result.is_same_class:
            self.score *= penalty.class_mix
        if page_check_result.is_bride_groom_mix:
            self.score *= penalty.color_mix

        self.score *= np.power(penalty.context_mix_penalty, max(1, page_check_result.number_of_unique_contexts) - 1)

    def get_score(self, photos: List[Photo], layouts_df: pd.DataFrame, penalty: Penalties) -> float:
        """
        Score this spread layout based on page consistency, orientation, cropping,
        and time ordering penalties. Sets and returns self.score.

        Args:
            photos: List of Photo objects for the group.
            layouts_df: DataFrame of layouts (used for orientation mixing flags).
            penalty: Penalty configuration.

        Returns:
            The computed score for this spread layout.
        """
        self.score = 1.0

        left_check = self.check_page_properties(self.left_page_photo_idxs, photos)
        self.apply_page_penalties(left_check, penalty)
        if layouts_df.at[self.layout_idx, 'left_mixed']:
            self.score *= penalty.orientation_mix

        right_check = self.check_page_properties(self.right_page_photo_idxs, photos)
        self.apply_page_penalties(right_check, penalty)
        if layouts_df.at[self.layout_idx, 'right_mixed']:
            self.score *= penalty.orientation_mix

        # if two pages has gray colors, give it much worse rating
        if not left_check.is_same_color and not right_check.is_same_color:
            self.score *= penalty.double_mix_color

        # penalty for cropping photos to square boxes
        self.score *= np.power(penalty.crop_penalty, self.number_of_squares)

        # if time order is not correct, give it a penalty
        photo_order_time = [photos[photo_id].general_time for photo_id in
                            list(self.left_page_photo_idxs) + list(self.right_page_photo_idxs)]
        for time_idx1 in range(len(photo_order_time)):
            for time_idx2 in range(time_idx1 + 1, len(photo_order_time)):
                if photo_order_time[time_idx1] > photo_order_time[time_idx2]:
                    self.score *= penalty.time_order_penalty

        return self.score

    def set_weight(self, weight: float) -> None:
        self.weight = weight

    @staticmethod
    def _assign_photos_order_by_area(photos: List[Photo], boxes: List[Dict[str, Any]],
                                    portraits_total: int, landscapes_total: int
                                    ) -> Tuple[List[Photo], int, int]:
        """
        Assign photos to boxes of a single area tier, matching orientation when possible.

        Iterates through boxes, assigning portrait photos to portrait boxes and landscape
        photos to landscape boxes first, then fills square boxes with remaining photos.
        Tracks how many portrait/landscape photos remain for subsequent area tiers.

        Args:
            photos: Photos available for assignment (portrait and landscape mixed),
                sorted by rank and time.
            boxes: List of box dicts with 'orientation', 'id', 'area', 'side', 'position'.
            portraits_total: Number of portrait photos reserved for later area tiers.
            landscapes_total: Number of landscape photos reserved for later area tiers.

        Returns:
            Tuple of (photos_order, remaining_portraits, remaining_landscapes) where
            photos_order is a list aligned with boxes containing assigned Photo objects.
        """
        photos_portrait = [photo for photo in photos if photo.ar < 1]
        photos_landscape = [photo for photo in photos if photo.ar >= 1]
        port_idx = 0
        land_idx = 0

        photos_order = [None] * len(boxes)
        for idx, box_data in enumerate(boxes):
            if box_data['orientation'] == 'portrait':
                if port_idx != len(photos_portrait):
                    cur_photos = photos_portrait[port_idx]
                    port_idx += 1
                    photos_order[idx] = cur_photos
                elif land_idx != len(photos_landscape):
                    cur_photos = photos_landscape[land_idx]
                    land_idx += 1
                    photos_order[idx] = cur_photos
                else:
                    print("Error: no more photos to add")
            if box_data['orientation'] == 'landscape':
                if land_idx != len(photos_landscape):
                    cur_photos = photos_landscape[land_idx]
                    land_idx += 1
                    photos_order[idx] = cur_photos
                elif port_idx != len(photos_portrait):
                    cur_photos = photos_portrait[port_idx]
                    port_idx += 1
                    photos_order[idx] = cur_photos
                else:
                    print("Error: no more photos to add")

        portraits_total = portraits_total - port_idx
        landscapes_total = landscapes_total - land_idx

        for idx, box_data in enumerate(boxes):
            if box_data['orientation'] == 'square':
                if port_idx != len(photos_portrait) - portraits_total:
                    cur_photos = photos_portrait[port_idx]
                    port_idx += 1
                    photos_order[idx] = cur_photos
                elif land_idx != len(photos_landscape) - landscapes_total:
                    cur_photos = photos_landscape[land_idx]
                    land_idx += 1
                    photos_order[idx] = cur_photos
                else:
                    print("Error: no more photos to add")

        return photos_order, portraits_total, landscapes_total

    @classmethod
    def _assign_part_photos_order(cls, boxes: List[Dict[str, Any]],
                                 photos: List[Photo]) -> Tuple[List[Photo], List[Photo]]:
        """
        Order photos into left and right page positions by matching box areas to ranks.

        Groups boxes by area (with 0.01 tolerance), then processes area tiers from
        largest to smallest. Within each tier, assigns photos by orientation via
        _assign_photos_order_by_area, sorts assigned photos by time, and places them
        into left/right page positions. Assigned photos are removed from the pool
        before processing the next tier.

        Args:
            boxes: List of box dicts, each with 'id', 'area', 'orientation', 'side',
                and 'position' keys.
            photos: Photos to assign, sorted by rank and time.

        Returns:
            Tuple of (left_photos_order, right_photos_order), each a list of Photo
            objects indexed by box position within that page.
        """
        portraits_total = len([box for box in boxes if box['orientation'] == 'portrait'])
        landscapes_total = len([box for box in boxes if box['orientation'] == 'landscape'])

        area2boxes = dict()
        for box in boxes:
            cur_area = box['area']
            added = False
            for saved_area in area2boxes.keys():
                if abs(cur_area - saved_area) < 0.01:
                    area2boxes[saved_area].append(box)
                    added = True
                    break
            if added:
                continue
            if cur_area not in area2boxes:
                area2boxes[cur_area] = list()
            area2boxes[cur_area].append(box)

        left_size = max([box['position'] for box in boxes if box['side'] == 0], default=-1) + 1
        right_size = max([box['position'] for box in boxes if box['side'] == 1], default=-1) + 1
        left_photos_order = [None] * left_size
        right_photos_order = [None] * right_size

        for area in sorted(area2boxes.keys(), reverse=True):
            cur_boxes = area2boxes[area]
            photos_order, portraits_total, landscapes_total = cls._assign_photos_order_by_area(photos, cur_boxes,
                                                                                          portraits_total,
                                                                                          landscapes_total)
            photos_order = sorted(photos_order, key=lambda x: x.general_time)
            for idx, box_data in enumerate(cur_boxes):
                cur_photo = photos_order[idx]
                if box_data['side'] == 0:
                    left_photos_order[box_data['position']] = cur_photo
                else:
                    right_photos_order[box_data['position']] = cur_photo

            # Remove assigned photos from all_photos so they aren't reused
            assigned_photos = [p for p in photos_order if p is not None]
            if assigned_photos:
                assigned_ids = set(id(p) for p in assigned_photos)
                photos = [p for p in photos if id(p) not in assigned_ids]

        return left_photos_order, right_photos_order

    def assign_photos_order(self, layout_id2data: Dict[int, Any],
                            design_box_id2data: Dict[Tuple[int, int], Any], merge_pages: bool) -> None:
        """
        Determine final photo ordering within this spread's left and right pages.

        Builds box metadata (area, orientation, position) from the layout and design
        data, then assigns photos to boxes by orientation and area ranking. If
        merge_pages is True, all photos are pooled and assigned across both pages
        together; otherwise each page is assigned independently.

        Args:
            layout_id2data: Mapping from layout index to layout metadata including
                'layout_id', 'left_box_ids', and 'right_box_ids'.
            design_box_id2data: Mapping from (layout_id, box_id) to box properties
                including 'area' and 'orientation'.
            merge_pages: If True, pool photos from both pages before assignment;
                if False, assign each page's photos independently.
        """
        layout_data = layout_id2data[self.layout_idx]
        left_boxes_ids = layout_data['left_box_ids']
        right_boxes_ids = layout_data['right_box_ids']
        left_page_boxes = [{'id': bid,
                      'side': 0,
                      'position': idx,
                      'area': design_box_id2data[(layout_data['layout_id'],bid)]['area'],
                      'orientation': design_box_id2data[(layout_data['layout_id'],bid)]['orientation']
                      } for idx, bid in enumerate(left_boxes_ids)]
        right_page_boxes = [{'id': bid,
                       'side': 1,
                       'position': idx,
                       'area': design_box_id2data[(layout_data['layout_id'],bid)]['area'],
                       'orientation': design_box_id2data[(layout_data['layout_id'],bid)]['orientation']
                       } for idx, bid in enumerate(right_boxes_ids)]

        if merge_pages:
            all_photos = sorted(list(self.left_page_photos) + list(self.right_page_photos),
                                key=lambda ph: (ph.rank, ph.general_time))
            left_photos_order, right_photos_order = self._assign_part_photos_order(left_page_boxes + right_page_boxes, all_photos)
        else:
            left_page_photos = sorted(self.left_page_photos, key=lambda ph: (ph.rank, ph.general_time))
            right_page_photos = sorted(self.right_page_photos, key=lambda ph: (ph.rank, ph.general_time))
            left_photos_order, _ = self._assign_part_photos_order(left_page_boxes, left_page_photos)
            _, right_photos_order = self._assign_part_photos_order(right_page_boxes, right_page_photos)

        self.set_photos_order(left_photos_order, right_photos_order)