from __future__ import annotations

import random
from itertools import combinations, product
from dataclasses import dataclass
from typing import List, Tuple, Set, Iterable, Callable, Any, Optional

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
    score: float = None
    left_page_photos: Set | List = None   # set of Photos after resolve_photos,
    right_page_photos: Set | List = None  # list after set_photos_order

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

        # The DataFrame with subset of photos for the given IDs
        df = pd.DataFrame(photos).loc[list(photo_set)]

        # Column-based checks
        is_same_color = df['color'].nunique() == 1
        is_same_class = df['photo_class'].nunique() == 1
        number_of_unique_contexts = df['original_context'].nunique()

        def calculate_bride_groom_mix():
            bride_centric = df['photo_class'].isin(bride_centric_classes).any()
            groom_centric = df['photo_class'].isin(groom_centric_classes).any()
            return bride_centric and groom_centric

        return cls.PageProperties(
            is_same_color=is_same_color,
            is_same_class=is_same_class,
            is_bride_groom_mix=calculate_bride_groom_mix() if is_same_class else False,
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

    def evaluate(self, photos: List[Photo], layouts_df: pd.DataFrame, penalty: Penalties) -> float:
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