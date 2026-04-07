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
from src.spreads_layout.spreads.list_of_spreads import SpreadLayoutsList, sample_layouts
from src.spreads_layout.spreads.spread import Penalties
from utils.configs import CONFIGS


class GroupLayoutsLists(Combination):
    '''
    The object of this class represents possible layout options for a certain Combination.
    '''
    def __init__(self, spreads: List[Set[int]], weight: float) -> None:
        super().__init__(spreads=spreads, weight=weight)
        self.possible_layouts: List[SpreadLayoutsList] = []

    @classmethod
    def from_comb(cls, comb: Combination) -> GroupLayoutsLists:
        """
        Create a GroupLayoutsLists from an existing Combination.
        """
        return cls(comb.spreads, comb.weight)

    def view(self, limit: Optional[int] = None, sep: str = '==') -> None:
        print(f'Layout options for {len(self.spreads)}-spread group: {self.spreads}')
        for i in range(len(self.spreads)):
            print(sep, i + 1, end = ' ')
            self.possible_layouts[i].view(limit=limit, sep = sep*2)

    def add_spread(self, layouts: SpreadLayoutsList) -> None:
        self.possible_layouts.append(layouts)

    def process(self, layouts_df: pd.DataFrame, photos: List[Photo], penalty: Optional[Penalties] = None) -> None:
        """
        Score and filter all spread layout options for this group.

        For each spread in each combination, computes a multiplicative score based on:
        page color/class consistency, orientation mixing, square-box cropping, and
        photo time ordering. Spreads scoring below score_threshold relative to the
        best spread in their combination are filtered out.

        Args:
            layouts_df: DataFrame of available layouts (used for orientation mixing flags).
            photos: List of Photo objects for the group.
            penalty: Penalty configuration. Uses default Penalties if None.
        """
        if penalty is None:
            penalty = Penalties()
        #print(f"the CONFIGS['spread_score_threshold'] is {penalty.score_threshold}")

        # Process layout lists for all spreads
        for spread in self.possible_layouts:
            # Process each layout for this spread
            spread.process(photos, layouts_df, penalty)


def layout_combination(combination: Combination, layouts_df: pd.DataFrame, photos: List[Photo], params: SpreadSearchParams) -> Optional[GroupLayoutsLists]:
    """
    Find all possible spread layouts for a single Combination.

    For each spread in the combination, delegates to sample_layouts to find
    valid page assignments. Collects results into a GroupLayoutsLists. Returns
    None early if any spread has no valid layout.

    Args:
        combination: The Combination defining which photos go in each spread.
        layouts_df: Full DataFrame of available layout designs.
        photos: List of Photo objects in the group.
        params: Search parameters controlling sampling limits and thresholds.

    Returns:
        GroupLayoutsLists with possible layouts per spread, or None if any
        spread has no valid layout.
    """
    n_spreads = len(combination.spreads)
    group_spreads_layouts = GroupLayoutsLists.from_comb(combination)

    for photo_idx_set in combination.spreads:
        single_spread_layouts = sample_layouts(photo_idx_set, n_spreads, photos, layouts_df, params)

        if single_spread_layouts is None:
            return None

        group_spreads_layouts.add_spread(single_spread_layouts)

    # group_spreads_layouts.view(limit=3)
    return group_spreads_layouts


def process_combination_inner(comb: Combination, photos: List[Photo], layouts_df: pd.DataFrame,
                        params: SpreadSearchParams) -> Optional[GroupLayoutsLists]:
    """
    Sample, score, and filter spread layouts for a single combination.

    Finds all possible spread layouts via layout_combination, then scores
    each layout using page consistency penalties. Uses relaxed penalties
    for large groups (13+ photos).

    Args:
        comb: The Combination defining which photos go in each spread.
        photos: List of Photo objects in the group.
        layouts_df: DataFrame of available layout designs.
        params: Search parameters controlling sampling limits and thresholds.

    Returns:
        GroupLayoutsLists with scored and filtered layouts, or None if no
        valid layout exists for any spread.
    """
    # sample
    multispread_layouts = layout_combination(comb, layouts_df, photos, params)
    # evaluate + filter
    if multispread_layouts is not None:
        if len(photos) < 13:
            penalty = Penalties(
                crop_penalty=CONFIGS['crop_penalty'],
                color_mix=CONFIGS['color_mix'],
                class_mix=CONFIGS['class_mix'],
                orientation_mix=CONFIGS['orientation_mix'],
                score_threshold=params.score_threshold,
                double_mix_color=CONFIGS['double_page_color_mix']
            )
        else:
            penalty = Penalties(
                crop_penalty=0.8,
                color_mix=CONFIGS['color_mix'],
                class_mix=CONFIGS['class_mix'],
                orientation_mix=CONFIGS['orientation_mix'],
                score_threshold=params.score_threshold,
                double_mix_color=CONFIGS['double_page_color_mix'],
                context_mix_penalty=0.00001,
                time_order_penalty=0.5
            )
        multispread_layouts.process(layouts_df, photos, penalty)
    return multispread_layouts