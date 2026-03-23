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
from src.spreads_layout.spreads.list_of_spreads import SpreadLayoutsList
from src.spreads_layout.spreads.spread import Penalties


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

    def evaluate(self, layouts_df: pd.DataFrame, photos: List[Photo], penalty: Optional[Penalties] = None) -> None:
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

        # Evaluate layouts in all spreads
        for single_spread_layouts in self.possible_layouts:
            # Evaluate each spread in this combination
            for spread in single_spread_layouts.possible_layouts:
                spread.evaluate(photos, layouts_df, penalty)

            single_spread_layouts.filter_by_score_threshold(penalty.score_threshold)


def layout_combination(single_class_comb: Combination, layout_df: pd.DataFrame, photos: List[Photo], params: SpreadSearchParams) -> Optional[GroupLayoutsLists]:
    """
    Find all possible spread layouts for a single Combination.

    For each spread in the combination, filters available layouts by photo
    count and orientation, then searches for valid page assignments. Large
    single-spread groups with all-square layouts get a fast path. Otherwise,
    runs greedy heuristics followed by exhaustive oriented search, sampling
    down if too many candidates are found.

    Args:
        single_class_comb: The Combination defining which photos go in each spread.
        layout_df: Full DataFrame of available layout designs.
        photos: List of Photo objects in the group.
        params: Search parameters controlling sampling limits and thresholds.

    Returns:
        GroupLayoutsLists with possible layouts per spread, or None if any
        spread has no valid layout.
    """
    n_spreads = len(single_class_comb.spreads)
    group_spreads_layouts = GroupLayoutsLists.from_comb(single_class_comb)

    for photo_idx_set in single_class_comb.spreads:
        # spread_photos = list(photo_idx_set)

        if len(photo_idx_set) == 0:
            return None

        n_photos_in_spread = len(photo_idx_set)
        portrait_set, landscape_set = get_portraits_landscapes(photo_idx_set, photos)

        layouts_df = filter_layouts(layout_df, n_photos_in_spread, len(portrait_set), len(landscape_set))
        layouts_df = count_squares(layouts_df)

        # large spreads with squares gets trivial layout
        if is_large_spread_with_squares(n_photos_in_spread, n_spreads, layouts_df):
            single_spread_layouts = SpreadLayoutsList.simple_layout(photo_idx_set, layouts_df)
            group_spreads_layouts.add_spread(single_spread_layouts)
            return group_spreads_layouts

        # greedy attempt to find layout based on separation of time, class and color
        greedy_single_spreads = SpreadLayoutsList.greedy_layout_search(photo_idx_set, photos, layouts_df)
        # other layouts sampling
        oriented_spreads = SpreadLayoutsList.full_oriented_layout_search(landscape_set, portrait_set, layouts_df, params)

        greedy_single_spreads_l = limit_sample_size(greedy_single_spreads.possible_layouts, params.max_spreads_sample)
        oriented_spreads_l = limit_sample_size(oriented_spreads.possible_layouts, params.max_spreads_sample - len(greedy_single_spreads_l))

        single_spread_layouts = SpreadLayoutsList(photo_idx_set, greedy_single_spreads_l + oriented_spreads_l)

        if len(single_spread_layouts.possible_layouts) == 0:
            return None

        group_spreads_layouts.add_spread(single_spread_layouts)

    # group_spreads_layouts.view(limit=3)
    return group_spreads_layouts