from __future__ import annotations

from itertools import product
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Iterable, Callable, Any, Optional

from scipy.stats import pearsonr
import pandas as pd

from src.spreads_layout.spreads.group_of_lists_of_spreads import GroupLayoutsLists, process_combination_inner
from src.spreads_layout.spreads.spread import SingleSpreadLayout
from src.core.photos import Photo
from src.spreads_layout.combinations import Combination
from src.core.models import SpreadSearchParams


@dataclass
class GroupSingleLayout:
    spreads_layouts: List[SingleSpreadLayout]   # unordered actually (mutable)
    score: Optional[float] = None
    weight: Optional[float] = None

    def update_weight(self, factor: float) -> None:
        self.weight *= factor

    def get_score(self, layout_id2data: Dict[int, Any], photos: List[Photo]) -> float:
        """
        Calculate the correlation between box areas and photo importance ranks.

        For each spread, pairs layout box areas with the corresponding photo ranks,
        normalizes both to 0-1 scale, and computes Pearson correlation. Higher-ranked
        photos in larger boxes yield a higher score.

        Args:
            layout_id2data: Mapping from layout ID to layout data dict containing
                'boxes_areas', 'left_box_ids', and 'right_box_ids'.
            photos: List of Photo objects for the group.

        Returns:
            Correlation score mapped to [0, 1] range, or 0.1 if data is insufficient.
        """
        default_box_area, default_rank = 0.1, 0.3

        box_areas = []
        ranks = []

        for spread in self.spreads_layouts:
            layout = layout_id2data[spread.layout_idx]
            layout_boxes = layout['boxes_areas']

            left_boxes_ids = layout['left_box_ids']
            right_boxes_ids = layout['right_box_ids']

            left_photos = list(spread.left_page_photo_idxs)
            right_photos = list(spread.right_page_photo_idxs)

            layout_boxes_dict = {box['id']: box['area'] for box in layout_boxes}

            for box_id, photo_id in zip(left_boxes_ids, left_photos):
                photo_data = photos[photo_id]
                box_area = layout_boxes_dict.get(box_id, default_box_area)
                image_rank = photo_data.rank

                box_areas.append(box_area)
                ranks.append(image_rank)

            for box_id, photo_id in zip(right_boxes_ids, right_photos):
                photo_data = photos[photo_id]
                box_area = layout_boxes_dict.get(box_id, default_box_area)
                image_rank = photo_data.rank

                box_areas.append(box_area)
                ranks.append(image_rank)

            # Check if box_areas or ranks are empty or if there is no variation
            if not box_areas or not ranks or len(set(box_areas)) == 1 or len(set(ranks)) == 1:
                return 0.1  # Return a default correlation score (e.g., 0) if data is not suitable for correlation

        # Normalization (maps ranks and areas to a 0-1 scale)
        max_rank = max(ranks) if ranks else 1  # Avoid division by zero
        max_area = max(box_areas) if box_areas else 1

        normalized_ranks = [1 - (rank / max_rank) for rank in ranks]  # Lower ranks are better
        normalized_areas = [area / max_area for area in box_areas]  # Larger areas are better
        # Calculate Pearson correlation coefficient
        correlation, _ = pearsonr(normalized_ranks, normalized_areas)

        self.score = correlation / 2 + 0.5
        return self.score

    @staticmethod
    def evaluate_list(filtered_spreads: List[GroupSingleLayout], photos: List[Photo],
                          layout_id2data: Dict[int, Any]) -> None:
        """
        Multiply each group layout's weight by its box-area/rank correlation score.

        Args:
            filtered_spreads: List of GroupSingleLayout candidates to score.
            photos: List of Photo objects for the group.
            layout_id2data: Mapping from layout ID to layout data dict.
        """
        for group_layout in filtered_spreads:
            score = group_layout.get_score(layout_id2data, photos)
            group_layout.update_weight(score)

    def resolve_and_order(self, photos: List[Photo], layout_id2data: Dict[int, Any],
                          design_box_id2data: Dict[Tuple[int, int], Any],
                          merge_pages: bool = False):
        """
        Resolve photo indices to Photo objects and assign them to layout boxes.

        Two-step process for each spread in this group layout:
        1. resolve_photos: converts photo index sets (left/right_page_photo_idxs)
           into actual Photo object sets (left/right_page_photos).
        2. assign_photos_order: matches photos to layout boxes by orientation
           (portrait/landscape) and area (largest boxes get highest-ranked photos),
           producing ordered lists for left and right pages.

        Args:
            photos: Full list of Photo objects for the group (indexed by photo_idxs).
            layout_id2data: Mapping from layout index to layout metadata including
                'layout_id', 'left_box_ids', 'right_box_ids', and 'boxes_areas'.
            design_box_id2data: Mapping from (layout_id, box_id) to box properties
                including 'area' and 'orientation'.
            merge_pages: If True, pool photos from both pages before assignment
                (useful when page separation is not meaningful). If False, assign
                each page's photos independently.
        """
        for spread in self.spreads_layouts:
            # Retrieve Photo objects from photo indices
            spread.resolve_photos(photos)
            # Assign photos to layout boxes
            spread.assign_photos_order(layout_id2data, design_box_id2data, merge_pages)


def process_group_lists(group_spreads_layouts: GroupLayoutsLists) -> List[GroupSingleLayout]:
    """
    Flatten a GroupLayoutsLists into individual GroupSingleLayout candidates.

    For single-spread groups, wraps each spread layout directly. For multi-spread
    groups, computes the cartesian product of all spread layout options and
    multiplies their weights together with the combination weight.

    Args:
        group_spreads_layouts: GroupLayoutsLists containing possible layouts
            per spread for a single combination.

    Returns:
        List of GroupSingleLayout objects, each representing one complete
        layout option for the group.
    """
    listed_spreads = []
    n_spreads = len(group_spreads_layouts.spreads)
    spreads_in_group = group_spreads_layouts.possible_layouts

    if n_spreads == 1:
        for spread_layout in spreads_in_group[0].possible_layouts:
            listed_spreads.append(GroupSingleLayout(
                spreads_layouts=[spread_layout],
                weight=spread_layout.weight * group_spreads_layouts.weight
            ))
    else:
        merged = [[spread_layout] for spread_layout in spreads_in_group[0].possible_layouts]
        for spread_idx in range(1, n_spreads):
            merged = list(product(merged, spreads_in_group[spread_idx].possible_layouts))
            merged = [merged[idx][0] + [merged[idx][1]] for idx in range(len(merged))]

        for merge in merged:
            merge_weight = group_spreads_layouts.weight
            for spread in merge:
                merge_weight *= spread.weight

            listed_spreads.append(GroupSingleLayout(
                spreads_layouts=merge,
                weight= merge_weight
            ))

    return listed_spreads


def process_combination_outer(comb: Combination, photos: List[Photo],
                              layouts_df: pd.DataFrame, params: SpreadSearchParams,
                              group_single_layouts: List[GroupSingleLayout]) -> List[GroupSingleLayout]:
    """
    Process a single combination: sample spread layouts and accumulate results.

    Generates multi-spread layouts for the combination via process_combination_inner,
    flattens them into GroupSingleLayout candidates, and appends to the running list.
    Trims to top 1000 by score if the list exceeds 10000 to bound memory usage.

    Args:
        comb: A partition combination defining how photos are split across spreads.
        photos: Full list of Photo objects for the group.
        layouts_df: DataFrame of available layout designs.
        params: Search parameters controlling sampling limits.
        group_single_layouts: Running accumulator of GroupSingleLayout candidates.

    Returns:
        Updated group_single_layouts list with new candidates appended (and trimmed if needed).
    """
    # sample
    multispread_layouts = process_combination_inner(comb, photos, layouts_df, params)
    if multispread_layouts is not None:
        group_single_layouts += process_group_lists(multispread_layouts)

    # filter
    if len(group_single_layouts) > 10000:
        group_single_layouts = sorted(group_single_layouts, key=lambda layout: layout.score, reverse=True)[:1000]

    return group_single_layouts


def get_group_single_layouts(combs: List[Combination], photos: List[Photo],
                             layouts_df: pd.DataFrame, params: SpreadSearchParams,
                             layout_id2data: Dict[int, Any]) -> Optional[List[GroupSingleLayout]]:
    """
    Find the best GroupSingleLayout candidates for a photo group.

    Three stages:
    1. Sample: iterates over all combinations, generating and accumulating
       GroupSingleLayout candidates via process_combination_outer.
    2. Filter: keeps top 1000 candidates by weight (discarding those below
       1% of the max weight).
    3. Evaluate: scores remaining candidates by box-area/rank correlation
       (Pearson) and returns them sorted by final weight.

    Args:
        combs: List of partition combinations to evaluate.
        photos: Full list of Photo objects for the group.
        layouts_df: DataFrame of available layout designs.
        params: Search parameters controlling sampling limits.
        layout_id2data: Mapping from layout index to layout metadata.

    Returns:
        Sorted list of GroupSingleLayout candidates (best first), or None if
        no valid layouts were found for any combination.
    """
    # sample
    group_single_layouts = []
    for idx, comb in enumerate(combs):
        group_single_layouts = process_combination_outer(comb, photos, layouts_df, params, group_single_layouts)

    if len(group_single_layouts) == 0:
        return None

    # filter
    filtered = sorted(group_single_layouts, key=lambda layout: layout.weight, reverse=True)
    max_weight = filtered[0].weight
    filtered = [layout for layout in filtered if layout.weight / max_weight > 0.01][:1000]

    # evaluate
    GroupSingleLayout.evaluate_list(filtered, photos, layout_id2data)
    return sorted(filtered, key=lambda x: x.weight, reverse=True)
