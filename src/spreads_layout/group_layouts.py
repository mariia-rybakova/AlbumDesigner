from __future__ import annotations

from itertools import product
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Iterable, Callable, Any, Optional

from scipy.stats import pearsonr

from src.spreads_layout.spreads.group_of_lists_of_spreads import GroupLayoutsLists
from src.spreads_layout.spreads.spread import SingleSpreadLayout
from src.core.photos import Photo


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


def assign_photos_order(group_layout: GroupSingleLayout, layout_id2data: Dict[int, Any],
                        design_box_id2data: Dict[Tuple[int, int], Any],
                        merge_pages: bool = False) -> GroupSingleLayout:
    """
    Assign final photo ordering to each spread's left and right pages.

    For each spread, builds box metadata from the layout design data and assigns
    photos to boxes by matching orientation and area. When merge_pages is True,
    both pages are treated as a single pool; otherwise each page is ordered
    independently.

    Args:
        group_layout: The selected GroupSingleLayout with resolved photos.
        layout_id2data: Mapping from layout ID to layout data dict.
        design_box_id2data: Mapping from (layout_id, box_id) to box properties
            including 'area' and 'orientation'.
        merge_pages: If True, pool photos from both pages before ordering.

    Returns:
        The same GroupSingleLayout with photos reordered in each spread.
    """
    for spread in group_layout.spreads_layouts:
        spread.assign_photos_order(layout_id2data, design_box_id2data, merge_pages)

    return group_layout