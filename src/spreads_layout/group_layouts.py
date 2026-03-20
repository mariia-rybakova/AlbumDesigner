from __future__ import annotations

from itertools import product
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Iterable, Callable, Any, Optional

from scipy.stats import pearsonr

from src.spreads_layout.spread_layouts import GroupLayoutsLists, SingleSpreadLayout
from src.core.photos import Photo


@dataclass
class GroupSingleLayout:
    spreads_layouts: List[SingleSpreadLayout]   # unordered actually (mutable)
    score: float = None

    def update_score(self, factor: float) -> None:
        self.score *= factor


# sample

def list_multi_spreads(group_spreads_layouts: GroupLayoutsLists) -> List[GroupSingleLayout]:
    listed_spreads = []
    n_spreads = len(group_spreads_layouts.spreads)
    spreads_in_group = group_spreads_layouts.possible_layouts

    if n_spreads == 1:
        for spread_layout in spreads_in_group[0].possible_layouts:
            listed_spreads.append(GroupSingleLayout(
                spreads_layouts=[spread_layout],
                score=spread_layout.score * group_spreads_layouts.weight
            ))
    else:
        merged = [[spread_layout] for spread_layout in spreads_in_group[0].possible_layouts]
        for spread_idx in range(1, n_spreads):
            merged = list(product(merged, spreads_in_group[spread_idx].possible_layouts))
            merged = [merged[idx][0] + [merged[idx][1]] for idx in range(len(merged))]

        for merge in merged:
            merge_score = 1
            for spread in merge:
                merge_score *= spread.score

            listed_spreads.append(GroupSingleLayout(
                spreads_layouts=merge,
                score=merge_score * group_spreads_layouts.weight
            ))

    return listed_spreads


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


def _assign_part_photos_order(boxes: List[Dict[str, Any]],
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

    left_size = max([box['position'] for box in boxes if box['side'] == 0], default=0) + 1
    right_size = max([box['position'] for box in boxes if box['side'] == 1], default=0) + 1
    left_photos_order = [None] * left_size
    right_photos_order = [None] * right_size

    for area in sorted(area2boxes.keys(), reverse=True):
        cur_boxes = area2boxes[area]
        photos_order, portraits_total, landscapes_total = _assign_photos_order_by_area(photos, cur_boxes,
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
        layout_data = layout_id2data[spread.layout_idx]
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
            all_photos = sorted(list(spread.left_page_photos) + list(spread.right_page_photos),
                                key=lambda ph: (ph.rank, ph.general_time))
            left_photos_order, right_photos_order = _assign_part_photos_order(left_page_boxes + right_page_boxes, all_photos)
        else:
            left_page_photos = sorted(spread.left_page_photos, key=lambda ph: (ph.rank, ph.general_time))
            right_page_photos = sorted(spread.right_page_photos, key=lambda ph: (ph.rank, ph.general_time))
            left_photos_order, _ = _assign_part_photos_order(left_page_boxes, left_page_photos)
            _, right_photos_order = _assign_part_photos_order(right_page_boxes, right_page_photos)

        spread.set_photos_order(left_photos_order, right_photos_order)

    return group_layout


# evaluation



def calculate_correlation_score(layout_id2data: Dict[int, Any], photos: List[Photo],
                                all_spreads_data: List[SingleSpreadLayout]) -> float:
    """
    Calculate the correlation between box areas and photo importance ranks.

    For each spread, pairs layout box areas with the corresponding photo ranks,
    normalizes both to 0-1 scale, and computes Pearson correlation. Higher-ranked
    photos in larger boxes yield a higher score.

    Args:
        layout_id2data: Mapping from layout ID to layout data dict containing
            'boxes_areas', 'left_box_ids', and 'right_box_ids'.
        photos: List of Photo objects for the group.
        all_spreads_data: List of SingleSpreadLayout objects with photo assignments.

    Returns:
        Correlation score mapped to [0, 1] range, or 0.1 if data is insufficient.
    """
    default_box_area, default_rank = 0.1, 0.3

    box_areas = []
    ranks = []

    for spread in all_spreads_data:
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

    return correlation / 2 + 0.5


def add_ranking_score(filtered_spreads: List[GroupSingleLayout], photos: List[Photo],
                      layout_id2data: Dict[int, Any]) -> List[GroupSingleLayout]:
    """
    Multiply each group layout's score by its box-area/rank correlation score.

    Args:
        filtered_spreads: List of GroupSingleLayout candidates to score.
        photos: List of Photo objects for the group.
        layout_id2data: Mapping from layout ID to layout data dict.

    Returns:
        The same list with updated scores.
    """
    for group_layout in filtered_spreads:
        correlation_score = calculate_correlation_score(layout_id2data, photos, group_layout.spreads_layouts)
        group_layout.update_score(correlation_score)

    return filtered_spreads
