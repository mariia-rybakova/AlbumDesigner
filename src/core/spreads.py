from __future__ import annotations

import random
from itertools import combinations, product, permutations
from dataclasses import dataclass
from typing import List, Tuple, Set, Iterable, Callable, Any, Optional
import inspect

import numpy as np
import pandas as pd

from src.spreads_layout.layouts_tools import (filter_layouts, count_squares,
                                              update_with_page_capacities, apply_layouts_mask)
from src.spreads_layout.partitions import Partition, get_partitions
from src.spreads_layout.combinations import Combination, get_combinations
from src.core.models import SpreadSearchParams
from src.core.photos import group_photos, get_portraits_landscapes
from utils.configs import CONFIGS


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

    def resolve_photos(self, group_photos: List) -> None:
        self.left_page_photos = {group_photos[idx] for idx in self.left_page_photo_idxs}
        self.right_page_photos = {group_photos[idx] for idx in self.right_page_photo_idxs}

    def set_photos_order(self, left_ordered: List, right_ordered: List) -> None:
        self.left_page_photos = left_ordered
        self.right_page_photos = right_ordered


@dataclass
class SpreadLayoutsList:
    spread_photo_idxs: Set[int]
    possible_layouts: List[SingleSpreadLayout] = None

    def view(self, limit: Optional[int] = None, sep: str = '==') -> None:
        print('Possible layouts for spread with photos:', self.spread_photo_idxs, f'- {len(self.possible_layouts)} options')
        for j, sp in enumerate(self.possible_layouts):
            if limit is not None and j > limit:
                print(sep, '... ... ...')
                break
            print(sep, j + 1, sp)

    def update_layouts(self, layouts_list: List[SingleSpreadLayout]) -> None:
        self.possible_layouts = layouts_list


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


def _is_large_spread_with_squares(n_photos_in_spread: int, n_spreads: int, layouts_df: pd.DataFrame) -> bool:
    """
    Check if a spread qualifies for the trivial all-squares layout shortcut.

    A spread qualifies when it is the only spread in the group, contains more
    than 13 photos, and a layout exists where every box is a square.

    Args:
        n_photos_in_spread: Number of photos in the current spread.
        n_spreads: Total number of spreads in the group.
        layouts_df: DataFrame of layouts with a 'number of squares' column.

    Returns:
        True if the spread should use the simple all-squares layout.
    """
    return (
            n_photos_in_spread > 13 and
            len(layouts_df[layouts_df['number of squares'] == n_photos_in_spread]) > 0 and
            n_spreads == 1
    )


def _simple_layout(layouts_df: pd.DataFrame, n_photos: int) -> List[SingleSpreadLayout]:
    selected_layouts = layouts_df[layouts_df['number of squares'] == n_photos]
    single_spreads = []

    for layout_idx, layout in selected_layouts.iterrows():
        single_spreads.append(
            SingleSpreadLayout(
                layout_idx=layout_idx,
                left_page_photo_idxs= set(range(0,                                len(layout['left_square_ids']))),
                right_page_photo_idxs=set(range(len(layout['left_square_ids']),   n_photos)),
                number_of_squares=n_photos
            )
        )
    return single_spreads


def _time_page_separation(spread_photos: List[int], photos: List, greedy_layouts: pd.DataFrame, greedy_single_spreads: List[SingleSpreadLayout]) -> List[SingleSpreadLayout]:
    """
    Try to split photos across left/right pages by time-based context groups.

    Groups photos by (original_context, color) using group_photos. If exactly
    two groups are found, treats them as left and right page candidates, finds
    compatible layouts based on orientation counts, and appends matching
    SingleSpreadLayouts to the greedy list.

    Args:
        spread_photos: Photo indices in the current spread.
        photos: Full list of Photo objects.
        greedy_layouts: DataFrame of layouts with page capacity columns.
        greedy_single_spreads: Accumulator list to append new layouts to.

    Returns:
        The updated greedy_single_spreads list (with new layouts appended if
        a two-group split was possible).
    """
    grouped_sequences = group_photos(spread_photos, photos)

    if len(grouped_sequences) == 2:
        left_landscapes = np.sum([photos[item[0]].ar > 1 for item in grouped_sequences[0]])
        left_portraits = len(grouped_sequences[0]) - left_landscapes
        right_landscapes = np.sum([photos[item[0]].ar > 1 for item in grouped_sequences[1]])
        right_portraits = len(grouped_sequences[1]) - right_landscapes

        possible_layouts = apply_layouts_mask(greedy_layouts, left_landscapes, left_portraits,
                                              right_landscapes, right_portraits)

        for layout_idx, layout in possible_layouts.iterrows():
            greedy_single_spreads.append(
                SingleSpreadLayout(
                    layout_idx=layout_idx,
                    left_page_photo_idxs= set([item[0] for item in grouped_sequences[0]]),
                    right_page_photo_idxs=set([item[0] for item in grouped_sequences[1]]),
                    number_of_squares=len(list(layout['left_square_ids']) + list(layout['right_square_ids']))
                )
            )

    return greedy_single_spreads


def _color_page_separation(spread_photos: List[int], photos: List, greedy_layouts: pd.DataFrame, greedy_single_spreads: List[SingleSpreadLayout]) -> List[SingleSpreadLayout]:
    """
    Try to split photos across left/right pages by color vs grayscale.

    When the spread contains both color and grayscale photos, assigns the
    group whose mean time is earlier to the left page (color-before-gray or
    vice versa). Finds compatible layouts based on per-page orientation counts
    and appends matching SingleSpreadLayouts to the greedy list.

    Args:
        spread_photos: Photo indices in the current spread.
        photos: Full list of Photo objects.
        greedy_layouts: DataFrame of layouts with page capacity columns.
        greedy_single_spreads: Accumulator list to append new layouts to.

    Returns:
        The updated greedy_single_spreads list (with new layouts appended if
        a color/grayscale split was possible).
    """
    colors = [photos[photo_id].color for photo_id in spread_photos]

    if len(set(colors)) == 2:
        photos_color = np.array([photos[photo_id].color for photo_id in spread_photos])
        color_time = np.mean([photos[photo_id].general_time for photo_id in spread_photos if photos[photo_id].color])
        gray_time = np.mean([photos[photo_id].general_time for photo_id in spread_photos if not photos[photo_id].color])

        if gray_time > color_time:
            left_condition = True
        else:
            left_condition = False

        left_landscapes =  np.sum([photos[item].ar > 1 and photos[item].color == left_condition for item in spread_photos])
        right_landscapes = np.sum([photos[item].ar > 1 and photos[item].color != left_condition for item in spread_photos])
        left_portraits =  np.sum(photos_color == left_condition) - left_landscapes
        right_portraits = np.sum(photos_color != left_condition) - right_landscapes

        possible_layouts = apply_layouts_mask(greedy_layouts, left_landscapes, left_portraits, right_landscapes,
                                       right_portraits)

        for layout_idx, layout in possible_layouts.iterrows():
            greedy_single_spreads.append(
                SingleSpreadLayout(
                    layout_idx=layout_idx,
                    left_page_photo_idxs= set([photo_id for photo_id in spread_photos if photos[photo_id].color == left_condition]),
                    right_page_photo_idxs=set([photo_id for photo_id in spread_photos if photos[photo_id].color != left_condition]),
                    number_of_squares=len(list(layout['left_square_ids']) + list(layout['right_square_ids']))
                )
            )

    return greedy_single_spreads


def greedy_layout_search(spread_photos: List[int], photos: List, layouts_df: pd.DataFrame) -> List[SingleSpreadLayout]:
    """
    Attempt to find spread layouts using greedy heuristics before exhaustive search.

    Tries two strategies in sequence: splitting photos by time/context groups
    and splitting by color/grayscale. Each strategy appends compatible layouts
    to the result list. Silently catches exceptions so the caller can fall back
    to exhaustive search.

    Args:
        spread_photos: Photo indices in the current spread.
        photos: Full list of Photo objects.
        layouts_df: DataFrame of filtered layouts for this spread size.

    Returns:
        List of SingleSpreadLayout candidates found by greedy heuristics.
        May be empty if no heuristic produced a valid split.
    """
    greedy_single_spreads = []
    try:
        if len(layouts_df) > 0:
            layouts_df = update_with_page_capacities(layouts_df)

            greedy_single_spreads = _time_page_separation(spread_photos, photos, layouts_df, greedy_single_spreads)
            greedy_single_spreads = _color_page_separation(spread_photos, photos, layouts_df, greedy_single_spreads)
    except Exception as e:
        print(f"Greedy layout attempt failed with error {e}")
    return greedy_single_spreads


def get_oriented_combs(landscape_spread_photo_idxs: Set[int], portrait_spread_photo_idxs: Set[int],
                       n_page_landscapes: int, n_page_portraits: int) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """
    Generate all (landscape, portrait) orientation combinations for a page.

    Computes the cartesian product of all ways to choose n_page_landscapes
    from the available landscapes and n_page_portraits from the available
    portraits.

    Args:
        landscape_spread_photo_idxs: Available landscape photo indices.
        portrait_spread_photo_idxs: Available portrait photo indices.
        n_page_landscapes: Number of landscape slots on the page.
        n_page_portraits: Number of portrait slots on the page.

    Returns:
        List of tuples, each containing two inner tuples:
        - landscape_tuple: A selection of n_page_landscapes photo indices
          chosen from landscape_spread_photo_idxs.
        - portrait_tuple: A selection of n_page_portraits photo indices
          chosen from portrait_spread_photo_idxs.
        The list contains every possible pairing (cartesian product) of
        landscape and portrait selections.
    """
    landscape_combs = list(combinations(landscape_spread_photo_idxs, n_page_landscapes))
    portrait_combs = list(combinations(portrait_spread_photo_idxs, n_page_portraits))
    oriented_combs = list(product(landscape_combs, portrait_combs))
    return oriented_combs


@dataclass
class OrientedSpread:
    """
    A candidate left/right page split with remaining unassigned photos.

    Tracks which photo indices go on each page and which landscape/portrait
    photos are still available for square-slot assignment.

    Attributes:
        left_page_photo_idxs: Photo indices assigned to the left page.
        right_page_photo_idxs: Photo indices assigned to the right page.
        rem_landscapes: Landscape photo indices not yet assigned to either page.
        rem_portraits: Portrait photo indices not yet assigned to either page.
    """
    left_page_photo_idxs: Set[int]
    right_page_photo_idxs: Set[int]
    rem_landscapes: Set[int]
    rem_portraits: Set[int]


def _get_left_pages(landscape_set: Set[int], portrait_set: Set[int],
                    n_left_landscapes: int, n_left_portraits: int,
                    max_oriented_combs: int) -> List[OrientedSpread]:
    """
    Build left-page photo assignments from all landscape/portrait orientation
    combinations, sampling if the count exceeds max_oriented_combs.

    Args:
        landscape_set: Full set of landscape photo indices in this spread.
        portrait_set: Full set of portrait photo indices in this spread.
        n_left_landscapes: Number of landscape slots on the left page.
        n_left_portraits: Number of portrait slots on the left page.
        max_oriented_combs: Maximum number of orientation combinations to keep.

    Returns:
        List of OrientedSpread objects with left_page_photo_idxs populated
        and right_page_photo_idxs empty (to be filled by _get_right_pages).
    """
    oriented_combs = get_oriented_combs(landscape_set, portrait_set, n_left_landscapes, n_left_portraits)

    if len(oriented_combs) > max_oriented_combs:
        sample_idxs = random.sample(range(len(oriented_combs)), max_oriented_combs)
        oriented_combs = [oriented_combs[i] for i in sample_idxs]

    results = []
    for landscape_tuple, portrait_tuple in oriented_combs:
        results.append(OrientedSpread(
            left_page_photo_idxs=set(landscape_tuple) | set(portrait_tuple),
            right_page_photo_idxs=set(),
            rem_landscapes=landscape_set - set(landscape_tuple),
            rem_portraits=portrait_set - set(portrait_tuple),
        ))
    return results


def _get_single_right_page(oriented_combs: List[Tuple], left_spread: OrientedSpread, results: List[OrientedSpread]) -> List[OrientedSpread]:
    """
    Build right-page assignments for a single left-page option.

    For each orientation combination, assigns landscapes and portraits to the
    right page using what remains after the left-page assignment. Pairs each
    result with the left_spread's left page to form a complete OrientedSpread.

    Args:
        oriented_combs: List of (landscape_tuple, portrait_tuple) combinations
            for the right page.
        left_spread: OrientedSpread with left page populated and remaining
            photos from _get_left_pages.
        results: Accumulator list to append completed OrientedSpreads to.

    Returns:
        The results list with new OrientedSpread entries appended.
    """
    for landscape_tuple, portrait_tuple in oriented_combs:
        results.append(OrientedSpread(
            left_page_photo_idxs=left_spread.left_page_photo_idxs,
            right_page_photo_idxs=set(landscape_tuple) | set(portrait_tuple),
            rem_landscapes=left_spread.rem_landscapes - set(landscape_tuple),
            rem_portraits=left_spread.rem_portraits - set(portrait_tuple),
        ))
    return results


def _get_right_pages(n_right_landscapes: int, n_right_portraits: int, left_spreads: List[OrientedSpread]) -> List[OrientedSpread]:
    """
    Build right-page assignments for all left-page options.

    For each left-page OrientedSpread, enumerates all landscape/portrait
    combinations that fit the right page's orientation requirements using
    the remaining photos, then delegates to _get_single_right_page to build
    complete OrientedSpreads with both pages populated.

    Args:
        n_right_landscapes: Number of landscape slots on the right page.
        n_right_portraits: Number of portrait slots on the right page.
        left_spreads: OrientedSpread objects from _get_left_pages with left
            pages populated and remaining photo sets.

    Returns:
        List of fully populated OrientedSpread objects with both left and
        right pages assigned and remaining photos for square-slot filling.
    """
    results = []
    for left_spread in left_spreads:
        oriented_combs = get_oriented_combs(left_spread.rem_landscapes, left_spread.rem_portraits, n_right_landscapes, n_right_portraits)
        results = _get_single_right_page(oriented_combs, left_spread, results)
    return results


def _process_squares(oriented_spreads: List[OrientedSpread], n_left_squares: int, n_right_squares: int,
                     layout_idx: int) -> List[SingleSpreadLayout]:
    """
    Expand oriented spreads into SingleSpreadLayouts by distributing remaining
    photos into square slots on left and right pages.

    Args:
        oriented_spreads: List of OrientedSpread objects with pages and
            remaining photos populated.
        n_left_squares: Number of square slots on the left page.
        n_right_squares: Number of square slots on the right page.
        layout_idx: Layout index to assign to each resulting SingleSpreadLayout.
        single_spreads: Accumulator list to append new layouts to.

    Returns:
        The updated single_spreads list with new layouts appended.
    """
    single_spreads = []
    for spread in oriented_spreads:
        rem_photos = spread.rem_landscapes | spread.rem_portraits
        square_combs = list(combinations(rem_photos, n_left_squares))
        for comb in square_combs:
            single_spreads.append(
                SingleSpreadLayout(
                    layout_idx=layout_idx,
                    left_page_photo_idxs=spread.left_page_photo_idxs | set(comb),
                    right_page_photo_idxs=spread.right_page_photo_idxs | (rem_photos - set(comb)),
                    number_of_squares=n_left_squares + n_right_squares
                )
            )
    return single_spreads


def full_oriented_layout_search(layouts_df: pd.DataFrame, landscape_set: Set[int], portrait_set: Set[int], params: SpreadSearchParams) -> List[SingleSpreadLayout]:
    """
    Exhaustive search for spread layouts across all available designs.

    For each layout design, enumerates all valid left/right page assignments
    based on orientation (portrait/landscape), then expands square slots.

    Args:
        layouts_df: DataFrame of filtered layout designs for this spread size.
        landscape_set: Landscape photo indices in this spread.
        portrait_set: Portrait photo indices in this spread.
        params: Search parameters controlling max oriented combinations.

    Returns:
        List of all SingleSpreadLayout candidates from exhaustive search.
    """
    spreads = []
    for layout_idx in layouts_df.index:
        n_left_landscapes = len(layouts_df.at[layout_idx, 'left_landscape_ids'])
        n_left_portraits = len(layouts_df.at[layout_idx, 'left_portrait_ids'])
        n_right_landscapes = len(layouts_df.at[layout_idx, 'right_landscape_ids'])
        n_right_portraits = len(layouts_df.at[layout_idx, 'right_portrait_ids'])
        n_left_squares = len(layouts_df.at[layout_idx, 'left_square_ids'])
        n_right_squares = len(layouts_df.at[layout_idx, 'right_square_ids'])

        left_spreads = _get_left_pages(landscape_set, portrait_set, n_left_landscapes, n_left_portraits, params.max_oriented_combs)
        oriented_spreads = _get_right_pages(n_right_landscapes, n_right_portraits, left_spreads)

        # single_spreads = greedy_single_spreads.copy()
        single_spreads = _process_squares(oriented_spreads, n_left_squares, n_right_squares, layout_idx)

        spreads += single_spreads
    return spreads


def layout_combination(single_class_comb: Combination, layout_df: pd.DataFrame, photos: List, params: SpreadSearchParams) -> Optional[GroupLayoutsLists]:
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
        spread_photos = list(photo_idx_set)

        if len(spread_photos) == 0:
            spread_photos # ToDo ???

        n_photos_in_spread = len(spread_photos)
        portrait_set, landscape_set = get_portraits_landscapes(spread_photos, photos)

        layouts_df = filter_layouts(layout_df, n_photos_in_spread, len(portrait_set), len(landscape_set))
        layouts_df = count_squares(layouts_df)

        # large spreads with squares gets trivial layout
        if _is_large_spread_with_squares(n_photos_in_spread, n_spreads, layouts_df):
            single_spreads = _simple_layout(layouts_df, n_photos_in_spread)
            single_spread_layouts = SpreadLayoutsList(photo_idx_set, single_spreads)
            group_spreads_layouts.add_spread(single_spread_layouts)
            return group_spreads_layouts

        # greedy attempt to find layout based on separation of time, class and color
        greedy_single_spreads = greedy_layout_search(spread_photos, photos, layouts_df)
        # other layouts sampling
        oriented_spreads = full_oriented_layout_search(layouts_df, landscape_set, portrait_set, params)

        def limit_sample_size(objects_list, max_threshold):
            if len(objects_list) > max_threshold:
                sample_idxs = random.sample(range(len(objects_list)), max_threshold)
                objects_list = [objects_list[i] for i in sample_idxs]
            return objects_list

        greedy_single_spreads = limit_sample_size(greedy_single_spreads, params.max_spreads_sample)
        oriented_spreads = limit_sample_size(oriented_spreads, params.max_spreads_sample - len(greedy_single_spreads))

        spreads = greedy_single_spreads + oriented_spreads

        if len(spreads) == 0:
            return None

        single_spread_layouts = SpreadLayoutsList(photo_idx_set, spreads)
        group_spreads_layouts.add_spread(single_spread_layouts)

    # group_spreads_layouts.view(limit=3)
    return group_spreads_layouts


@dataclass
class PageEvaluationResult:
    is_same_color: bool
    is_same_class: bool
    is_bride_groom_mix: bool
    number_of_unique_contexts: int


def check_page(photo_set: Set[int], photos: List) -> PageEvaluationResult:
    bride_centric_classes = ['bride', 'bride party', 'wedding dress', 'getting hair-makeup','bride getting dressed']
    groom_centric_classes = ['groom','groom party','suit']

    if len(photo_set) == 1:
        return PageEvaluationResult(True, True, False, 1)

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

    return PageEvaluationResult(
        is_same_color=is_same_color,
        is_same_class=is_same_class,
        is_bride_groom_mix=calculate_bride_groom_mix() if is_same_class else False,
        number_of_unique_contexts=number_of_unique_contexts
    )


@dataclass()
class Penalties:
    crop_penalty: float = 0.5
    color_mix: float = 0.000000001
    class_mix: float = 0.01
    orientation_mix: float = 0.1
    score_threshold: float = 0.01
    double_mix_color: float = 0.000000000000000001
    context_mix_penalty: float = 0.00001
    time_order_penalty: float = 0.005


def apply_page_penalties(page_check_result: PageEvaluationResult, score: float, penalty: Penalties) -> float:
    if not page_check_result.is_same_color:
        score = score * penalty.color_mix
    if not page_check_result.is_same_class:
        score = score * penalty.class_mix
    if page_check_result.is_bride_groom_mix:
        score = score * penalty.color_mix

    score = score * np.power(penalty.context_mix_penalty, max(1, page_check_result.number_of_unique_contexts) - 1)
    return score


def eval_multi_spreads(group_spreads_layouts: GroupLayoutsLists, layouts_df: pd.DataFrame, photos: List, penalty: Optional[Penalties] = None) -> GroupLayoutsLists:
    if penalty is None:
        penalty = Penalties()
    #print(f"the CONFIGS['spread_score_threshold'] is {penalty.score_threshold}")

    # Evaluate layouts in all spreads
    for single_spread_layouts in group_spreads_layouts.possible_layouts:
        # Evaluate each spread in this combination
        for spread in single_spread_layouts.possible_layouts:
            score = 1.0

            left_check = check_page(spread.left_page_photo_idxs, photos)
            score = apply_page_penalties(left_check, score, penalty)
            if layouts_df.at[spread.layout_idx, 'left_mixed']:
                score = score * penalty.orientation_mix

            right_check = check_page(spread.right_page_photo_idxs, photos)
            score = apply_page_penalties(right_check, score, penalty)
            if layouts_df.at[spread.layout_idx, 'right_mixed']:
                score = score * penalty.orientation_mix

            # if two pages has gray colors, give it much worse rating
            if not left_check.is_same_color and not right_check.is_same_color:
                score = score * penalty.double_mix_color

            # penalty for cropping photos to square boxes
            score = score * np.power(penalty.crop_penalty, spread.number_of_squares)

            # if time order is not correct, give it a penalty
            photo_order_time = [photos[photo_id].general_time for photo_id in
                                list(spread.left_page_photo_idxs) + list(spread.right_page_photo_idxs)]
            for time_idx1 in range(len(photo_order_time)):
                for time_idx2 in range(time_idx1 + 1, len(photo_order_time)):
                    if photo_order_time[time_idx1] > photo_order_time[time_idx2]:
                        score = score * penalty.time_order_penalty

            spread.score = score

        if len(single_spread_layouts.possible_layouts) > 0:
            max_score = max(spread.score for spread in single_spread_layouts.possible_layouts)
            filtered = [spread for spread in single_spread_layouts.possible_layouts
                        if spread.score / max_score > penalty.score_threshold]
            single_spread_layouts.update_layouts(filtered)

    return group_spreads_layouts


@dataclass
class GroupSingleLayout:
    spreads_layouts: List[SingleSpreadLayout]   # unordered actually (mutable)
    score: float = None

    def update_score(self, factor: float) -> None:
        self.score *= factor


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


def generate_filtered_multi_spreads(photos: List, layouts_df: pd.DataFrame, spread_params: List[float], params: SpreadSearchParams, logger) -> Optional[List[GroupSingleLayout]]:
    photos_df = pd.DataFrame([photo.__dict__ for photo in photos])
    photos_df = photos_df.sort_values('general_time')
    partitions = get_partitions(photos_df, spread_params, params, layouts_df=layouts_df)
    # logger.info('Number of photos: {}. Possible partitions: {}'.format(len(photos), layout_parts))

    combs = get_combinations(partitions, photos, layouts_df, spread_params, params)

    #print("Getting the filtered multi srpreads")
    group_single_layouts = []
    for idx, comb in enumerate(combs):
        multispread_layouts = layout_combination(comb, layouts_df, photos, params)
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
            multispread_layouts = eval_multi_spreads(multispread_layouts, layouts_df, photos, penalty)
            group_single_layouts += list_multi_spreads(multispread_layouts)

        if len(group_single_layouts) > 10000:
            group_single_layouts = sorted(group_single_layouts, key=lambda layout: layout.score, reverse=True)[:1000]

    if len(group_single_layouts) == 0:
        return None

    filtered = sorted(group_single_layouts, key=lambda layout: layout.score, reverse=True)
    max_score = filtered[0].score
    filtered = [layout for layout in filtered if layout.score / max_score > 0.01]

    return filtered[:1000]

