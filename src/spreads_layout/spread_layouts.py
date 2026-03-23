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

    def evaluate(self, photos: List[Photo], layouts_df: pd.DataFrame, penalty: Penalties):
        score = 1.0

        left_check = check_page_properties(self.left_page_photo_idxs, photos)
        score = apply_page_penalties(left_check, score, penalty)
        if layouts_df.at[self.layout_idx, 'left_mixed']:
            score = score * penalty.orientation_mix

        right_check = check_page_properties(self.right_page_photo_idxs, photos)
        score = apply_page_penalties(right_check, score, penalty)
        if layouts_df.at[self.layout_idx, 'right_mixed']:
            score = score * penalty.orientation_mix

        # if two pages has gray colors, give it much worse rating
        if not left_check.is_same_color and not right_check.is_same_color:
            score = score * penalty.double_mix_color

        # penalty for cropping photos to square boxes
        score = score * np.power(penalty.crop_penalty, self.number_of_squares)

        # if time order is not correct, give it a penalty
        photo_order_time = [photos[photo_id].general_time for photo_id in
                            list(self.left_page_photo_idxs) + list(self.right_page_photo_idxs)]
        for time_idx1 in range(len(photo_order_time)):
            for time_idx2 in range(time_idx1 + 1, len(photo_order_time)):
                if photo_order_time[time_idx1] > photo_order_time[time_idx2]:
                    score = score * penalty.time_order_penalty

        self.score = score
        return score


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

    @classmethod
    def simple_layout(cls, spread_photo_idxs: Set[int], layouts_df: pd.DataFrame) -> SpreadLayoutsList:
        """
        Create trivial layouts where all boxes are squares.

        Selects layouts where the number of square boxes equals the photo count,
        then assigns photos sequentially: left page gets indices 0..left_count-1,
        right page gets the rest.

        Args:
            spread_photo_idxs: Set of photo indices for this spread.
            layouts_df: DataFrame of layouts with a 'number of squares' column.

        Returns:
            New SpreadLayoutsList with possible_layouts populated.
        """
        ll = cls(spread_photo_idxs=spread_photo_idxs)

        n_photos = len(ll.spread_photo_idxs)
        selected_layouts = layouts_df[layouts_df['number of squares'] == n_photos]

        for layout_idx, layout in selected_layouts.iterrows():
            ll.possible_layouts.append(
                SingleSpreadLayout(
                    layout_idx=layout_idx,
                    left_page_photo_idxs= set(range(0,                                len(layout['left_square_ids']))),
                    right_page_photo_idxs=set(range(len(layout['left_square_ids']),   n_photos)),
                    number_of_squares=n_photos
                )
            )
        return ll

    def _context_page_separation(self, photos: List, greedy_layouts: pd.DataFrame) -> None:
        """
        Try to split photos across left/right pages by time-based context groups.

        Groups photos by (original_context, color) using group_photos. If exactly
        two groups are found, treats them as left and right page candidates, finds
        compatible layouts based on orientation counts, and appends matching
        SingleSpreadLayouts to self.possible_layouts.

        Args:
            spread_photos: Photo indices in the current spread.
            photos: Full list of Photo objects.
            greedy_layouts: DataFrame of layouts with page capacity columns.
        """
        grouped_sequences = group_photos(list(self.spread_photo_idxs), photos)

        if len(grouped_sequences) == 2:
            left_landscapes = np.sum([photos[item[0]].ar > 1 for item in grouped_sequences[0]])
            left_portraits = len(grouped_sequences[0]) - left_landscapes
            right_landscapes = np.sum([photos[item[0]].ar > 1 for item in grouped_sequences[1]])
            right_portraits = len(grouped_sequences[1]) - right_landscapes

            possible_layouts = apply_layouts_mask(greedy_layouts, left_landscapes, left_portraits,
                                                  right_landscapes, right_portraits)

            for layout_idx, layout in possible_layouts.iterrows():
                self.possible_layouts.append(
                    SingleSpreadLayout(
                        layout_idx=layout_idx,
                        left_page_photo_idxs= set([item[0] for item in grouped_sequences[0]]),
                        right_page_photo_idxs=set([item[0] for item in grouped_sequences[1]]),
                        number_of_squares=len(list(layout['left_square_ids']) + list(layout['right_square_ids']))
                    )
                )

    def _color_page_separation(self, photos: List, greedy_layouts: pd.DataFrame) -> None:
        """
        Try to split photos across left/right pages by color vs grayscale.

        When the spread contains both color and grayscale photos, assigns the
        group whose mean time is earlier to the left page (color-before-gray or
        vice versa). Finds compatible layouts based on per-page orientation counts
        and appends matching SingleSpreadLayouts to self.possible_layouts.

        Args:
            spread_photos: Photo indices in the current spread.
            photos: Full list of Photo objects.
            greedy_layouts: DataFrame of layouts with page capacity columns.
        """
        spread_photos = list(self.spread_photo_idxs)
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

            possible_layouts_df = apply_layouts_mask(greedy_layouts, left_landscapes, left_portraits, right_landscapes,
                                           right_portraits)

            for layout_idx, layout in possible_layouts_df.iterrows():
                self.possible_layouts.append(
                    SingleSpreadLayout(
                        layout_idx=layout_idx,
                        left_page_photo_idxs= set([photo_id for photo_id in spread_photos if photos[photo_id].color == left_condition]),
                        right_page_photo_idxs=set([photo_id for photo_id in spread_photos if photos[photo_id].color != left_condition]),
                        number_of_squares=len(list(layout['left_square_ids']) + list(layout['right_square_ids']))
                    )
                )

    @classmethod
    def greedy_layout_search(cls, spread_photos: Set[int], photos: List, layouts_df: pd.DataFrame) -> SpreadLayoutsList:
        """
        Attempt to find spread layouts using greedy heuristics before exhaustive search.

        Tries two strategies in sequence: splitting photos by time/context groups
        and splitting by color/grayscale. Each strategy appends compatible layouts
        to self.possible_layouts. Silently catches exceptions so the caller can
        fall back to exhaustive search.

        Args:
            spread_photos: Photo indices in the current spread.
            photos: Full list of Photo objects.
            layouts_df: DataFrame of filtered layouts for this spread size.

        Returns:
            self with possible_layouts populated by greedy heuristics.
        """
        ll = cls(spread_photo_idxs=spread_photos, possible_layouts=[])

        try:
            if len(layouts_df) > 0:
                layouts_df = update_with_page_capacities(layouts_df)

                ll._context_page_separation(photos, layouts_df)
                ll._color_page_separation(photos, layouts_df)
        except Exception as e:
            print(f"Greedy layout attempt failed with error {e}")
        return ll

    @staticmethod
    def _get_oriented_combs(landscape_spread_photo_idxs: Set[int], portrait_spread_photo_idxs: Set[int],
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

    @classmethod
    def _get_left_pages(cls, landscape_set: Set[int], portrait_set: Set[int],
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
        oriented_combs = cls._get_oriented_combs(landscape_set, portrait_set, n_left_landscapes, n_left_portraits)

        if len(oriented_combs) > max_oriented_combs:
            sample_idxs = random.sample(range(len(oriented_combs)), max_oriented_combs)
            oriented_combs = [oriented_combs[i] for i in sample_idxs]

        results = []
        for landscape_tuple, portrait_tuple in oriented_combs:
            results.append(cls.OrientedSpread(
                left_page_photo_idxs=set(landscape_tuple) | set(portrait_tuple),
                right_page_photo_idxs=set(),
                rem_landscapes=landscape_set - set(landscape_tuple),
                rem_portraits=portrait_set - set(portrait_tuple),
            ))
        return results

    @classmethod
    def _get_single_right_page(cls, oriented_combs: List[Tuple], left_spread: OrientedSpread,
                               results: List[OrientedSpread]) -> List[OrientedSpread]:
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
            results.append(cls.OrientedSpread(
                left_page_photo_idxs=left_spread.left_page_photo_idxs,
                right_page_photo_idxs=set(landscape_tuple) | set(portrait_tuple),
                rem_landscapes=left_spread.rem_landscapes - set(landscape_tuple),
                rem_portraits=left_spread.rem_portraits - set(portrait_tuple),
            ))
        return results

    @classmethod
    def _get_right_pages(cls, n_right_landscapes: int, n_right_portraits: int,
                         left_spreads: List[OrientedSpread]) -> List[OrientedSpread]:
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
            oriented_combs = cls._get_oriented_combs(left_spread.rem_landscapes, left_spread.rem_portraits, n_right_landscapes, n_right_portraits)
            results = cls._get_single_right_page(oriented_combs, left_spread, results)
        return results


    def _process_squares(self, oriented_spreads: List[OrientedSpread], n_left_squares: int, n_right_squares: int,
                         layout_idx: int) -> None:
        """
        Expand oriented spreads into SingleSpreadLayouts by distributing remaining
        photos into square slots on left and right pages.

        For each OrientedSpread, enumerates all ways to assign remaining photos
        to left-page square slots; the rest go to right-page squares. Results are
        appended to self.possible_layouts.

        Args:
            oriented_spreads: List of OrientedSpread objects with pages and
                remaining photos populated.
            n_left_squares: Number of square slots on the left page.
            n_right_squares: Number of square slots on the right page.
            layout_idx: Layout index to assign to each resulting SingleSpreadLayout.
        """
        for spread in oriented_spreads:
            rem_photos = spread.rem_landscapes | spread.rem_portraits
            square_combs = list(combinations(rem_photos, n_left_squares))

            for comb in square_combs:
                self.possible_layouts.append(
                    SingleSpreadLayout(
                        layout_idx=layout_idx,
                        left_page_photo_idxs=spread.left_page_photo_idxs | set(comb),
                        right_page_photo_idxs=spread.right_page_photo_idxs | (rem_photos - set(comb)),
                        number_of_squares=n_left_squares + n_right_squares
                    )
                )

    @classmethod
    def full_oriented_layout_search(cls, landscape_set: Set[int], portrait_set: Set[int], layouts_df: pd.DataFrame, params: SpreadSearchParams) -> SpreadLayoutsList:
        """
        Exhaustive search for spread layouts across all available designs.

        For each layout design, enumerates all valid left/right page assignments
        based on orientation (portrait/landscape), then expands square slots.
        Results are accumulated in self.possible_layouts.

        Args:
            landscape_set: Landscape photo indices in this spread.
            portrait_set: Portrait photo indices in this spread.
            layouts_df: DataFrame of filtered layout designs for this spread size.
            params: Search parameters controlling max oriented combinations.

        Returns:
            self with possible_layouts populated from exhaustive search.
        """
        ll = cls(spread_photo_idxs= landscape_set | portrait_set, possible_layouts=[])

        for layout_idx in layouts_df.index:
            n_left_landscapes = len(layouts_df.at[layout_idx, 'left_landscape_ids'])
            n_left_portraits = len(layouts_df.at[layout_idx, 'left_portrait_ids'])
            n_right_landscapes = len(layouts_df.at[layout_idx, 'right_landscape_ids'])
            n_right_portraits = len(layouts_df.at[layout_idx, 'right_portrait_ids'])
            n_left_squares = len(layouts_df.at[layout_idx, 'left_square_ids'])
            n_right_squares = len(layouts_df.at[layout_idx, 'right_square_ids'])

            left_spreads = cls._get_left_pages(landscape_set, portrait_set, n_left_landscapes, n_left_portraits, params.max_oriented_combs)
            oriented_spreads = cls._get_right_pages(n_right_landscapes, n_right_portraits, left_spreads)

            ll._process_squares(oriented_spreads, n_left_squares, n_right_squares, layout_idx)

        return ll

    def filter_by_score_threshold(self, score_threshold):
        if len(self.possible_layouts) > 0:
            max_score = max(spread.score for spread in self.possible_layouts)
            self.possible_layouts = [spread for spread in self.possible_layouts
                                     if spread.score / max_score > score_threshold]


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


def check_page_properties(photo_set: Set[int], photos: List[Photo]) -> PageProperties:
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
        return PageProperties(True, True, False, 1)

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

    return PageProperties(
        is_same_color=is_same_color,
        is_same_class=is_same_class,
        is_bride_groom_mix=calculate_bride_groom_mix() if is_same_class else False,
        number_of_unique_contexts=number_of_unique_contexts
    )


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


def apply_page_penalties(page_check_result: PageProperties, score: float, penalty: Penalties) -> float:
    """
    Apply multiplicative penalties to a score based on page consistency checks.

    Penalizes mixed colors, mixed classes, bride/groom mixing, and multiple
    original_context values on a single page.

    Args:
        page_check_result: PageProperties for the page being evaluated.
        score: Current score to penalize.
        penalty: Penalty configuration.

    Returns:
        Updated score after applying all applicable penalties.
    """
    if not page_check_result.is_same_color:
        score = score * penalty.color_mix
    if not page_check_result.is_same_class:
        score = score * penalty.class_mix
    if page_check_result.is_bride_groom_mix:
        score = score * penalty.color_mix

    score = score * np.power(penalty.context_mix_penalty, max(1, page_check_result.number_of_unique_contexts) - 1)
    return score


def eval_multi_spreads(group_spreads_layouts: GroupLayoutsLists, layouts_df: pd.DataFrame,
                       photos: List[Photo], penalty: Optional[Penalties] = None) -> GroupLayoutsLists:
    """
    Score and filter all spread layout options for a group.

    For each spread in each combination, computes a multiplicative score based on:
    page color/class consistency, orientation mixing, square-box cropping, and
    photo time ordering. Spreads scoring below score_threshold relative to the
    best spread in their combination are filtered out.

    Args:
        group_spreads_layouts: All layout options for the group's combinations.
        layouts_df: DataFrame of available layouts (used for orientation mixing flags).
        photos: List of Photo objects for the group.
        penalty: Penalty configuration. Uses default Penalties if None.

    Returns:
        The same GroupLayoutsLists with scores set and low-scoring layouts filtered.
    """
    if penalty is None:
        penalty = Penalties()
    #print(f"the CONFIGS['spread_score_threshold'] is {penalty.score_threshold}")

    # Evaluate layouts in all spreads
    for single_spread_layouts in group_spreads_layouts.possible_layouts:
        # Evaluate each spread in this combination
        for spread in single_spread_layouts.possible_layouts:
            spread.evaluate(photos, layouts_df, penalty)

        single_spread_layouts.filter_by_score_threshold(penalty.score_threshold)

    return group_spreads_layouts
