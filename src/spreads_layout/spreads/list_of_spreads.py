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
from src.spreads_layout.spreads.spread import SingleSpreadLayout, Penalties


@dataclass
class SpreadLayoutsList:
    spread_photo_idxs: Set[int]
    possible_layouts: Optional[List[SingleSpreadLayout]] = None

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
        ll = cls(spread_photo_idxs=spread_photo_idxs, possible_layouts=[])

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

    def _context_page_separation(self, photos: List[Photo], greedy_layouts: pd.DataFrame) -> None:
        """
        Try to split photos across left/right pages by time-based context groups.

        Groups photos by (original_context, color) using group_photos. If exactly
        two groups are found, treats them as left and right page candidates, finds
        compatible layouts based on orientation counts, and appends matching
        SingleSpreadLayouts to self.possible_layouts.

        Args:
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

    def _color_page_separation(self, photos: List[Photo], greedy_layouts: pd.DataFrame) -> None:
        """
        Try to split photos across left/right pages by color vs grayscale.

        When the spread contains both color and grayscale photos, assigns the
        group whose mean time is earlier to the left page (color-before-gray or
        vice versa). Finds compatible layouts based on per-page orientation counts
        and appends matching SingleSpreadLayouts to self.possible_layouts.

        Args:
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
    def greedy_layout_search(cls, spread_photos: Set[int], photos: List[Photo], layouts_df: pd.DataFrame) -> SpreadLayoutsList:
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
            New SpreadLayoutsList with possible_layouts populated by greedy heuristics.
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
            New SpreadLayoutsList with possible_layouts populated from exhaustive search.
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

    def filter_by_score_threshold(self, score_threshold: float) -> None:
        """Filter out layouts scoring below score_threshold relative to the best."""
        if len(self.possible_layouts) > 0:
            max_score = max(spread.score for spread in self.possible_layouts)
            self.possible_layouts = [spread for spread in self.possible_layouts
                                     if spread.score / max_score > score_threshold]

    def process(self, photos: List[Photo], layouts_df: pd.DataFrame, penalty: Penalties) -> None:
        """
        Score all spread layouts and filter out low-scoring ones.

        Args:
            photos: List of Photo objects for the group.
            layouts_df: DataFrame of available layouts.
            penalty: Penalty configuration for scoring.
        """
        # evaluate
        for spread_layout in self.possible_layouts:
            spread_layout.set_weight(spread_layout.get_score(photos, layouts_df, penalty))

        # filter
        self.filter_by_score_threshold(penalty.score_threshold)


def sample_layouts(photo_idx_set: Set[int], n_spreads: int, photos: List[Photo],
                   layouts_df: pd.DataFrame, params: SpreadSearchParams) -> Optional[SpreadLayoutsList]:
    """
    Find all possible spread layouts for a single set of photos.

    Filters layouts by photo count and orientation, then tries a fast path
    for large all-square spreads, greedy heuristics (context/color separation),
    and exhaustive oriented search. Combines and samples results.

    Args:
        photo_idx_set: Set of photo indices for this spread.
        n_spreads: Total number of spreads in the parent combination.
        photos: Full list of Photo objects for the group.
        layouts_df: DataFrame of available layout designs.
        params: Search parameters controlling sampling limits.

    Returns:
        SpreadLayoutsList with candidate layouts, or None if no valid layout exists.
    """
    if len(photo_idx_set) == 0:
        return None

    n_photos_in_spread = len(photo_idx_set)
    portrait_set, landscape_set = get_portraits_landscapes(photo_idx_set, photos)

    layouts_df = filter_layouts(layouts_df, n_photos_in_spread, len(portrait_set), len(landscape_set))
    layouts_df = count_squares(layouts_df)

    # large spreads with squares gets trivial layout
    if is_large_spread_with_squares(n_photos_in_spread, n_spreads, layouts_df):
        return SpreadLayoutsList.simple_layout(photo_idx_set, layouts_df)

    # greedy attempt to find layout based on separation of time, class and color
    greedy_spreads = SpreadLayoutsList.greedy_layout_search(photo_idx_set, photos, layouts_df)
    # other layouts sampling
    oriented_spreads = SpreadLayoutsList.full_oriented_layout_search(landscape_set, portrait_set, layouts_df, params)

    greedy_spreads_l = limit_sample_size(greedy_spreads.possible_layouts, params.max_spreads_sample)
    oriented_spreads_l = limit_sample_size(oriented_spreads.possible_layouts,
                                           params.max_spreads_sample - len(greedy_spreads_l))

    spread_layouts = SpreadLayoutsList(photo_idx_set, greedy_spreads_l + oriented_spreads_l)

    if len(spread_layouts.possible_layouts) == 0:
        return None

    return spread_layouts
