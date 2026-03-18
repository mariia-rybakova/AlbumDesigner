from __future__ import annotations

import random
from itertools import combinations, product, permutations
from dataclasses import dataclass
from typing import List, Tuple, Set, Iterable, Callable, Any, Optional
import inspect

import numpy as np
import pandas as pd

from src.spreads_layout.layouts_tools import (filter_layouts, count_squares,
                                              calculate_capacities, apply_layouts_mask)
from src.spreads_layout.partitions import Partition, get_partitions
from src.spreads_layout.combinations import Combination, get_combinations
from src.core.models import SpreadSearchParams
from src.core.photos import group_photos
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


def _get_portraits_landscapes_for_spread(spread_photos: List[int], photos: List) -> Tuple[int, int, Set[int], Set[int]]:
    '''
    Read orientation from photos in group
    '''
    landscape_set = set()
    portrait_set = set()

    n_photos = len(spread_photos)
    landscapes = 0

    for i in range(len(spread_photos)):
        if photos[spread_photos[i]].ar < 1:
            portrait_set.add(spread_photos[i])
        else:
            landscapes += 1
            landscape_set.add(spread_photos[i])

    portraits = n_photos - landscapes
    return portraits, landscapes, portrait_set, landscape_set


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


def _process_with_time(spread_photos: List[int], photos: List, greedy_layouts: pd.DataFrame, greedy_single_spreads: List[SingleSpreadLayout]) -> List[SingleSpreadLayout]:
    grouped_sequences = group_photos(spread_photos, photos)

    if len(grouped_sequences) == 2:
        left_landscapes = np.sum([photos[item[0]].ar > 1 for item in grouped_sequences[0]])
        left_portraits = len(grouped_sequences[0]) - left_landscapes
        right_landscapes = np.sum([photos[item[0]].ar > 1 for item in grouped_sequences[1]])
        right_portraits = len(grouped_sequences[1]) - right_landscapes

        possible_layouts = apply_layouts_mask(greedy_layouts, left_landscapes, left_portraits, right_landscapes,
                                       right_portraits)

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


def _process_with_color(spread_photos: List[int], photos: List, greedy_layouts: pd.DataFrame, greedy_single_spreads: List[SingleSpreadLayout]) -> List[SingleSpreadLayout]:
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


def _get_left_pages(oriented_combs: List[Tuple], landscape_set: Set[int], portrait_set: Set[int], rem_landscapes: List[Set[int]], rem_portraits: List[Set[int]]) -> Tuple[List[Set[int]], List[Set[int]], List[Set[int]]]:
    left_pages = list()
    for comb in oriented_combs:
        single_left = set(comb[0])

        if len(comb[0]) == 0:
            rem_landscapes.append(landscape_set)
        else:
            rem_landscapes.append(landscape_set - set(comb[0]))

        for portrait in comb[1]:
            single_left.add(portrait)

        if len(comb[1]) == 0:
            rem_portraits.append(portrait_set)
        else:
            rem_portraits.append(portrait_set - set(comb[1]))
        left_pages.append(single_left)
    return rem_landscapes, rem_portraits, left_pages


def _get_single_right_page(oriented_combs: List[Tuple], rem_right_landscapes: List[Set[int]], rem_right_portraits: List[Set[int]], rem_landscapes: List[Set[int]], rem_portraits: List[Set[int]],
                           idx: int, left_set: Set[int], oriented_spreads: List[List[Set[int]]]) -> Tuple[List[List[Set[int]]], List[Set[int]], List[Set[int]]]:
    for comb in oriented_combs:
        single_right = set(comb[0])

        if len(comb[0]) == 0:
            rem_right_landscapes.append(rem_landscapes[idx])
        else:
            rem_right_landscapes.append(rem_landscapes[idx] - set(comb[0]))

        for portrait in comb[1]:
            single_right.add(portrait)

        if len(comb[1]) == 0:
            rem_right_portraits.append(rem_portraits[idx])
        else:
            rem_right_portraits.append(rem_portraits[idx] - set(comb[1]))
        oriented_spreads.append([left_set, single_right])
    return oriented_spreads, rem_right_landscapes, rem_right_portraits


def _get_right_pages(right_landscapes: int, right_portraits: int, rem_landscapes: List[Set[int]], rem_portraits: List[Set[int]], left_pages: List[Set[int]]) -> Tuple[List[List[Set[int]]], List[Set[int]], List[Set[int]]]:
    oriented_spreads = []
    rem_right_landscapes = []
    rem_right_portraits = []
    for idx, left_set in enumerate(left_pages):
        landscape_combs = list(combinations(rem_landscapes[idx], right_landscapes))
        portrait_combs = list(combinations(rem_portraits[idx], right_portraits))
        oriented_combs = list(product(landscape_combs, portrait_combs))

        oriented_spreads, rem_right_landscapes, rem_right_portraits = _get_single_right_page(
            oriented_combs, rem_right_landscapes, rem_right_portraits, rem_landscapes, rem_portraits,
            idx, left_set, oriented_spreads
        )
    return oriented_spreads, rem_right_landscapes, rem_right_portraits


def _expand_single_spreads(oriented_spreads: List[List[Set[int]]], rem_right_landscapes: List[Set[int]], rem_right_portraits: List[Set[int]], left_squares: int, right_squares: int,
                           layout: int, single_spreads: List[SingleSpreadLayout]) -> List[SingleSpreadLayout]:
    for idx, oriented_spread in enumerate(oriented_spreads):
        rem_photos = rem_right_landscapes[idx].union(rem_right_portraits[idx])
        landscape_left_combs = list(combinations(rem_photos, left_squares))
        for comb in landscape_left_combs:
            single_spreads.append(
                SingleSpreadLayout(
                    layout_idx=layout,
                    left_page_photo_idxs= oriented_spread[0].union(set(comb)),
                    right_page_photo_idxs=oriented_spread[1].union(rem_photos) - set(comb),
                    number_of_squares=left_squares + right_squares
                )
            )
    return single_spreads


def _get_spreads(layouts: pd.DataFrame, landscape_set: Set[int], portrait_set: Set[int], params: SpreadSearchParams, greedy_single_spreads: List[SingleSpreadLayout]) -> List[SingleSpreadLayout]:
    spreads = []
    for layout in layouts.index:
        left_landscapes = len(layouts.at[layout, 'left_landscape_ids'])
        left_portraits = len(layouts.at[layout, 'left_portrait_ids'])
        landscape_combs = list(combinations(landscape_set, left_landscapes))
        portrait_combs = list(combinations(portrait_set, left_portraits))
        oriented_combs = list(product(landscape_combs, portrait_combs))
        rem_landscapes = []
        rem_portraits = []
        # print(f"CONFIGS['MaxOrientedCombs'] is {CONFIGS['MaxOrientedCombs']}")
        # if len(oriented_combs) > CONFIGS['MaxOrientedCombs']:
        if len(oriented_combs) > params.max_oriented_combs:
            # print('MaxOrientedCombs crossed sampling oriented combinations instead of full listing')
            # sample_idxs = random.sample(range(len(oriented_combs)), CONFIGS['MaxOrientedCombs'])
            sample_idxs = random.sample(range(len(oriented_combs)), params.max_oriented_combs)
            oriented_combs = [oriented_combs[i] for i in sample_idxs]

        rem_landscapes, rem_portraits, left_pages = _get_left_pages(
            oriented_combs,
            landscape_set, portrait_set,
            rem_landscapes, rem_portraits,
        )

        right_landscapes = len(layouts.at[layout, 'right_landscape_ids'])
        right_portraits = len(layouts.at[layout, 'right_portrait_ids'])

        oriented_spreads, rem_right_landscapes, rem_right_portraits = _get_right_pages(
            right_landscapes, right_portraits, rem_landscapes, rem_portraits, left_pages
        )

        left_squares = len(layouts.at[layout, 'left_square_ids'])
        right_squares = len(layouts.at[layout, 'right_square_ids'])

        if len(oriented_spreads) != len(rem_right_landscapes):
            rem_right_landscapes # ToDo ???

        # single_spreads = []
        single_spreads = greedy_single_spreads.copy()
        single_spreads = _expand_single_spreads(oriented_spreads, rem_right_landscapes, rem_right_portraits,
                                                left_squares, right_squares,
                                                layout, single_spreads)

        spreads += single_spreads
    return spreads


def layoutSingleCombination(single_class_comb: Combination, layout_df: pd.DataFrame, photos: List, params: SpreadSearchParams) -> Optional[GroupLayoutsLists]:
    n_spreads = len(single_class_comb.spreads)
    group_spreads_layouts = GroupLayoutsLists.from_comb(single_class_comb)

    for photo_idx_set in single_class_comb.spreads:
        spread_photos = list(photo_idx_set)

        if len(spread_photos) == 0:
            spread_photos # ToDo ???

        n_photos_in_spread = len(spread_photos)
        portraits, landscapes, portrait_set, landscape_set = _get_portraits_landscapes_for_spread(spread_photos, photos)

        layouts = filter_layouts(layout_df, n_photos_in_spread, portraits, landscapes)
        layouts = count_squares(layouts)

        # large spreads with squares gets trivial layout
        if (
                n_photos_in_spread > 13 and
                len(layouts[layouts['number of squares'] == n_photos_in_spread]) > 0 and
                n_spreads == 1
            ):
            single_spreads = _simple_layout(layouts, n_photos_in_spread)
            single_spread_layouts = SpreadLayoutsList(photo_idx_set, single_spreads)
            group_spreads_layouts.add_spread(single_spread_layouts)
            return group_spreads_layouts

        ### greedy attempt to find layout based on separation of time, class and color
        greedy_single_spreads = []
        try:
            if len(layouts) > 0:
                greedy_layouts = calculate_capacities(layouts)

                greedy_single_spreads = _process_with_time(spread_photos, photos, greedy_layouts, greedy_single_spreads)
                greedy_single_spreads = _process_with_color(spread_photos, photos, greedy_layouts, greedy_single_spreads)
        except Exception as e:
            print(f"Greedy layout attempt failed with error {e}")

        spreads = _get_spreads(layouts, landscape_set, portrait_set, params, greedy_single_spreads)

        if len(spreads) == 0:
            return None
        if len(spreads) > params.max_spreads_sample:
            # print(f"Sampling {params.max_spreads_sample} spreads from {len(spreads)}")
            sample_idxs = random.sample(range(len(spreads)), params.max_spreads_sample)
            spreads = [spreads[i] for i in sample_idxs]

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
        multispread_layouts = layoutSingleCombination(comb, layouts_df, photos, params)
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

