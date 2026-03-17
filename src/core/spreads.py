from __future__ import annotations

import random
from itertools import combinations, product, permutations
from dataclasses import dataclass
from typing import List, Tuple, Set, Iterable, Callable, Any, Optional
import inspect

import numpy as np
import pandas as pd

from src.spreads_layout.math_tools import all_unique_partitions, simple_partitions, partitions_with_swaps
from src.spreads_layout.layouts_tools import (get_layouts_dict, get_spread_layouts_list, filter_layouts, count_squares,
                                              calculate_capacities, apply_layouts_mask)
from src.core.models import SpreadSearchParams
from src.core.photos import group_photos
from utils.configs import CONFIGS


@dataclass
class Partition:
    """
    Represents a partition of photos into spreads with an associated weight.

    A partition defines how a group of photos is divided across album spreads.
    The weight reflects how well the partition matches the expected spread size
    distribution for the photo’s context class (based on a Gaussian score).

    Attributes:
        spread_sizes: Number of photos in each spread.
        weight: Score indicating how well this partition fits the class distribution.
    """
    spread_sizes: List[int]
    weight: float

    def __str__(self) -> str:
        return (f'Partition. {len(self.spread_sizes)} spreads for group, spread sizes: {self.spread_sizes}. ' +
                f'Partition weight: {self.weight}' if self.weight is not None else '')

    @staticmethod
    def class_weight(n_photos: List[int], class_spread_params: List[float]) -> float:
        """
        Calculate the class contribution to the partition score.

        Score is the product of Gaussians with the provided [mean, std] parameters,
        evaluated at each spread’s photo count.

        Args:
            n_photos: Array of photo counts per spread for a specific context class.
            class_spread_params: [mean, std] Gaussian parameters for the context class.

        Returns:
            Product of Gaussian values across all spreads (higher = better fit).
        """
        n_photos = np.array(n_photos)
        n_photos = n_photos[n_photos > 0]
        weight = np.prod(np.exp(-0.5 * np.power(((n_photos - class_spread_params[0]) / class_spread_params[1]), 2)))
        return weight

    @classmethod
    def get_weights_for_parts(cls, parts: List[List[int]], class_spread_params: List[float],
                              n_photos: int) -> np.ndarray:
        """
        Compute weights for all partition candidates.

        If all weights are zero (no partition fits the Gaussian), widens the
        std to allow broader matching. Otherwise normalizes by the max weight.

        Args:
            parts: List of partitions, each a list of spread sizes.
            class_spread_params: [mean, std] Gaussian parameters. May be modified
                in place if all initial weights are zero.
            n_photos: Total number of photos in the group.

        Returns:
            Array of normalized weights, one per partition.
        """
        weights = np.zeros(len(parts))
        for idx, part in enumerate(parts):
            weights[idx] = cls.class_weight(part, class_spread_params)

        if np.all(weights == 0):
            class_spread_params[1] = np.abs(n_photos - class_spread_params[0]) / 3
            for idx, part in enumerate(parts):
                weights[idx] = cls.class_weight(part, class_spread_params)
        else:
            weights /= np.max(weights)
        return weights

    @staticmethod
    def filter_by_layout(parts: List[Partition], layouts_dict: dict,
                                n_portraits: int, n_landscapes: int, params: SpreadSearchParams) -> List[Partition]:
        """
        Filter Partition objects by layout feasibility.

        Checks each partition against available layouts to verify that portrait
        and landscape counts can be accommodated. Applies early stopping when
        enough partitions are found and weight drops below threshold.

        Args:
            parts: List of Partition objects sorted by weight (descending).
            layouts_dict: Dict mapping box count to DataFrame of layout configs.
            n_portraits: Total number of portrait photos.
            n_landscapes: Total number of landscape photos.
            params: Search parameters containing weight_threshold_divisor.

        Returns:
            Filtered list of feasible Partition objects.
        """
        filtered_parts: List[Partition] = []
        weight_threshold = max(p.weight for p in parts) / params.weight_threshold_divisor

        for partition in parts:
            part_landscape = n_landscapes
            part_portrait = n_portraits
            part_layout_matched = True

            for spread in partition.spread_sizes:
                n_layouts = layouts_dict[spread]
                spread_layout_matched = False

                for _, row in n_layouts.iterrows():
                    rem_portrait = max(part_portrait - row['max portraits'], 0)
                    rem_landscape = max(part_landscape - row['max landscapes'], 0)

                    if (part_landscape + part_portrait) - spread >= (rem_portrait + rem_landscape):
                        spread_layout_matched = True
                        part_portrait = rem_portrait
                        part_landscape = rem_landscape
                        break

                if not spread_layout_matched:
                    part_layout_matched = False
                    break

            if part_layout_matched:
                filtered_parts.append(partition)
                # Early stopping if too many parts and weight is below threshold
                if len(filtered_parts) > 2 and partition.weight < weight_threshold:
                    break

        return filtered_parts

    def is_valid(self, min_len: int, n_photos: int) -> bool:
        """
        Check if this Partition meets the selection criteria.

        A partition is valid if its length equals min_len, or if it’s at most
        1 longer, has at most 2 spreads, and the group has fewer than 16 photos.

        Args:
            min_len: Minimum spread count across all candidate partitions.
            n_photos: Total number of photos in the group.

        Returns:
            True if the partition passes the filter.
        """
        part_len = len(self.spread_sizes)
        return (
            (
                part_len - min_len <= 1 and
                part_len <= 2 and
                n_photos < 16
            )
            or (part_len == min_len)
        )

    @classmethod
    def filter_by_len(cls, parts: List[Partition], n_photos: int) -> List[Partition]:
        """
        Filter Partition objects by their length relative to min/max spread sizes.

        Only applies filtering when there is variation in partition lengths.

        Args:
            parts: List of Partition objects.
            n_photos: Total number of photos in the group.

        Returns:
            Filtered list of Partition objects.
        """
        if not parts:
            return parts

        part_len_list = [len(part.spread_sizes) for part in parts]
        min_len, max_len = np.min(part_len_list), np.max(part_len_list)

        # Apply filtering only if there’s variation in lengths
        if max_len > min_len:
            parts = [part for part in parts if part.is_valid(min_len, n_photos)]

        return parts


@dataclass
class Combination:
    """
    A specific assignment of photos to spreads within a Partition.

    Given a Partition that defines how many spreads and their sizes, a Combination
    determines which exact photos go into each spread. The weight reflects the
    quality of this assignment based on temporal coherence and label consistency.

    Attributes:
        spreads: List of sets, each containing photo indices assigned to that spread.
        weight: Quality score for this combination (None until evaluated).
    """
    spreads: List[Set[int]]
    weight: Optional[float] = None

    def __str__(self) -> str:
        return ('Combination. '
                + ', '.join([f'Photos in {i + 1} spread: {spread}' for i, spread in enumerate(self.spreads)])
                + f'. Combination weight: {self.weight}' if self.weight is not None else '')

    def get_evaluation_score(self, photo_times: List[float], cluster_labels: List[int]) -> float:
        """
        Evaluate the quality of this combination based on time and label grouping.

        For each spread, penalizes high time variance (photos far apart in time)
        and duplicate cluster labels (mixed contexts within a spread). The final
        score is the product of penalties across all spreads.

        Args:
            photo_times: List of general_time values indexed by photo index.
            cluster_labels: List of cluster label values indexed by photo index.

        Returns:
            A score where higher values indicate better temporal and contextual
            coherence within spreads.
        """
        score = 1
        for spread in self.spreads:

            spread_times = [photo_times[id] / 60.0 for id in spread]
            spread_labels = [cluster_labels[id] for id in spread]

            time_std = np.std(spread_times)
            if time_std > 0.0001:
                score /= time_std
            if not np.all(np.array(spread_labels) == None):
                score /= (1 + len(spread_labels) - len(set(spread_labels)))
        return score

    def set_weight(self, weight: float) -> None:
        """Set the final weight (evaluation score * partition weight)."""
        self.weight = weight


def selectPartitions(photos_df, classSpreadParams, params, layouts_df):
    # finds all available partitions for a class cluster of size n_photos
    # eliminates unlikely partitions based ont the cluster class score parameters
    # parameter: n_photos - total number of photos for a class cluster
    # parameter: classSpreadParams - array size 2 [mean,std] containing the gaussian parameter for the context class score

    n_photos = len(photos_df.index)
    n_portraits = len(photos_df[photos_df['ar'] < 1].index)
    n_landscapes = n_photos - n_portraits

    available_n = set(layouts_df['number of boxes'].unique())
    layouts_dict = get_layouts_dict(layouts_df, available_n)

    classSpreadParams[1] = max(classSpreadParams[1], 0.5)

    # generate partitions
    ## part values
    parts = all_unique_partitions(n_photos)
    parts = [part for part in parts if set(part).issubset(available_n)]
    ## part weights
    weights = Partition.get_weights_for_parts(parts, classSpreadParams, n_photos)
    parts = [Partition(parts[i], weights[i]) for i in range(len(parts))]
    # process partitions
    sorted_parts = sorted(parts, key=lambda p: p.weight, reverse=True)
    filtered_parts = Partition.filter_by_layout(sorted_parts, layouts_dict, n_portraits, n_landscapes, params)
    valid_parts = Partition.filter_by_len(filtered_parts, n_photos)

    return valid_parts


def listSingleCombinations(photos, layout_part: Partition, maxCombs):
    photos_ids = list(range(len(photos)))
    photos_ids = set(photos_ids)

    if len(layout_part.spread_sizes) > 1:
        layout_combs = partitions_with_swaps(list(photos_ids), layout_part.spread_sizes, 2)
        layout_combs = [Combination([set(part) for part in comb]) for comb, v in layout_combs]
    else:
        layout_combs = simple_partitions(photos_ids, layout_part.spread_sizes, maxCombs)
        layout_combs = [Combination(comb_list) for comb_list in layout_combs]
    return layout_combs


def _get_portraits_landscapes(photos):
    photos_ids = list(range(len(photos)))

    # n_photos = len(photos)
    # landscapes = 0
    landscape_photos_ids = list()
    portrait_photos_ids = list()

    for i in range(len(photos)):
        if photos[i].ar < 1:
            portrait_photos_ids.append(photos_ids[i])
        else:
            # landscapes += 1
            landscape_photos_ids.append(photos_ids[i])
    # portraits = n_photos - landscapes
    context2photos_number = dict()
    for photo in photos:
        if photo.original_context not in context2photos_number:
            context2photos_number[photo.original_context] = list()
        context2photos_number[photo.original_context].append(photo.general_time)
    landscape_photos_ids = sorted(landscape_photos_ids,
                                  key=lambda x: [np.mean(context2photos_number[photos[x].original_context]),
                                                 photos[x].general_time])
    portrait_photos_ids = sorted(portrait_photos_ids,
                                 key=lambda x: [np.mean(context2photos_number[photos[x].original_context]),
                                                photos[x].general_time])
    return portrait_photos_ids, landscape_photos_ids


def _prepare_all_combinations(spread_layouts_list):
    if len(spread_layouts_list) >= 4:
        all_combinations_of_layouts = [[sublist[0] for sublist in spread_layouts_list]]
    else:
        all_combinations_of_layouts = list(product(*spread_layouts_list))

    if len(all_combinations_of_layouts) > 1000:
        all_combinations_of_layouts = random.sample(all_combinations_of_layouts, 1000)

    return all_combinations_of_layouts


def _get_final_combinations(all_combinations_of_layouts, photos, portrait_photos_ids, landscape_photos_ids):
    n_photos = len(photos)
    portraits, landscapes = len(portrait_photos_ids), len(landscape_photos_ids)

    final_layout_combs_list = list()
    for layouts_comb in all_combinations_of_layouts:
        total_number_of_boxes = sum([int(cur_layout['number of boxes'].iloc[0]) for cur_layout in layouts_comb])
        total_number_of_portraits = sum(
            [len(cur_layout['left_portrait_ids'].iloc[0]) + len(cur_layout['right_portrait_ids'].iloc[0])
             for cur_layout in layouts_comb]
        )
        total_number_of_landscapes = sum(
            [len(cur_layout['left_landscape_ids'].iloc[0]) + len(cur_layout['right_landscape_ids'].iloc[0])
             for cur_layout in layouts_comb]
        )
        if total_number_of_boxes != n_photos or total_number_of_portraits > portraits or total_number_of_landscapes > landscapes:
            continue

        cur_comb = [set() for _ in range(len(layouts_comb))]
        portraits_idx = 0
        landscapes_idx = 0
        for cur_idx, cur_layout in enumerate(layouts_comb):
            for _ in range(len(cur_layout['left_portrait_ids'].iloc[0]) + len(cur_layout['right_portrait_ids'].iloc[0])):
                cur_comb[cur_idx].add(portrait_photos_ids[portraits_idx])
                portraits_idx += 1
        for cur_idx, cur_layout in enumerate(layouts_comb):
            for _ in range(len(cur_layout['left_landscape_ids'].iloc[0]) + len(cur_layout['right_landscape_ids'].iloc[0])):
                cur_comb[cur_idx].add(landscape_photos_ids[landscapes_idx])
                landscapes_idx += 1

        # add squares
        for cur_idx, cur_layout in enumerate(layouts_comb):
            while len(cur_comb[cur_idx]) < int(cur_layout['number of boxes'].iloc[0]):
                if portraits_idx == len(portrait_photos_ids) and landscapes_idx == len(landscape_photos_ids):
                    raise Exception('Something wrong. Not enough photos in greedy layouts search.')
                elif portraits_idx == len(portrait_photos_ids):
                    cur_comb[cur_idx].add(landscape_photos_ids[landscapes_idx])
                    landscapes_idx += 1
                elif landscapes_idx == len(landscape_photos_ids):
                    cur_comb[cur_idx].add(portrait_photos_ids[portraits_idx])
                    portraits_idx += 1
                else:
                    next_portrait = photos[portrait_photos_ids[portraits_idx]].general_time
                    next_landscape = photos[landscape_photos_ids[landscapes_idx]].general_time
                    if next_portrait < next_landscape:
                        cur_comb[cur_idx].add(portrait_photos_ids[portraits_idx])
                        portraits_idx += 1
                    else:
                        cur_comb[cur_idx].add(landscape_photos_ids[landscapes_idx])
                        landscapes_idx += 1
        final_layout_combs_list.append(cur_comb)
    return final_layout_combs_list


def _filter_combinations(final_layout_combs_list):
    cleaned_comb_data = []
    seen = set()

    for inner_list in final_layout_combs_list:
        frozen_inner_list = tuple(frozenset(s) for s in inner_list)

        if frozen_inner_list not in seen:
            seen.add(frozen_inner_list)
            cleaned_comb_data.append(inner_list)
    return cleaned_comb_data


def greedy_combination_search(photos, layout_part, layout_df):
    portrait_photos_ids, landscape_photos_ids = _get_portraits_landscapes(photos)

    spread_layouts_list = get_spread_layouts_list(layout_df, layout_part.spread_sizes)

    all_combinations_of_layouts = _prepare_all_combinations(spread_layouts_list)

    final_layout_combs_list = _get_final_combinations(all_combinations_of_layouts, photos,
                                                      portrait_photos_ids, landscape_photos_ids)

    cleaned_comb_data = _filter_combinations(final_layout_combs_list)
    return [Combination(comb_list) for comb_list in cleaned_comb_data]


@dataclass
class SingleSpreadLayout:
    layout_idx: int
    left_page_photo_idxs: Set[int]      # photo index in local list of photos per group
    right_page_photo_idxs: Set[int]     # (not photo ID)
    number_of_squares: int
    score: float = None
    left_page_photos: Set | List = None   # set of Photos after resolve_photos,
    right_page_photos: Set | List = None  # list after set_photos_order

    def __str__(self):
        return (f'Layout_idx: {self.layout_idx}; '
                f'Photos: [left page: {self.left_page_photo_idxs}, right page: {self.right_page_photo_idxs}]; '
                f'Square boxes in spread: {self.number_of_squares}')

    def resolve_photos(self, group_photos: List):
        self.left_page_photos = {group_photos[idx] for idx in self.left_page_photo_idxs}
        self.right_page_photos = {group_photos[idx] for idx in self.right_page_photo_idxs}

    def set_photos_order(self, left_ordered: List, right_ordered: List):
        self.left_page_photos = left_ordered
        self.right_page_photos = right_ordered


@dataclass
class SpreadLayoutsList:
    spread_photo_idxs: Set[int]
    possible_layouts: List[SingleSpreadLayout] = None

    def view(self, limit=None, sep='=='):
        print('Possible layouts for spread with photos:', self.spread_photo_idxs, f'- {len(self.possible_layouts)} options')
        for j, sp in enumerate(self.possible_layouts):
            if limit is not None and j > limit:
                print(sep, '... ... ...')
                break
            print(sep, j + 1, sp)

    def update_layouts(self, layouts_list: List[SingleSpreadLayout]):
        self.possible_layouts = layouts_list


class GroupLayoutsLists(Combination):
    '''
    The object of this class represents possible layout options for a certain Combination.
    '''
    def __init__(self, spreads: List[Set[int]], weight: float):
        super().__init__(spreads=spreads, weight=weight)
        self.possible_layouts: List[SpreadLayoutsList] = []

    @classmethod
    def from_comb(cls, comb: Combination) -> GroupLayoutsLists:
        """
        Create a GroupLayoutsLists from an existing Combination.
        """
        return cls(comb.spreads, comb.weight)

    def view(self, limit = None, sep='=='):
        print(f'Layout options for {len(self.spreads)}-spread group: {self.spreads}')
        for i in range(len(self.spreads)):
            print(sep, i + 1, end = ' ')
            self.possible_layouts[i].view(limit=limit, sep = sep*2)

    def add_spread(self, layouts: SpreadLayoutsList):
        self.possible_layouts.append(layouts)


def _get_portraits_landscapes_for_spread(spread_photos, photos):
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


def _simple_layout(layouts_df, n_photos):
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


def _process_with_time(spread_photos, photos, greedy_layouts, greedy_single_spreads):
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


def _process_with_color(spread_photos, photos, greedy_layouts, greedy_single_spreads):
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


def _get_left_pages(oriented_combs, landscape_set, portrait_set, rem_landscapes, rem_portraits):
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


def _get_single_right_page(oriented_combs, rem_right_landscapes, rem_right_portraits, rem_landscapes, rem_portraits,
                           idx, left_set, oriented_spreads):
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


def _get_right_pages(right_landscapes, right_portraits, rem_landscapes, rem_portraits, left_pages):
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


def _expand_single_spreads(oriented_spreads, rem_right_landscapes, rem_right_portraits, left_squares, right_squares,
                           layout, single_spreads):
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


def _get_spreads(layouts, landscape_set, portrait_set, params, greedy_single_spreads):
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


def layoutSingleCombination(single_class_comb: Combination, layout_df, photos, params):
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


def check_page(photo_set, photos):
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


def apply_page_penalties(page_check_result, score, penalty):
    if not page_check_result.is_same_color:
        score = score * penalty.color_mix
    if not page_check_result.is_same_class:
        score = score * penalty.class_mix
    if page_check_result.is_bride_groom_mix:
        score = score * penalty.color_mix

    score = score * np.power(penalty.context_mix_penalty, max(1, page_check_result.number_of_unique_contexts) - 1)
    return score


def eval_multi_spreads(group_spreads_layouts: GroupLayoutsLists, layouts_df, photos, penalty = None):
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

    def update_score(self, factor: float):
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


def _get_combinations(partitions, photos, layouts_df, spread_params, params):
    combs = []

    photoTimes = [item.general_time for item in photos]
    cluster_labels = [item.cluster_label for item in photos]

    def eval_combination(combination, partition):
        combination_weight = combination.get_evaluation_score(photoTimes, cluster_labels)
        combination.set_weight(combination_weight * partition.weight)

    maxCombsParam = params.max_spreads_sample if len(photos) <= params.small_group_threshold else params.max_combs_small_group

    for i, partition in enumerate(partitions):
        # print(partition)
        maxCombs = int(maxCombsParam / np.power(2, i))

        if len(photos) <= 8 and len(photos) / spread_params[0] <= 2:
            single_combs = listSingleCombinations(photos, partition, maxCombs)
        else:
            single_combs = greedy_combination_search(photos, partition, layouts_df)
        # print(f"Single Combinations {len(single_combs)} and maxCombs {maxCombs}")

        if len(single_combs) > maxCombs:
            #logger.info('combinations Found {}, sampled {} combinations foe evaluation'.format(len(single_combs), maxCombs))
            sample_idxs = random.sample(range(len(single_combs)), maxCombs)
            single_combs = [single_combs[sample_idx] for sample_idx in sample_idxs]

        for comb in single_combs:
            eval_combination(comb, partition)
            # print(comb)
        combs += single_combs

    return combs


def generate_filtered_multi_spreads(photos, layouts_df, spread_params, params, logger):
    photos_df = pd.DataFrame([photo.__dict__ for photo in photos])
    photos_df = photos_df.sort_values('general_time')
    partitions = selectPartitions(photos_df, spread_params, params, layouts_df=layouts_df)
    # logger.info('Number of photos: {}. Possible partitions: {}'.format(len(photos), layout_parts))

    combs = _get_combinations(partitions, photos, layouts_df, spread_params, params)

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

