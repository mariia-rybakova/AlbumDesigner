from __future__ import annotations

import random
from itertools import combinations, product, groupby, permutations
from dataclasses import dataclass
from typing import List, Tuple, Set, Iterable, Callable, Any, Optional
import inspect

import numpy as np
import pandas as pd

from utils.configs import CONFIGS


@dataclass
class Partition:
    spread_sizes: List[int]
    weight: float

    def __str__(self):
        return (f'Partition. {len(self.spread_sizes)} spreads for group, spread sizes: {self.spread_sizes}. ' +
                f'Partition weight: {self.weight}' if self.weight is not None else '')

    @staticmethod
    def classWeight(nPhotos, classSpredParams):
        # calculates the class contribution to score.
        # score is gaussian with provided array of [mean,std]
        # input parameter nPhotos is an array of number of photos for all spreads for specific context class
        # the result classWeight is the product of all gaussians for the context class

        nPhotos = np.array(nPhotos)
        nPhotos = nPhotos[nPhotos > 0]
        classWeight = np.prod(np.exp(-0.5 * np.power(((nPhotos - classSpredParams[0]) / classSpredParams[1]), 2)))
        return classWeight

    @classmethod
    def get_weights_for_parts(cls, parts: List[List[int]], classSpreadParams, nPhotos):
        weights = np.zeros(len(parts))
        for idx, part in enumerate(parts):
            weights[idx] = cls.classWeight(part, classSpreadParams)

        if np.all(weights == 0):
            classSpreadParams[1] = np.abs(nPhotos - classSpreadParams[0]) / 3
            for idx, part in enumerate(parts):
                weights[idx] = cls.classWeight(part, classSpreadParams)
        else:
            weights /= np.max(weights)
        return weights

    @staticmethod
    def filter_by_layout(parts: List[Partition], layouts_dict: dict,
                                n_portraits: int, n_landscapes: int, params: Any) -> List[Partition]:
        """
        Filter Partition objects by layout feasibility.
        Each Partition already has its weight set.
        Returns a filtered list of Partition objects.
        """
        filtered_parts: List[Partition] = []
        weight_threshold = max(p.weight for p in parts) / params[1]

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
        Returns a filtered list of Partition objects.
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
    spreads: List[Set[int]]
    weight: float = None

    def __str__(self):
        return ('Combination. '
                + ', '.join([f'Photos in {i + 1} spread: {spread}' for i, spread in enumerate(self.spreads)])
                + f'. Combination weight: {self.weight}' if self.weight is not None else '')

    def eval_single_comb(self, photo_times, cluster_labels):
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

    def set_weight(self, weight):
        self.weight = weight


def printAllUniqueParts(n):
    p = [0] * n  # An array to store a partition
    k = 0  # Index of last element in a partition
    p[k] = n  # Initialize first partition
    # as number itself

    # This loop first prints current partition,
    # then generates next partition.The loop
    # stops when the current partition has all 1s

    parts = []

    while True:

        parts.append(p[:k + 1].copy())
        # Generate next partition

        # Find the rightmost non-one value in p[].
        # Also, update the rem_val so that we know
        # how much value can be accommodated
        rem_val = 0
        while k >= 0 and p[k] == 1:
            rem_val += p[k]
            k -= 1

        # if k < 0, all the values are 1 so
        # there are no more partitions
        if k < 0:
            return parts

        # Decrease the p[k] found above
        # and adjust the rem_val
        p[k] -= 1
        rem_val += 1

        # If rem_val is more, then the sorted
        # order is violated. Divide rem_val in
        # different values of size p[k] and copy
        # these values at different positions after p[k]
        while rem_val > p[k]:
            p[k + 1] = p[k]
            rem_val = rem_val - p[k]
            k += 1

        # Copy rem_val to next position
        # and increment position
        p[k + 1] = rem_val
        k += 1


def _get_layouts_dict(layouts_df, available_n):
    layouts_dict = dict()
    for item in list(available_n):
        layouts_dict[item] = layouts_df[layouts_df['number of boxes'] == item][['max portraits', 'max landscapes']].drop_duplicates()
    return layouts_dict


def selectPartitions(photos_df, classSpreadParams, params, layouts_df):
    # finds all available partitions for a class cluster of size n_photos
    # eliminates unlikely partitions based ont the cluster class score parameters
    # parameter: n_photos - total number of photos for a class cluster
    # parameter: classSpreadParams - array size 2 [mean,std] containing the gaussian parameter for the context class score

    n_photos = len(photos_df.index)
    n_portraits = len(photos_df[photos_df['ar'] < 1].index)
    n_landscapes = n_photos - n_portraits

    available_n = set(layouts_df['number of boxes'].unique())
    layouts_dict = _get_layouts_dict(layouts_df, available_n)

    classSpreadParams[1] = max(classSpreadParams[1], 0.5)

    # generate partitions
    ## part values
    parts = printAllUniqueParts(n_photos)
    parts = [part for part in parts if set(part).issubset(available_n)]
    ## part weights
    weights = Partition.get_weights_for_parts(parts, classSpreadParams, n_photos)
    parts = [Partition(parts[i], weights[i]) for i in range(len(parts))]
    # process partitions
    sorted_parts = sorted(parts, key=lambda p: p.weight, reverse=True)
    filtered_parts = Partition.filter_by_layout(sorted_parts, layouts_dict, n_portraits, n_landscapes, params)
    valid_parts = Partition.filter_by_len(filtered_parts, n_photos)

    return valid_parts


def partitions_with_swaps(seq, sizes, m):
    """
    Generate all partitions of `seq` into len(sizes) groups of given sizes.
    Order inside a group doesn't affect the cost.
    Cost = minimal #adjacent swaps needed to restore the default consecutive split,
           which equals the number of inversions between group labels along the
           original index order.

    Returns: list of (groups, swaps) where groups is a list of lists of elements.
    """
    n = len(seq)
    assert sum(sizes) == n, "sizes must sum to len(seq)"
    G = len(sizes)
    indices = tuple(range(n))

    # assignment over original positions: -1 = unassigned, else group id 0..G-1
    assign = [-1] * n
    assigned_positions = []  # list of positions already assigned
    results = []

    def add_inversions_for_new(pos, g):
        """Count inversions introduced by assigning position `pos` -> group `g`,
        against all previously assigned positions."""
        inc = 0
        for j in assigned_positions:
            gj = assign[j]
            if j < pos and gj > g:
                inc += 1
            elif j > pos and g > gj:
                inc += 1
        return inc

    def backtrack(group_id, remaining_idx_set, swaps_so_far, groups_idx):
        if swaps_so_far > m:
            return
        if group_id == G:
            # Build concrete groups (keep each group's indices in ascending original order)
            groups = []
            for gi, idxs in enumerate(groups_idx):
                groups.append([seq[i] for i in sorted(idxs)])
            results.append((groups, swaps_so_far))
            return

        s = sizes[group_id]
        rem_list = sorted(remaining_idx_set)
        # choose s indices (as a set) for this group
        for chosen in combinations(rem_list, s):
            # assign them (order within chosen doesn't change cost since same label)
            inc = 0
            # assign one-by-one so we can update swaps incrementally
            for pos in chosen:
                assign[pos] = group_id
                inc += add_inversions_for_new(pos, group_id)
                assigned_positions.append(pos)

            backtrack(
                group_id + 1,
                remaining_idx_set - set(chosen),
                swaps_so_far + inc,
                groups_idx + [chosen],
            )

            # undo
            for pos in chosen:
                assigned_positions.pop()  # last appended
                assign[pos] = -1

    backtrack(0, set(indices), 0, [])
    # sort nicely (by swaps, then lexicographically)
    results.sort(key=lambda x: (x[1], x[0]))
    return results


def simple_partitions(photos_ids, layout_part, maxCombs):
    l0_combs = list(combinations(photos_ids, layout_part[0]))

    if len(l0_combs) > maxCombs:
        sample_idxs = random.sample(range(len(l0_combs)), maxCombs)
        l0_combs = [l0_combs[i] for i in sample_idxs]

    l0_combs = [[set(l0_comb)] for l0_comb in l0_combs]
    rem_photos = [photos_ids - l0_comb[0] for l0_comb in l0_combs]
    layout_combs = l0_combs

    for layout_index in range(1, len(layout_part) - 1):
        merged_combs = []
        merged_rem_photos = []
        for comb_idx in range(len(layout_combs)):
            next_combs = list(combinations(rem_photos[comb_idx], layout_part[layout_index]))
            next_combs = [set(next_comb) for next_comb in next_combs]
            if len(layout_combs) > maxCombs:
                next_combs = [next_combs[0]]
            single_comb = [layout_combs[comb_idx].copy() for _ in range(len(next_combs))]
            single_rem_photos = [rem_photos[comb_idx].copy() for _ in range(len(next_combs))]
            for single_idx in range(len(single_comb)):
                single_comb[single_idx].append(next_combs[single_idx])
                single_rem_photos[single_idx] = single_rem_photos[single_idx] - next_combs[single_idx]
            merged_combs += single_comb
            merged_rem_photos += single_rem_photos
        layout_combs = merged_combs
        rem_photos = merged_rem_photos

    if len(layout_part) > 1:
        if len(layout_combs) > maxCombs:
            # print(f"Sampling {maxCombs} combinations from {len(layout_combs)}")
            sample_idxs = random.sample(range(len(layout_combs)), maxCombs)
            layout_combs = [layout_combs[i] for i in sample_idxs]
            # rem_photos = [rem_photos[i] for i in sample_idxs]
        for comb_idx in range(len(layout_combs)):
            layout_combs[comb_idx].append(rem_photos[comb_idx])
    return layout_combs


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


def _get_spread_layouts_list(layout_df, layout_part):
    spread_layouts_list = list()
    for spread_size in layout_part:
        layouts = layout_df.loc[(layout_df['number of boxes'] == spread_size)]
        # &
        # (len(layout_df['left_portrait_ids']) + len(layout_df['right_portrait_ids']) <= portraits) &
        # (len(layout_df['left_landscape_ids']) + len(layout_df['right_landscape_ids']) <= landscapes)
        list_of_single_row_layouts = []
        for index, row in layouts.iterrows():
            single_row_df = row.to_frame().T
            list_of_single_row_layouts.append(single_row_df)
        spread_layouts_list.append(list_of_single_row_layouts)
    return spread_layouts_list


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

    spread_layouts_list = _get_spread_layouts_list(layout_df, layout_part.spread_sizes)

    all_combinations_of_layouts = _prepare_all_combinations(spread_layouts_list)

    final_layout_combs_list = _get_final_combinations(all_combinations_of_layouts, photos,
                                                      portrait_photos_ids, landscape_photos_ids)

    cleaned_comb_data = _filter_combinations(final_layout_combs_list)
    return [Combination(comb_list) for comb_list in cleaned_comb_data]


@dataclass
class SingleSpreadLayout:
    # ToDo combine with Spread class
    layout_idx: int
    left_page_photo_idxs: Set[int]  # photo index in local list of photos per group
    right_page_photo_idxs: Set[int]
    number_of_squares: int
    score: float = None

    def __str__(self):
        return (f'Layout_idx: {self.layout_idx}; '
                f'Photos: [left page: {self.left_page_photo_idxs}, right page: {self.right_page_photo_idxs}]; '
                f'Square boxes in spread: {self.number_of_squares}')

    def to_list(self):
        return list(self.__dict__.values())

    @property
    def as_list(self):
        return self.to_list()


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


def _filter_layouts(layout_df, n_photos, portraits, landscapes):
    return layout_df.loc[
        (layout_df['number of boxes'] == n_photos) &
        (layout_df['max portraits'] >= portraits) &
        (layout_df['max landscapes'] >= landscapes)
    ].copy()


def _count_squares(layouts_df):
    if not layouts_df.empty:
        layouts_df['number of squares'] = layouts_df.apply(
            lambda x: len(list(x['left_square_ids'])) + len(list(x['right_square_ids'])), axis=1
        )
    else:
        layouts_df['number of squares'] = 0
    return layouts_df


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


def _calculate_capacities(layouts):
    greedy_layouts = layouts.copy()

    for side in ('left', 'right'):
        square_len = greedy_layouts[f'{side}_square_ids'].apply(len)
        total_capacity = square_len.copy()
        for orientation in ('portrait', 'landscape'):
            orient_len = greedy_layouts[f'{side}_{orientation}_ids'].apply(len)
            greedy_layouts[f'max_{side}_{orientation}s'] = orient_len + square_len
            total_capacity += orient_len
        greedy_layouts[f'{side}_total_capacity'] = total_capacity
    return greedy_layouts


def _get_time_sequences(spread_photos, photos):
    time_sequeces = [
        (
            photo_id,
            photos[photo_id].general_time,
            (
                photos[photo_id].original_context,
                photos[photo_id].color
            )
        ) for photo_id in spread_photos
    ]
    return time_sequeces


def _group_by_time(spread_photos, photos):
    time_sequeces = _get_time_sequences(spread_photos, photos)
    # sort by id
    time_sequeces = sorted(time_sequeces, key=lambda x: x[1])
    # group by 'general_time'
    grouped = groupby(time_sequeces, key=lambda x: x[2])

    grouped_sequences = []
    for key, group in grouped:
        grouped_sequences.append(list(group))

    return grouped_sequences


def _apply_mask(greedy_layouts, left_landscapes, left_portraits, right_landscapes, right_portraits):
    mask = (
        (greedy_layouts['max_left_landscapes']  >= left_landscapes) &
        (greedy_layouts['max_left_portraits']   >= left_portraits) &
        (greedy_layouts['max_right_landscapes'] >= right_landscapes) &
        (greedy_layouts['max_right_portraits']  >= right_portraits) &
        ((left_landscapes + left_portraits) == greedy_layouts['left_total_capacity']) &
        ((right_landscapes + right_portraits) == greedy_layouts['right_total_capacity'])
    )
    return greedy_layouts.loc[mask]


def _process_with_time(spread_photos, photos, greedy_layouts, greedy_single_spreads):
    grouped_sequences = _group_by_time(spread_photos, photos)

    if len(grouped_sequences) == 2:
        left_landscapes = np.sum([photos[item[0]].ar > 1 for item in grouped_sequences[0]])
        left_portraits = len(grouped_sequences[0]) - left_landscapes
        right_landscapes = np.sum([photos[item[0]].ar > 1 for item in grouped_sequences[1]])
        right_portraits = len(grouped_sequences[1]) - right_landscapes

        possible_layouts = _apply_mask(greedy_layouts, left_landscapes, left_portraits, right_landscapes,
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

        possible_layouts = _apply_mask(greedy_layouts, left_landscapes, left_portraits, right_landscapes,
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
        if len(oriented_combs) > params[4]:
            # print('MaxOrientedCombs crossed sampling oriented combinations instead of full listing')
            # sample_idxs = random.sample(range(len(oriented_combs)), CONFIGS['MaxOrientedCombs'])
            sample_idxs = random.sample(range(len(oriented_combs)), params[4])
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

        layouts = _filter_layouts(layout_df, n_photos_in_spread, portraits, landscapes)
        layouts = _count_squares(layouts)

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
                greedy_layouts = _calculate_capacities(layouts)

                greedy_single_spreads = _process_with_time(spread_photos, photos, greedy_layouts, greedy_single_spreads)
                greedy_single_spreads = _process_with_color(spread_photos, photos, greedy_layouts, greedy_single_spreads)
        except Exception as e:
            print(f"Greedy layout attempt failed with error {e}")

        spreads = _get_spreads(layouts, landscape_set, portrait_set, params, greedy_single_spreads)

        if len(spreads) == 0:
            return None
        if len(spreads) > params[2]:
            # print(f"Sampling {params[2]} spreads from {len(spreads)}")
            sample_idxs = random.sample(range(len(spreads)), params[2])
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


def list_multi_spreads(group_spreads_layouts: GroupLayoutsLists):
    listed_spreads = []
    n_spreads = len(group_spreads_layouts.spreads)
    spreads_in_group = group_spreads_layouts.possible_layouts

    if n_spreads == 1:
        for spread_layout in spreads_in_group[0].possible_layouts:
            listed_spreads.append([[spread_layout.as_list], spread_layout.score * group_spreads_layouts.weight])
    else:
        merged = [[spread_layout] for spread_layout in spreads_in_group[0].possible_layouts]
        for spread_idx in range(1, n_spreads):
            merged = list(product(merged, spreads_in_group[spread_idx].possible_layouts))
            merged = [merged[idx][0] + [merged[idx][1]] for idx in range(len(merged))]

        for merge in merged:
            merge_score = 1
            for spread in merge:
                merge_score *= spread.score

            merge = [spread_layout.as_list for spread_layout in merge]
            listed_spreads.append([merge, merge_score * group_spreads_layouts.weight])

    return listed_spreads


def _get_combinations(partitions, photos, layouts_df, spread_params, params):
    combs = []

    photoTimes = [item.general_time for item in photos]
    cluster_labels = [item.cluster_label for item in photos]

    def eval_combination(combination, partition):
        combination_weight = combination.eval_single_comb(photoTimes, cluster_labels)
        combination.set_weight(combination_weight * partition.weight)

    maxCombsParam = params[2] if len(photos) <= params[5] else params[3]

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
    filtered_multi_spreads = []
    for idx, comb in enumerate(combs):
        multi_spreads = layoutSingleCombination(comb, layouts_df, photos, params)
        if multi_spreads is not None:
            if len(photos) < 13:
                penalty = Penalties(
                    crop_penalty=CONFIGS['crop_penalty'],
                    color_mix=CONFIGS['color_mix'],
                    class_mix=CONFIGS['class_mix'],
                    orientation_mix=CONFIGS['orientation_mix'],
                    score_threshold=params[0],
                    double_mix_color=CONFIGS['double_page_color_mix']
                )
            else:
                penalty = Penalties(
                    crop_penalty=0.8,
                    color_mix=CONFIGS['color_mix'],
                    class_mix=CONFIGS['class_mix'],
                    orientation_mix=CONFIGS['orientation_mix'],
                    score_threshold=params[0],
                    double_mix_color=CONFIGS['double_page_color_mix'],
                    context_mix_penalty=0.00001,
                    time_order_penalty=0.5
                )
            single_filtered_multi_spreads = eval_multi_spreads(multi_spreads, layouts_df, photos, penalty)
            filtered_multi_spreads += list_multi_spreads(single_filtered_multi_spreads)

        # ToDo optimize this
        if len(filtered_multi_spreads) > 10000:
            scores = np.zeros(len(filtered_multi_spreads))
            for multi_spread in range(len(filtered_multi_spreads)):
                scores[multi_spread] = filtered_multi_spreads[multi_spread][1]

            args = np.argsort(scores)[::-1]
            filtered_multi_spreads = [filtered_multi_spreads[args[idx]] for idx in range(1000)]

    if len(filtered_multi_spreads) == 0:
        return None

    scores = np.zeros(len(filtered_multi_spreads))
    for multi_spread in range(len(filtered_multi_spreads)):
        scores[multi_spread] = filtered_multi_spreads[multi_spread][1]

    filtered_scores_idx = np.where(scores / np.max(scores) > 0.01)[0]

    if len(filtered_scores_idx) < 1000:
        filtered_scores = [filtered_multi_spreads[idx] for idx in filtered_scores_idx]
    else:
        args = np.argsort(scores)[::-1]
        filtered_scores = [filtered_multi_spreads[args[idx]] for idx in range(1000)]

    return filtered_scores

