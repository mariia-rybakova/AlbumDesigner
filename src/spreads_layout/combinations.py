from __future__ import annotations

import random
from itertools import product
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional

import numpy as np
import pandas as pd

from src.spreads_layout.math_tools import simple_partitions, partitions_with_swaps
from src.spreads_layout.layouts_tools import get_spread_layouts_list
from src.spreads_layout.partitions import Partition
from src.core.photos import get_portraits_landscapes, count_photo_times_per_class
from src.core.models import SpreadSearchParams


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


def simple_combination_search(photos: List, layout_part: Partition, max_combs: int) -> List[Combination]:
    """
    Enumerate concrete photo-to-spread assignments for a given partition.

    For multi-spread partitions, uses swap-based enumeration to generate
    assignments. For single-spread partitions, uses simple partitioning
    capped at max_combs.

    Args:
        photos: List of Photo objects in the group.
        layout_part: The Partition defining spread sizes.
        max_combs: Maximum number of combinations to generate (applies to
            single-spread partitions only).

    Returns:
        List of Combination objects, each assigning photo indices to spreads.
    """
    photos_ids = list(range(len(photos)))
    photos_ids = set(photos_ids)

    if len(layout_part.spread_sizes) > 1:
        layout_combs = partitions_with_swaps(list(photos_ids), layout_part.spread_sizes, 2)
        layout_combs = [Combination([set(part) for part in comb]) for comb, v in layout_combs]
    else:
        layout_combs = simple_partitions(photos_ids, layout_part.spread_sizes, max_combs)
        layout_combs = [Combination(comb_list) for comb_list in layout_combs]
    return layout_combs


def _get_portraits_landscapes_sorted(photos: List) -> Tuple[List[int], List[int]]:
    """
    Separate photo indices into portrait and landscape lists, sorted by context
    mean time then individual time.

    Uses get_portraits_landscapes to split by orientation, then sorts each list
    by the mean timestamp of the photo's original_context class (primary key)
    and the photo's own timestamp (secondary key).

    Args:
        photos: List of Photo objects in the group.

    Returns:
        Tuple of (portrait_photo_ids, landscape_photo_ids), each sorted by
        the mean time of their original_context group, then by their own time.
    """
    photos_idxs = list(range(len(photos)))
    portrait_idxs, landscape_idxs = get_portraits_landscapes(photos_idxs, photos)
    portrait_idxs, landscape_idxs = list(portrait_idxs), list(landscape_idxs)

    context_times = count_photo_times_per_class(photos)
    context_means = {class_name: np.mean(times_list) for class_name, times_list in context_times.items()}

    def get_time_values(idx: int) -> List[float]:
        class_name = photos[idx].original_context
        return [context_means[class_name], photos[idx].general_time]

    landscape_idxs = sorted(landscape_idxs, key=get_time_values)
    portrait_idxs = sorted(portrait_idxs, key=get_time_values)

    return portrait_idxs, landscape_idxs


def _prepare_all_combinations(spread_layouts_list: List[List[pd.DataFrame]]) -> List:
    """
    Build all layout combinations across spreads.

    For groups with 4+ spreads, takes only the first layout option per spread
    to avoid combinatorial explosion. Otherwise, computes the full cartesian
    product. Caps the result at 1000 by random sampling.

    Args:
        spread_layouts_list: List of layout options per spread, where each
            element is a list of single-row DataFrames.

    Returns:
        List of layout combinations (each a tuple/list of DataFrames).
    """
    if len(spread_layouts_list) >= 4:
        all_combinations_of_layouts = [[sublist[0] for sublist in spread_layouts_list]]
    else:
        all_combinations_of_layouts = list(product(*spread_layouts_list))

    if len(all_combinations_of_layouts) > 1000:
        all_combinations_of_layouts = random.sample(all_combinations_of_layouts, 1000)

    return all_combinations_of_layouts


def _get_final_combinations(all_combinations_of_layouts: List, photos: List,
                            portrait_photos_ids: List[int], landscape_photos_ids: List[int]) -> List[List[Set[int]]]:
    """
    Assign photo indices to spreads based on layout orientation requirements.

    For each layout combination, assigns portraits first, then landscapes,
    then fills remaining square slots by choosing the photo with the earlier
    timestamp. Skips combinations where total boxes, portraits, or landscapes
    don't match the available photos.

    Args:
        all_combinations_of_layouts: Layout combinations from _prepare_all_combinations.
        photos: List of Photo objects.
        portrait_photos_ids: Sorted portrait photo indices.
        landscape_photos_ids: Sorted landscape photo indices.

    Returns:
        List of combinations, each a list of sets of photo indices per spread.
    """
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


def _filter_combinations(final_layout_combs_list: List[List[Set[int]]]) -> List[List[Set[int]]]:
    """
    Deduplicate combinations by removing those with identical frozen sets.

    Args:
        final_layout_combs_list: List of combinations to deduplicate.

    Returns:
        List of unique combinations.
    """
    cleaned_comb_data = []
    seen = set()

    for inner_list in final_layout_combs_list:
        frozen_inner_list = tuple(frozenset(s) for s in inner_list)

        if frozen_inner_list not in seen:
            seen.add(frozen_inner_list)
            cleaned_comb_data.append(inner_list)
    return cleaned_comb_data


def greedy_combination_search(photos: List, layout_part: Partition, layout_df: pd.DataFrame) -> List[Combination]:
    """
    Build combinations by greedily assigning photos to layouts based on orientation.

    Generates layout combinations for the partition's spread sizes, assigns
    photos by orientation (portrait/landscape) and time order, then deduplicates.

    Args:
        photos: List of Photo objects in the group.
        layout_part: The Partition defining spread sizes.
        layout_df: DataFrame of available layout designs.

    Returns:
        List of unique Combination objects.
    """
    portrait_photos_ids, landscape_photos_ids = _get_portraits_landscapes_sorted(photos)

    spread_layouts_list = get_spread_layouts_list(layout_df, layout_part.spread_sizes)

    all_combinations_of_layouts = _prepare_all_combinations(spread_layouts_list)

    final_layout_combs_list = _get_final_combinations(all_combinations_of_layouts, photos,
                                                      portrait_photos_ids, landscape_photos_ids)

    cleaned_comb_data = _filter_combinations(final_layout_combs_list)
    return [Combination(comb_list) for comb_list in cleaned_comb_data]


def get_combinations(partitions: List[Partition], photos: List, layouts_df: pd.DataFrame, spread_params: List[float], params: SpreadSearchParams) -> List[Combination]:
    """
    Generate and evaluate all photo-to-spread combinations across partitions.

    For each partition, generates combinations using either simple enumeration
    (small groups) or greedy search (larger groups), samples if too many are
    found, then scores each by temporal coherence and cluster label consistency.
    Earlier partitions (higher weight) get a larger combination budget that
    halves with each subsequent partition.

    Args:
        partitions: List of Partition objects to generate combinations for.
        photos: List of Photo objects in the group.
        layouts_df: DataFrame of available layout designs.
        spread_params: [mean, std] spread size parameters for the context class.
        params: Search parameters controlling combination limits and thresholds.

    Returns:
        List of scored Combination objects across all partitions.
    """
    combs = []

    photo_times = [item.general_time for item in photos]
    cluster_labels = [item.cluster_label for item in photos]

    def eval_combination(combination: Combination, part: Partition) -> None:
        combination_weight = combination.get_evaluation_score(photo_times, cluster_labels)
        combination.set_weight(combination_weight * part.weight)

    max_combs_param = params.max_spreads_sample if len(photos) <= params.small_group_threshold else params.max_combs_small_group

    for i, partition in enumerate(partitions):
        max_combs = int(max_combs_param / np.power(2, i))

        if len(photos) <= 8 and len(photos) / spread_params[0] <= 2:
            single_combs = simple_combination_search(photos, partition, max_combs)
        else:
            single_combs = greedy_combination_search(photos, partition, layouts_df)

        if len(single_combs) > max_combs:
            sample_idxs = random.sample(range(len(single_combs)), max_combs)
            single_combs = [single_combs[sample_idx] for sample_idx in sample_idxs]

        for comb in single_combs:
            eval_combination(comb, partition)
        combs += single_combs

    return combs