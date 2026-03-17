from __future__ import annotations

import random
from itertools import combinations, product, groupby, permutations
from dataclasses import dataclass
from typing import List, Tuple, Set, Iterable, Callable, Any, Optional
import inspect

import numpy as np
import pandas as pd

from utils.configs import CONFIGS


def all_unique_partitions(n: int) -> list[list[int]]:
    """
    Generate all unique integer partitions of n in descending order.

    Each partition is a list of positive integers that sum to n, sorted in
    non-increasing order. For example, partitions of 4 are:
    [[4], [3,1], [2,2], [2,1,1], [1,1,1,1]].

    Args:
        n: The positive integer to partition.

    Returns:
        A list of all unique partitions, where each partition is a list of ints.
    """
    p = [0] * n  # An array to store a partition
    k = 0  # Index of last element in a partition
    p[k] = n  # Initialize first partition
    # as number itself

    # This loop first prints current partition,
    # then generates next partition. The loop
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


def partitions_with_swaps(seq: List, sizes: List[int], m: int) -> List[Tuple[List[List], int]]:
    """
    Generate all partitions of `seq` into groups of given sizes, ranked by swap cost.

    The cost of a partition is the minimum number of adjacent swaps needed to
    restore the default consecutive split, which equals the number of inversions
    between group labels along the original index order. Partitions exceeding
    `m` swaps are pruned during backtracking.

    Args:
        seq: The sequence of elements to partition.
        sizes: List of group sizes; must sum to len(seq).
        m: Maximum allowed swap cost. Partitions exceeding this are discarded.

    Returns:
        A list of (groups, swaps) tuples sorted by swap cost then lexicographically,
        where groups is a list of lists of elements and swaps is the inversion count.
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


def simple_partitions(photos_ids: Set[int], layout_part: List[int], max_combs: int) -> List[List[Set[int]]]:
    """
    Generate combinatorial partitions of photo IDs into groups of specified sizes.

    Iteratively builds partitions by selecting combinations for each group size
    in `layout_part`. The last group receives all remaining photos. Sampling is
    applied when the number of combinations exceeds `max_combs`.

    Args:
        photos_ids: Set of photo indices to partition.
        layout_part: List of group sizes defining the partition structure.
        max_combs: Maximum number of combinations to keep at each step.

    Returns:
        A list of partitions, where each partition is a list of sets of photo indices.
    """
    l0_combs = list(combinations(photos_ids, layout_part[0]))

    if len(l0_combs) > max_combs:
        sample_idxs = random.sample(range(len(l0_combs)), max_combs)
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
            if len(layout_combs) > max_combs:
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
        if len(layout_combs) > max_combs:
            # print(f"Sampling {max_combs} combinations from {len(layout_combs)}")
            sample_idxs = random.sample(range(len(layout_combs)), max_combs)
            layout_combs = [layout_combs[i] for i in sample_idxs]
            # rem_photos = [rem_photos[i] for i in sample_idxs]
        for comb_idx in range(len(layout_combs)):
            layout_combs[comb_idx].append(rem_photos[comb_idx])
    return layout_combs