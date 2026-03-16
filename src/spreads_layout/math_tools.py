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