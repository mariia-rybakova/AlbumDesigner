from __future__ import annotations

import random
from itertools import combinations, product, groupby, permutations
from dataclasses import dataclass
from typing import List, Tuple, Set, Iterable, Callable, Any, Optional
import inspect

import numpy as np
import pandas as pd

from src.spreads_layout.math_tools import all_unique_partitions
from utils.configs import CONFIGS


def get_layouts_dict(layouts_df: pd.DataFrame, available_n: Set[int]) -> dict[int, pd.DataFrame]:
    """
    Build a mapping from box count to available layout configurations.

    For each unique box count in `available_n`, extracts the distinct
    (max portraits, max landscapes) pairs from the layouts DataFrame.

    Args:
        layouts_df: DataFrame of layouts with 'number of boxes',
            'max portraits', and 'max landscapes' columns.
        available_n: Set of box counts to include in the mapping.

    Returns:
        Dict mapping each box count to a DataFrame of its unique
        (max portraits, max landscapes) combinations.
    """
    layouts_dict = dict()
    for item in list(available_n):
        layouts_dict[item] = layouts_df[layouts_df['number of boxes'] == item][['max portraits', 'max landscapes']].drop_duplicates()
    return layouts_dict