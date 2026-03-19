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


def get_spread_layouts_list(layout_df: pd.DataFrame, layout_part: List[int]) -> List[List[pd.DataFrame]]:
    """
    Group available layouts by spread size for each element in the partition.

    For each spread size in `layout_part`, selects matching layouts from
    `layout_df` and converts each row into a single-row DataFrame.

    Args:
        layout_df: DataFrame of all available layouts with a 'number of boxes' column.
        layout_part: List of spread sizes (number of boxes per spread).

    Returns:
        A nested list where each element corresponds to a spread size and contains
        a list of single-row DataFrames, one per matching layout.
    """
    spread_layouts_list = list()
    for spread_size in layout_part:
        layouts = layout_df.loc[(layout_df['number of boxes'] == spread_size)]
        list_of_single_row_layouts = []
        for index, row in layouts.iterrows():
            single_row_df = row.to_frame().T
            list_of_single_row_layouts.append(single_row_df)
        spread_layouts_list.append(list_of_single_row_layouts)
    return spread_layouts_list


def filter_layouts(layout_df: pd.DataFrame, n_photos: int, portraits: int, landscapes: int) -> pd.DataFrame:
    """
    Filter layouts that can accommodate the given photo orientation counts.

    Args:
        layout_df: DataFrame of all available layouts.
        n_photos: Total number of photos in the spread.
        portraits: Number of portrait photos.
        landscapes: Number of landscape photos.

    Returns:
        A copy of the filtered DataFrame containing only layouts matching
        the box count and supporting enough portrait/landscape slots.
    """
    return layout_df.loc[
        (layout_df['number of boxes'] == n_photos) &
        (layout_df['max portraits'] >= portraits) &
        (layout_df['max landscapes'] >= landscapes)
    ].copy()


def count_squares(layouts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'number of squares' column to the layouts DataFrame.

    Counts square slots by summing left and right square ID list lengths
    for each layout row.

    Args:
        layouts_df: DataFrame of layouts with 'left_square_ids' and
            'right_square_ids' columns. Modified in place.

    Returns:
        The same DataFrame with the 'number of squares' column added.
    """
    if not layouts_df.empty:
        layouts_df['number of squares'] = layouts_df.apply(
            lambda x: len(list(x['left_square_ids'])) + len(list(x['right_square_ids'])), axis=1
        )
    else:
        layouts_df['number of squares'] = 0
    return layouts_df


def update_with_page_capacities(layouts: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-side orientation capacities for each layout.

    For both left and right sides, calculates the maximum number of portraits
    and landscapes each layout can hold (orientation slots + square slots),
    and the total capacity per side.

    Args:
        layouts: DataFrame of layouts with columns for left/right square,
            portrait, and landscape ID lists.

    Returns:
        A copy of the DataFrame with added columns: 'max_left_portraits',
        'max_left_landscapes', 'left_total_capacity', and their right equivalents.
    """
    updated_layouts = layouts.copy()

    for side in ('left', 'right'):
        square_len = updated_layouts[f'{side}_square_ids'].apply(len)
        total_capacity = square_len.copy()
        for orientation in ('portrait', 'landscape'):
            orient_len = updated_layouts[f'{side}_{orientation}_ids'].apply(len)
            updated_layouts[f'max_{side}_{orientation}s'] = orient_len + square_len
            total_capacity += orient_len
        updated_layouts[f'{side}_total_capacity'] = total_capacity
    return updated_layouts


def apply_layouts_mask(extended_layouts: pd.DataFrame,
                       left_landscapes: int, left_portraits: int,
                       right_landscapes: int, right_portraits: int) -> pd.DataFrame:
    """
    Filter layouts whose per-side capacities exactly match the given orientation counts.

    Selects layouts where each side can accommodate at least the required number
    of portraits and landscapes, and the total per-side capacity equals the
    sum of portraits and landscapes for that side.

    Args:
        extended_layouts: DataFrame with capacity columns from `update_with_page_capacities`.
        left_landscapes: Number of landscape photos on the left page.
        left_portraits: Number of portrait photos on the left page.
        right_landscapes: Number of landscape photos on the right page.
        right_portraits: Number of portrait photos on the right page.

    Returns:
        Filtered DataFrame of layouts matching the orientation requirements.
    """
    mask = (
        (extended_layouts['max_left_landscapes']  >= left_landscapes) &
        (extended_layouts['max_left_portraits']   >= left_portraits) &
        (extended_layouts['max_right_landscapes'] >= right_landscapes) &
        (extended_layouts['max_right_portraits']  >= right_portraits) &
        ((left_landscapes + left_portraits) == extended_layouts['left_total_capacity']) &
        ((right_landscapes + right_portraits) == extended_layouts['right_total_capacity'])
    )
    return extended_layouts.loc[mask]