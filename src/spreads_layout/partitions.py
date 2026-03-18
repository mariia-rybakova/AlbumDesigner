from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from src.spreads_layout.math_tools import all_unique_partitions
from src.spreads_layout.layouts_tools import get_layouts_dict
from src.core.models import SpreadSearchParams


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


def get_partitions(photos_df: pd.DataFrame, class_spread_params: List[float],
                     params: SpreadSearchParams, layouts_df: pd.DataFrame) -> List[Partition]:
    """
    Find all feasible partitions for a group of photos and rank them by fit.

    Generates every unique partition of n_photos into spread sizes that exist
    in the available layouts, scores each partition using a Gaussian model
    parameterized by the context class, then filters by layout feasibility
    (portrait/landscape capacity) and spread count.

    Args:
        photos_df: DataFrame of photos in the group. Must contain an 'ar'
            (aspect ratio) column to distinguish portraits (ar < 1) from landscapes.
        class_spread_params: [mean, std] Gaussian parameters for the context
            class spread-size distribution. std is clamped to at least 0.5.
        params: Search parameters controlling weight threshold and other limits.
        layouts_df: DataFrame of available layout designs with a
            'number of boxes' column.

    Returns:
        List of valid Partition objects sorted by weight (descending), filtered
        by layout feasibility and spread-count constraints. May be empty if no
        partition fits the available layouts.
    """
    n_photos = len(photos_df.index)
    n_portraits = len(photos_df[photos_df['ar'] < 1].index)
    n_landscapes = n_photos - n_portraits

    available_n = set(layouts_df['number of boxes'].unique())
    layouts_dict = get_layouts_dict(layouts_df, available_n)

    class_spread_params[1] = max(class_spread_params[1], 0.5)

    # generate partitions
    ## part values
    parts = all_unique_partitions(n_photos)
    parts = [part for part in parts if set(part).issubset(available_n)]
    ## part weights
    weights = Partition.get_weights_for_parts(parts, class_spread_params, n_photos)
    parts = [Partition(parts[i], weights[i]) for i in range(len(parts))]
    # process partitions
    sorted_parts = sorted(parts, key=lambda p: p.weight, reverse=True)
    filtered_parts = Partition.filter_by_layout(sorted_parts, layouts_dict, n_portraits, n_landscapes, params)
    valid_parts = Partition.filter_by_len(filtered_parts, n_photos)

    return valid_parts
