"""Shared preparation applied to every category before its strategy runs.

Three steps, in order:

1. **Temporal narrowing** — drop photos that sit alone in time. A frame with no
   neighbours within 20 minutes is almost always an outlier rather than part of
   a moment worth a spread.
2. **Fixed-span clustering** — bucket what is left into ~4 minute runs, which is
   the unit the diversity pass allocates across.
3. **Colour split** — strategies pick from colour; greyscale is held back to top
   up a shortfall, so a spread does not end up mixing the two by accident.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.selection.ai_wedding_selection import time_clusters_fixed_span
from utils.selection.time_orientation_selection import identify_temporal_clusters
from utils.time_processing import convert_to_timestamp

#: A photo joins a temporal cluster if it is within this many minutes of another.
TEMPORAL_LINK_MINUTES = 20
#: Clusters smaller than this are dropped as isolated.
TEMPORAL_MIN_GROUP = 4
#: Keep everything when the pool is smaller than need x this.
NARROW_HEADROOM = 3


def add_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    frame['image_time_date'] = frame['image_time'].apply(convert_to_timestamp)
    return frame


def drop_temporal_orphans(frame: pd.DataFrame, need: int, logger) -> pd.DataFrame:
    """Remove temporally isolated photos, unless the pool is already thin.

    A pool with less than ``need * NARROW_HEADROOM`` photos is left alone —
    there is no room to be picky.
    """
    if len(frame) < int(np.ceil(need * NARROW_HEADROOM)):
        return frame
    return identify_temporal_clusters(
        frame, 'image_time_date', TEMPORAL_LINK_MINUTES, TEMPORAL_MIN_GROUP, logger
    )


def add_time_clusters(frame: pd.DataFrame, logger) -> pd.DataFrame:
    return time_clusters_fixed_span(frame, logger)


def split_by_color(frame: pd.DataFrame):
    """Returns ``(colour, greyscale)``."""
    return frame[frame['image_color'] != 0], frame[frame['image_color'] == 0]


def order_index(frame: pd.DataFrame, scored: bool) -> dict:
    """``{image_id: ranking key}`` in the direction implied by ``scored``."""
    if scored:
        return frame.set_index('image_id')['total_score'].sort_values(ascending=False).to_dict()
    return frame.set_index('image_id')['image_order'].sort_values(ascending=True).to_dict()
