"""Per-run recorder context shared by all stage recorders.

Holds the two run-level facts a recorder needs to report time the way
album1.pdf does: whether the gallery's timeline is synthetic
(`is_artificial_time`), and a `general_time -> image_time_date` index for the
recorders that work from `Photo` objects, which carry only the relative time.
Both are set once, before grouping starts, by
`album_processing` — the point in the pipeline where the flag (computed back in
`process_gallery_time`) is known. Every recorder reads it at flush time and
stamps it onto its JSON, so the stage visualizers can pick the right time field
(wall-clock `image_time_date` normally; elapsed `general_time` when artificial),
mirroring what `process_gallery.py` does for album1.pdf.

Module-level state mirrors the existing recorder style (`merges._merge_events`):
the recorders are global, per-run singletons, not threaded objects.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from utils.stages_recorder.time_utils import build_general_time_to_image_time

_is_artificial_time = False
_general_time_to_image_time: Dict[float, Any] = {}


def set_is_artificial_time(flag: bool) -> None:
    """Set the artificial-time flag for the current run (called once before grouping)."""
    global _is_artificial_time
    _is_artificial_time = bool(flag)


def get_is_artificial_time() -> bool:
    """Whether the current gallery's timeline is synthetic."""
    return _is_artificial_time


def set_general_time_index(photos_df: pd.DataFrame) -> None:
    """Build this run's `general_time -> image_time_date` lookup.

    Called beside `set_is_artificial_time`, from the same place and for the same
    reason: a recorder downstream of `get_photos_from_df` sees `Photo` objects,
    and `Photo` has no `image_time_date` field. Without this the only time such
    a recorder can write is the relative `general_time`, and its PDF then prints
    seconds-since-the-first-photo under thumbnails that album1.pdf labels with a
    wall clock.
    """
    global _general_time_to_image_time
    _general_time_to_image_time = build_general_time_to_image_time(photos_df)


def get_image_time_date(general_time: Any) -> Optional[Any]:
    """The absolute timestamp for a `general_time`, or None if unknown.

    None rather than a placeholder: the visualizers already render a missing
    `image_time_date` as an empty caption line, and inventing a time here would
    be worse than showing none.
    """
    if general_time is None:
        return None
    try:
        return _general_time_to_image_time.get(float(general_time))
    except (TypeError, ValueError):
        return None