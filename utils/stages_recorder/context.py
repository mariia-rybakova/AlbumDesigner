"""Per-run recorder context shared by all stage recorders.

Currently holds a single fact: whether the gallery's timeline is synthetic
(`is_artificial_time`). It is set once, before grouping starts, by
`album_processing` — the point in the pipeline where the flag (computed back in
`process_gallery_time`) is known. Every recorder reads it at flush time and
stamps it onto its JSON, so the stage visualizers can pick the right time field
(wall-clock `image_time_date` normally; elapsed `general_time` when artificial),
mirroring what `process_gallery.py` does for album1.pdf.

Module-level state mirrors the existing recorder style (`merges._merge_events`):
the recorders are global, per-run singletons, not threaded objects.
"""

from __future__ import annotations

_is_artificial_time = False


def set_is_artificial_time(flag: bool) -> None:
    """Set the artificial-time flag for the current run (called once before grouping)."""
    global _is_artificial_time
    _is_artificial_time = bool(flag)


def get_is_artificial_time() -> bool:
    """Whether the current gallery's timeline is synthetic."""
    return _is_artificial_time