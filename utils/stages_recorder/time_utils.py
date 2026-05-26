"""Time-format helpers used by stage visualizers.

The pipeline reasons about time in two flavours:
  * `general_time` — int seconds since the first photo (relative offset).
  * `image_time_date` — absolute wall-clock timestamp (matches album1.pdf).

The visualizers want to show wall-clock time. These helpers translate.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def hhmmss_from_iso(v: Any) -> str:
    """Extract `HH:MM:SS` from any ISO timestamp string or pd.Timestamp."""
    if v is None or v == '':
        return ''
    if isinstance(v, pd.Timestamp):
        v = v.isoformat()
    s = str(v)
    if 'T' in s:
        s = s.split('T', 1)[1]
    elif ' ' in s:
        s = s.split(' ', 1)[1]
    return s[:8]


def build_general_time_to_clock(photos_df: pd.DataFrame) -> Dict[float, str]:
    """Build a `general_time → "HH:MM:SS"` lookup from `photos_df`.

    Used to translate raw `general_time` floats (the values the splitting /
    merging code reasons about) into wall-clock strings for visualizer logs,
    so the same time-of-day appears under the photo thumbnails and in the
    text trace.

    Empty dict if either column is missing.
    """
    if photos_df is None or len(photos_df) == 0:
        return {}
    if 'general_time' not in photos_df.columns or 'image_time_date' not in photos_df.columns:
        return {}
    out: Dict[float, str] = {}
    for _, row in photos_df[['general_time', 'image_time_date']].iterrows():
        gt = row['general_time']
        if hasattr(gt, 'item'):
            try:
                gt = gt.item()
            except Exception:
                pass
        try:
            key = float(gt)
        except (TypeError, ValueError):
            continue
        out.setdefault(key, hhmmss_from_iso(row['image_time_date']))
    return out
