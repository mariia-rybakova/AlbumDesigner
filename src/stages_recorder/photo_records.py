"""Project photo DataFrames to JSON-safe records for stage visualizers."""

from __future__ import annotations

from typing import List

import pandas as pd


PHOTO_RECORD_COLUMNS = ('image_id', 'image_time_date', 'general_time',
                        'original_context', 'cluster_context', 'time_cluster',
                        'group_sub_index', 'group_size')


def photos_to_records(df: pd.DataFrame) -> List[dict]:
    """Project a photo DataFrame to a list of JSON-safe dicts.

    Carries the fields the visualizers need: `image_id` (to locate the file
    on disk), `image_time_date` (absolute wall-clock time, matches what
    album1.pdf shows), `general_time` (relative offset, kept for completeness),
    `original_context` (for merge captions), plus group-membership context.
    Missing columns are silently skipped; pd.Timestamp values are serialized
    as ISO strings.
    """
    if df is None or len(df) == 0:
        return []
    keep = [c for c in PHOTO_RECORD_COLUMNS if c in df.columns]
    out = []
    for _, row in df[keep].iterrows():
        rec = {}
        for c in keep:
            v = row[c]
            if isinstance(v, pd.Timestamp):
                v = v.isoformat()
            elif hasattr(v, 'item'):
                try:
                    v = v.item()
                except Exception:
                    pass
            rec[c] = v
        out.append(rec)
    return out
