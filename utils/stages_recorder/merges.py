"""Merge event log — feeds merge.json / merge.pdf.

The merge pipeline emits three event types in chronological order:

  * search           — one src group's hunt for a partner. Includes every
                       candidate evaluated by `merge_illegal_group_by_time`
                       with raw + adjusted time distances, plus the selected
                       partner (or None).
  * merge_skipped    — proposed merge rejected at apply time (src or partner
                       already used, bridegroom size-balance failure, or one
                       of the singleton fallback rejection paths).
  * merge_succeeded  — merge that actually happened; carries photo records
                       for the merged group and (optionally) a reminder group.

`merge_illegal_group_by_time` itself accepts an optional `details=` list and
appends per-candidate dicts to it as it iterates. The functions in this module
turn that raw trace + caller context into the events above and push them onto
a single chronological log, flushed once at the end of grouping.
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional

import pandas as pd

from utils.configs import CONFIGS
from utils.stages_recorder.photo_records import photos_to_records


_GROUPS_OUT_DIR = os.path.join('files', 'stages_info', 'groups')
_OUT_FILE = 'merge.json'

_merge_events: List[dict] = []


def _save_on() -> bool:
    return bool(CONFIGS.get('save_files', {}).get('groups', False))


# ---------- log lifecycle ----------

def reset_merge_events() -> None:
    """Clear the per-run event log. Called once at the start of grouping."""
    _merge_events.clear()


def get_merge_events() -> List[dict]:
    """Snapshot of the event log so far."""
    return list(_merge_events)


def flush_merge_events() -> None:
    """Write the accumulated events to merge.json (no-op when disabled)."""
    if not _save_on():
        return
    try:
        os.makedirs(_GROUPS_OUT_DIR, exist_ok=True)
        with open(os.path.join(_GROUPS_OUT_DIR, _OUT_FILE), 'w', encoding='utf-8') as f:
            json.dump({'events': list(_merge_events)}, f, indent=2, default=str)
    except Exception:
        pass


# ---------- private helpers ----------

def _df_group_key(df: pd.DataFrame) -> tuple:
    """Extract `(time_cluster, cluster_context, group_sub_index)` from a group df."""
    row = df.iloc[0]

    def _scalar(v):
        if hasattr(v, 'item'):
            try:
                return v.item()
            except Exception:
                pass
        return v
    return (_scalar(row['time_cluster']),
            _scalar(row['cluster_context']),
            _scalar(row['group_sub_index']))


def _mean_general_time(df: pd.DataFrame) -> Optional[float]:
    if df is None or 'general_time' not in df.columns or len(df) == 0:
        return None
    try:
        return float(df['general_time'].mean())
    except Exception:
        return None


def _mean_image_time_date(df: pd.DataFrame) -> Optional[str]:
    if df is None or 'image_time_date' not in df.columns or len(df) == 0:
        return None
    try:
        ts = pd.to_datetime(df['image_time_date'], errors='coerce').dropna()
        if len(ts) == 0:
            return None
        return ts.mean().isoformat()
    except Exception:
        return None


def _append(event_type: str, **fields: Any) -> None:
    if not _save_on():
        return
    _merge_events.append({'type': event_type, **fields})


# ---------- public recorders ----------

def new_candidate_details() -> Optional[List[dict]]:
    """Return a list for `merge_illegal_group_by_time(... details=...)`.

    None when recording is disabled, so the function can short-circuit its
    extra bookkeeping.
    """
    return [] if _save_on() else None


def record_search(merge_type: str,
                  src_key: tuple,
                  src_group: pd.DataFrame,
                  candidates: Optional[List[dict]],
                  selected_cluster: Optional[pd.DataFrame],
                  selected_time_diff: Optional[float],
                  **extra: Any) -> None:
    """Record one src group's search for a partner."""
    if not _save_on():
        return
    try:
        selected_partner_key = _df_group_key(selected_cluster) if selected_cluster is not None else None
    except Exception:
        selected_partner_key = None
    _append(
        'search',
        merge_type=merge_type,
        src_key=list(src_key),
        src_n_photos=int(len(src_group)),
        src_mean_general_time=_mean_general_time(src_group),
        src_mean_image_time_date=_mean_image_time_date(src_group),
        candidates=candidates or [],
        selected_partner_key=list(selected_partner_key) if selected_partner_key else None,
        selected_time_diff=selected_time_diff,
        **extra,
    )


def record_skip(merge_type: str,
                src_key: tuple,
                partner_key: Optional[tuple],
                time_diff: Optional[float],
                reason: str) -> None:
    """Record a proposed merge that didn't go through."""
    _append(
        'merge_skipped',
        merge_type=merge_type,
        src_key=list(src_key),
        partner_key=list(partner_key) if partner_key else None,
        time_diff=time_diff,
        reason=reason,
    )


def record_succeeded(merge_type: str,
                     src_key: tuple,
                     partner_key: tuple,
                     time_diff: Optional[float],
                     merged_group: pd.DataFrame,
                     reminder_group: Optional[pd.DataFrame] = None) -> None:
    """Record a merge that actually happened. Carries photo records for rendering."""
    if not _save_on():
        return
    if merged_group is not None and 'general_time' in merged_group.columns:
        merged_group = merged_group.sort_values('general_time')
    if reminder_group is not None and 'general_time' in reminder_group.columns:
        reminder_group = reminder_group.sort_values('general_time')
    _append(
        'merge_succeeded',
        merge_type=merge_type,
        src_key=list(src_key),
        partner_key=list(partner_key),
        time_diff=time_diff,
        merged_photos=photos_to_records(merged_group),
        reminder_photos=photos_to_records(reminder_group),
    )
