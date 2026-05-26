"""Split-records accumulator — built up during `handle_wedding_splitting`.

Each iteration of the loop in `handle_wedding_splitting` does one of:
  * size-based split  -> always records the attempt
  * time-based split  -> attempts via `get_split_points`; usually returns None,
                          but the rejection trace is recorded too (the user
                          wants to see why each attempt fizzled).

A `SplitRecorder` holds one record per iteration plus the `general_time ->
wall-clock` map needed to enrich the saved log with HH:MM:SS strings that
match what the image captions show. `flush()` writes `split.json`.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.configs import CONFIGS
from utils.stages_recorder.photo_records import photos_to_records
from utils.stages_recorder.time_utils import build_general_time_to_clock


_GROUPS_OUT_DIR = os.path.join('files', 'stages_info', 'groups')
_OUT_FILE = 'split.json'

# Group-key classes that get the time-based split path. Mirrors the literal in
# `get_split_points`; duplicated here only for note formatting.
_ALLOWED_TIME_BASED_KEYS = ('walking the aisle', 'bride', 'groom', 'bride and groom',
                            'groom party', 'bride party', 'portrait')


def build_split_notes(group_key_class: str, split_method: str,
                      details: Optional[dict],
                      split_points: Optional[List[float]],
                      updated_group: Optional[pd.DataFrame]) -> str:
    """One-line outcome summary, used both for prints and the split.pdf header."""
    if split_method == 'size_based':
        return ('size-based split applied' if updated_group is not None
                else 'size-based: split_big_group returned None')

    if details is None:
        return 'time-based: no decision trace recorded'

    if not details.get('group_key_matched'):
        return f"time-based skipped: group_key {group_key_class!r} not in allowed list"
    if details.get('n_group_times', 0) < 2:
        return 'time-based skipped: fewer than 2 group times'
    if split_points is None:
        return 'time-based: no interval with >2 photos between'
    return f'time-based split applied at {len(split_points)} points'


class SplitRecorder:
    """Accumulates split-attempt records and flushes them once to JSON.

    Built from the same `photos_df` the pipeline iterates so the wall-clock
    lookup it needs is built once, not per attempt.
    """

    def __init__(self, photos_df: pd.DataFrame):
        self._enabled = bool(CONFIGS.get('save_files', {}).get('groups', False))
        self._records: List[dict] = []
        self._general_time_to_clock: Dict[float, str] = (
            build_general_time_to_clock(photos_df) if self._enabled else {}
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ---- private ----
    def _clock(self, gt: Any) -> str:
        if gt is None:
            return ''
        try:
            return self._general_time_to_clock.get(float(gt), '')
        except (TypeError, ValueError):
            return ''

    def _augment_log_with_wall_clock(self, log: dict) -> None:
        """Add `*_clock` HH:MM:SS strings beside every raw general_time in `log`."""
        for interval in log.get('intervals') or []:
            interval['start_clock'] = self._clock(interval.get('start'))
            interval['end_clock'] = self._clock(interval.get('end'))
            interval['between_clocks'] = [self._clock(t) for t in (interval.get('between_times') or [])]
        sp = log.get('split_points') or []
        log['split_points_clock'] = [self._clock(t) for t in sp]

    # ---- public ----
    def record(self,
               group_key: tuple,
               original_group: pd.DataFrame,
               group_spread_size: Any,
               number_of_spreads: int,
               split_method: str,
               notes: str,
               details: Optional[dict],
               updated_group: Optional[pd.DataFrame]) -> None:
        """Record one split attempt — successful or not.

        `details` is the dict `get_split_points` populates when called with
        the `details=` parameter; it carries the per-interval trace. For
        size-based splits or rejected time-based attempts where there is no
        trace, pass None / an empty dict respectively.
        """
        if not self._enabled:
            return

        if details:
            self._augment_log_with_wall_clock(details)

        sub_groups_records = []
        if (updated_group is not None
                and isinstance(updated_group, pd.DataFrame)
                and 'group_sub_index' in updated_group.columns):
            for sub_idx in sorted(updated_group['group_sub_index'].dropna().unique()):
                sub_df = updated_group[updated_group['group_sub_index'] == sub_idx]
                if 'general_time' in sub_df.columns:
                    sub_df = sub_df.sort_values('general_time')
                sub_groups_records.append({
                    'sub_index': int(sub_idx) if hasattr(sub_idx, '__int__') else sub_idx,
                    'photos': photos_to_records(sub_df),
                })

        original_sorted = (original_group.sort_values('general_time')
                           if 'general_time' in original_group.columns else original_group)

        self._records.append({
            'group_key': list(group_key),
            'split_method': split_method,
            'group_spread_size': group_spread_size,
            'number_of_spreads': number_of_spreads,
            'notes': notes,
            'split_points_log': details if details else None,
            'original_photos': photos_to_records(original_sorted),
            'sub_groups': sub_groups_records,
        })

    def flush(self) -> None:
        """Write the accumulated records to split.json (no-op when disabled)."""
        if not self._enabled:
            return
        try:
            os.makedirs(_GROUPS_OUT_DIR, exist_ok=True)
            with open(os.path.join(_GROUPS_OUT_DIR, _OUT_FILE), 'w', encoding='utf-8') as f:
                json.dump({'splits': self._records}, f, indent=2, default=str)
        except Exception:
            pass
