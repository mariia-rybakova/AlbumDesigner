"""Subgroup snapshots — used for subgroups_{0,1,2}.json."""

from __future__ import annotations

import json
import os
from typing import List

import pandas as pd

from utils.configs import CONFIGS
from utils.stages_recorder.context import get_is_artificial_time
from utils.stages_recorder.photo_records import photos_to_records


_GROUPS_OUT_DIR = os.path.join('files', 'stages_info', 'groups')


def snapshot_subgroups(photos_df: pd.DataFrame) -> List[dict]:
    """Project `photos_df` to a chronologically-ordered list of subgroup records.

    "Subgroup" here is one `(time_cluster, cluster_context, group_sub_index)`
    bucket — same key the rest of the pipeline uses. For each bucket we record:
      - `group_key`
      - `mean_general_time`: the value the merge logic in
        `merge.py:merge_illegal_group_by_time` actually compares groups by.
      - `mean_image_time_date`: ISO mean of `image_time_date`, used by
        the visualizer's wall-clock display.
      - `n_photos`
      - `photos`: `photos_to_records` output, sorted by `general_time` so the
        strip reads chronologically left-to-right.

    The outer list is sorted by `mean_general_time`.
    """
    if photos_df is None or len(photos_df) == 0:
        return []
    if not all(c in photos_df.columns for c in ('time_cluster', 'cluster_context', 'group_sub_index')):
        return []

    snapshots = []
    for key, group in photos_df.groupby(['time_cluster', 'cluster_context', 'group_sub_index']):
        sorted_group = group.sort_values('general_time') if 'general_time' in group.columns else group

        mean_general = None
        if 'general_time' in group.columns:
            try:
                mean_general = float(group['general_time'].mean())
            except Exception:
                mean_general = None

        mean_date = None
        if 'image_time_date' in group.columns:
            try:
                ts = pd.to_datetime(group['image_time_date'], errors='coerce').dropna()
                if len(ts) > 0:
                    mean_date = ts.mean().isoformat()
            except Exception:
                mean_date = None

        snapshots.append({
            'group_key': [(k.item() if hasattr(k, 'item') else k) for k in key],
            'mean_general_time': mean_general,
            'mean_image_time_date': mean_date,
            'n_photos': int(len(group)),
            'photos': photos_to_records(sorted_group),
        })

    snapshots.sort(key=lambda s: s['mean_general_time'] if s['mean_general_time'] is not None else 0.0)
    return snapshots


def save_subgroups_snapshot(photos_df: pd.DataFrame, filename: str) -> None:
    """Write a chronological snapshot of `photos_df`'s subgroups to JSON.

    No-op when `save_files['groups']` is disabled. Creates the parent dir.
    Failures are swallowed (visualizer code already handles missing files).
    """
    if not CONFIGS.get('save_files', {}).get('groups', False):
        return
    try:
        os.makedirs(_GROUPS_OUT_DIR, exist_ok=True)
        snapshot = {'is_artificial_time': get_is_artificial_time(),
                    'subgroups': snapshot_subgroups(photos_df)}
        path = os.path.join(_GROUPS_OUT_DIR, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2, default=str)
    except Exception:
        pass
