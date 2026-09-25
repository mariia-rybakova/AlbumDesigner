"""Spread-split records — used for files/stages_info/combinations/<group>.json.

Answers one question per sub-group: *why was it cut into these spreads, with
these photos in each?* Stages 1 and 2 of `find_spreads_layouts_for_subgroup`
decide that:

  stage 1 (`get_partitions`)   — how many spreads and how big each one is.
      Every integer partition of n_photos is scored by a Gaussian on the
      class's photos-per-spread parameter, then filtered by layout feasibility
      and spread count. The trace keeps *all* candidates plus the stage that
      dropped each one, so "why not 3 spreads of 5?" has an answer.
  stage 2 (`get_combinations`) — which exact photos land in which spread.
      Per surviving partition, candidate assignments are scored by time spread
      and duplicate cluster labels, weighted by the partition weight.

Stage 3 (layouts) then re-ranks everything, so the combination with the top
stage-2 weight is *not* necessarily the one that shipped. The `winner` block
records which combination the finally-selected layout actually corresponds to —
that gap is usually the interesting part of the PDF.

Records are stashed as the search produces them and written later, when the
group id and the winning layout are known. That keeps the retry/split loop in
`spreads_layout.main` free of recorder plumbing: the stash is keyed by the
sub-group's photo ids, so a retry on the same photos simply overwrites the
earlier attempt and only the surviving search is exported.

Gated on `CONFIGS['save_files']['combinations']`; every entry point is a cheap
no-op when the flag is off.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List, Optional, Sequence

from src.core.photos import Photo
from utils.configs import CONFIGS, SPECIAL_GROUP_SEP
from utils.stages_recorder.context import get_image_time_date, get_is_artificial_time


_COMBINATIONS_OUT_DIR = os.path.join('files', 'stages_info', 'combinations')

# Combinations kept per partition in the JSON. The search can sample up to
# `max_spreads_sample` of them; the PDF only ever shows a handful, and the
# winner is force-included even when it falls outside this cut.
_TOP_COMBS_PER_PARTITION = 5

# Pending records for the group currently being processed, keyed by the
# sub-group's photo ids. Module-level per-run state, matching the style of the
# other recorders (`merges._merge_events`).
_pending_records: Dict[tuple, dict] = {}


def is_enabled() -> bool:
    """Whether combination recording is switched on for this run."""
    return bool(CONFIGS.get('save_files', {}).get('combinations', False))


def reset_output_dir() -> None:
    """Wipe stale per-subgroup JSONs from a previous run and recreate the dir.

    Same reasoning as the spreads recorder: files whose group key isn't reused
    by the new run would otherwise linger and mix into the next PDF.
    """
    if not is_enabled():
        return
    shutil.rmtree(_COMBINATIONS_OUT_DIR, ignore_errors=True)
    os.makedirs(_COMBINATIONS_OUT_DIR, exist_ok=True)


def _photo_records(photos: Sequence[Photo]) -> List[dict]:
    """One record per photo, keyed by the *local* index the combinations use.

    Combinations address photos by their position in the sub-group's photo
    list, never by `image_id`, so the index has to travel with the record for
    the visualizer to resolve a spread back to actual images.
    """
    return [
        {
            'idx': idx,
            'image_id': photo.id,
            'general_time': float(photo.general_time) if photo.general_time is not None else None,
            # The wall clock album1.pdf prints. `Photo` has no such field, so it
            # is looked up by `general_time` through the run's time index; the
            # visualizer picks whichever of the two suits the gallery.
            'image_time_date': get_image_time_date(photo.general_time),
            'ar': float(photo.ar) if photo.ar is not None else None,
            'orientation': 'portrait' if (photo.ar is not None and photo.ar < 1) else 'landscape',
            'rank': float(photo.rank) if photo.rank is not None else None,
            'photo_class': photo.photo_class,
            'original_context': photo.original_context,
            'cluster_label': (photo.cluster_label.item()
                              if hasattr(photo.cluster_label, 'item') else photo.cluster_label),
            'color': bool(photo.color),
        }
        for idx, photo in enumerate(photos)
    ]


def _comb_key(spreads: Sequence[Sequence[int]]) -> tuple:
    """Order-insensitive identity of a photo-to-spread assignment.

    Spread order inside a candidate isn't meaningful for identity (the layout
    stage re-orders spreads by time), so compare the *set* of spreads.
    """
    return tuple(sorted(tuple(sorted(int(i) for i in spread)) for spread in spreads))


def build_subgroup_record(photos: Sequence[Photo], spread_params: Sequence[float],
                          partitions_trace: dict, combinations_trace: List[dict]) -> dict:
    """Assemble the JSON-safe record for one sub-group search.

    Called right after stages 1-2 finish, while the traces are still in scope.
    Trimming happens here (not in the pipeline) so the layout code never pays
    for serialization it doesn't need.

    Args:
        photos: The sub-group's photos, in the order combinations index them.
        spread_params: `[mean, std]` photos-per-spread actually used for this
            search — already scaled down if an earlier attempt failed.
        partitions_trace: Filled by `get_partitions(..., trace=...)`.
        combinations_trace: Filled by `get_combinations(..., trace=...)`.

    Returns:
        A dict ready for `save_subgroup_record`, minus the `winner` block which
        is only known after stage 3.
    """
    partitions_by_combs = []
    for entry in combinations_trace:
        combs = list(entry.get('combinations') or [])
        ranked = sorted(combs,
                        key=lambda comb: (comb.weight if comb.weight is not None else -1.0),
                        reverse=True)
        partitions_by_combs.append({
            'partition_idx': entry['partition_idx'],
            'spread_sizes': entry['spread_sizes'],
            'partition_weight': entry['partition_weight'],
            'search': entry['search'],
            'max_combs': entry['max_combs'],
            'n_generated': entry['n_generated'],
            # None when the chronology-first filter did not run for this group.
            'n_time_disjoint': entry.get('n_time_disjoint'),
            'n_sampled': entry['n_sampled'],
            'top_combinations': [
                dict(comb.to_dict(), rank=rank)
                for rank, comb in enumerate(ranked[:_TOP_COMBS_PER_PARTITION])
            ],
        })

    return {
        'is_artificial_time': get_is_artificial_time(),
        'spread_params': {'mean': float(spread_params[0]), 'std': float(spread_params[1])},
        'n_photos': len(photos),
        'photos': _photo_records(photos),
        'partitions': partitions_trace,
        'combinations': partitions_by_combs,
        'winner': None,
    }


def _photos_key(photos: Sequence[Photo]) -> tuple:
    """Stash key: the sub-group's photo ids in search order."""
    return tuple(photo.id for photo in photos)


def reset_subgroup_records() -> None:
    """Drop pending records; called once per group before its search starts."""
    _pending_records.clear()


def stash_subgroup_record(photos: Sequence[Photo], record: dict) -> None:
    """Hold a sub-group record until its group id and winning layout are known.

    A retry with relaxed spread params re-searches the same photos and stores
    under the same key, so the last (surviving) attempt is the one exported.
    """
    if not is_enabled():
        return
    _pending_records[_photos_key(photos)] = record


def _attach_winner(record: Optional[dict], best_layout: Any) -> None:
    """Mark which recorded combination the finally-selected layout came from.

    `best_layout.spreads_layouts` carries the same local photo indices the
    combinations use, so the union of each spread's two pages reconstructs the
    combination that won. When that combination fell outside the per-partition
    top cut, it is appended to its partition's list (flagged `outside_top`) so
    the PDF can always show it.

    Silently does nothing when there is no record or no layout — recording is
    best-effort and must never break the pipeline.
    """
    if not record or best_layout is None:
        return

    winning_spreads = [sorted(set(spread.left_page_photo_idxs) | set(spread.right_page_photo_idxs))
                       for spread in best_layout.spreads_layouts]
    key = _comb_key(winning_spreads)

    matched_partition_idx = None
    matched_rank = None
    for partition in record.get('combinations', []):
        for comb in partition.get('top_combinations', []):
            if _comb_key(comb['spreads']) == key:
                comb['is_winner'] = True
                matched_partition_idx = partition['partition_idx']
                matched_rank = comb.get('rank')
                break
        if matched_partition_idx is not None:
            break

    if matched_partition_idx is None:
        # Winner scored below the top cut (or came from a partition whose
        # sample was trimmed) — attach it to the partition with matching
        # spread sizes so the PDF still shows the assignment that shipped.
        sizes = sorted(len(spread) for spread in winning_spreads)
        for partition in record.get('combinations', []):
            if sorted(partition['spread_sizes']) == sizes:
                partition.setdefault('top_combinations', []).append({
                    'spreads': winning_spreads,
                    'score': None,
                    'weight': None,
                    'score_breakdown': None,
                    'rank': None,
                    'is_winner': True,
                    'outside_top': True,
                })
                matched_partition_idx = partition['partition_idx']
                break

    record['winner'] = {
        'spreads': winning_spreads,
        'spread_sizes': [len(spread) for spread in winning_spreads],
        'partition_idx': matched_partition_idx,
        'combination_rank': matched_rank,
        'final_layout_score': (float(best_layout.score) if best_layout.score is not None else None),
        'final_layout_weight': (float(best_layout.weight) if best_layout.weight is not None else None),
    }


def save_subgroup_record(group_id_str: str, photos: Sequence[Photo], best_layout: Any) -> None:
    """Pop this sub-group's stashed record, mark the winner and write it out.

    Filename mirrors `spreads_layout.main.export_subgroup` so the two PDFs can
    be cross-read group by group. Failures are swallowed — the visualizers
    already tolerate missing files, and a debug dump must never take down a
    real album run.
    """
    if not is_enabled():
        return
    try:
        record = _pending_records.pop(_photos_key(photos), None)
        if not record:
            return
        _attach_winner(record, best_layout)
        os.makedirs(_COMBINATIONS_OUT_DIR, exist_ok=True)
        name = group_id_str.replace(" ", "_").replace("*", "_").replace(SPECIAL_GROUP_SEP, "_")
        record = dict(record, group_id=group_id_str)
        with open(os.path.join(_COMBINATIONS_OUT_DIR, f"{name}.json"), 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, default=str)
    except Exception:
        pass