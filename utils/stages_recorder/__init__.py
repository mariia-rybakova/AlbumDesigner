"""Stages-info data collection helpers.

Pipeline functions in `src/groups_operations` and `src/spreads_layout` stay
focused on their algorithm; everything that exists *only* to feed the analysis
PDFs (`split.pdf`, `merge.pdf`, `subgroups_*.pdf`, `spreads_layouts.pdf`)
lives here.

Public API (re-exported below) covers the four collection responsibilities:

  * photo records — projecting a photo DataFrame to a JSON-safe list of dicts
    that the visualizers can render. Shared shape used by every grouping PDF.
  * time helpers   — mapping the pipeline's relative `general_time` (seconds)
    back to the wall-clock `image_time_date` strings that match album1.pdf.
  * subgroups      — snapshot the current state of `photos_df` into one of
    `subgroups_0/1/2.json`.
  * splits         — accumulate per-attempt records and flush `split.json`.
  * merges         — chronological event log (`search` / `merge_skipped` /
    `merge_succeeded`) flushed to `merge.json`.
  * combinations   — per sub-group trace of *why* it was cut into these
    spreads with these photos, flushed to `combinations/<group>.json`.

Grouping collection is gated on `CONFIGS['save_files']['groups']` and the
spread-split collection on `CONFIGS['save_files']['combinations']`; when off,
the record/flush calls are cheap no-ops so they can be left in the pipeline.
"""

from utils.stages_recorder.context import (
    set_is_artificial_time,
    set_general_time_index,
    get_is_artificial_time,
)
from utils.stages_recorder.photo_records import PHOTO_RECORD_COLUMNS, photos_to_records
from utils.stages_recorder.time_utils import (
    build_general_time_to_clock,
    hhmmss_from_iso,
)
from utils.stages_recorder.subgroups import (
    snapshot_subgroups,
    save_subgroups_snapshot,
)
from utils.stages_recorder.splits import (
    SplitRecorder,
    build_split_notes,
)
from utils.stages_recorder.merges import (
    reset_merge_events,
    get_merge_events,
    flush_merge_events,
    new_candidate_details,
    record_search,
    record_skip,
    record_succeeded,
)
from utils.stages_recorder.combinations import (
    is_enabled as combinations_recording_enabled,
    reset_output_dir as reset_combinations_dir,
    build_subgroup_record,
    reset_subgroup_records,
    stash_subgroup_record,
    save_subgroup_record,
)

__all__ = [
    'set_is_artificial_time',
    'set_general_time_index',
    'get_is_artificial_time',
    'PHOTO_RECORD_COLUMNS',
    'photos_to_records',
    'build_general_time_to_clock',
    'hhmmss_from_iso',
    'snapshot_subgroups',
    'save_subgroups_snapshot',
    'SplitRecorder',
    'build_split_notes',
    'reset_merge_events',
    'get_merge_events',
    'flush_merge_events',
    'new_candidate_details',
    'record_search',
    'record_skip',
    'record_succeeded',
    'combinations_recording_enabled',
    'reset_combinations_dir',
    'build_subgroup_record',
    'reset_subgroup_records',
    'stash_subgroup_record',
    'save_subgroup_record',
]
