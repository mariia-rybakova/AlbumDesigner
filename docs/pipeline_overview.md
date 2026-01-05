# AlbumDesigner Pipeline Overview

This document maps the control flow, classes, methods, and major decision points that drive the AlbumDesigner service. It traces execution from `main.py` through every stage involved in processing an album request.

---

## 1. Entry Point (`main.py`)

| Component | Purpose | Key Conditions |
|-----------|---------|----------------|
| `main()` | Instantiates `MessageProcessor` and calls `run()` | None |
| `MessageProcessor.run()` | Loads hosting settings, initializes PTInfra, fetches secrets, selects Azure queues, instantiates stages, and starts them | `PTEnvironment` controls queue selection (`dev`, `production`, or custom prefix) |
| Queue Selection | Uses `CONFIGS['collection_name']` and `CONFIGS['visibility_timeout']` to configure `MessageQueue` / `RoundRobinReader` | In `dev`, reads from both `dev` and `dev3`; in production, round-robins `prod` and `ep`; otherwise uses prefix directly |

The runtime seeds `numpy`, sets `PYTHONHASHSEED`, and ensures `ConfigServiceURL` points to dev when unset.

---

## 2. Stage Architecture

All stages inherit from `ptinfra.stage.Stage`, run single-threaded (`batch_size=1`, `max_threads=1`), and exchange messages via `MemoryQueue`s:

1. `ReadStage` → reads and enriches incoming messages.
2. `SelectionStage` → chooses photos via AI or manual overrides.
3. `ProcessStage` → clusters, crops, and lays out spreads.
4. `ReportStage` → posts success/error payloads to Azure Queues and deletes source messages.

Each stage logs timing into `read_time_list`, `processing_time_list`, or `reporting_time_list` for later reporting.

---

## 3. Read Stage (`ReadStage.read_messages` → `src/request_processing.read_messages`)

### Initialization
- Connects to MongoDB using `CONFIGS['DB_CONNECTION_STRING_VAR']`, stores `project_status_collection`.
- Creates a Qdrant client targeting `CONFIGS['QDRANT_HOST']` (`port=6333`, `grpc_port=6334`).

### Message Flow
1. **Abort Handling**: if `AbortRequested`, exits early.
2. **Validation**: ensures each message contains `photos` and `base_url`.
3. **Design Info (`read_layouts_data`)**:
   - If `designInfo` is `None`, fetches JSON from blob storage via `PTFile` using `designInfoTempLocation`.
   - Populates `message.pagesInfo`, `message.designsInfo`, `album_ar`, and layout DataFrames (any/first/last page).
   - Ensures `anyPage` designs exist; otherwise marks error.
4. **Project Metadata**: retrieves `isInVectorDatabase` and `imageModelVersion` from Mongo; if true, `fetch_vectors_from_qdrant` scrolls the configured collection and builds a CLIP embedding DataFrame.
5. **Gallery Protobufs (`get_info_protobufs`)**: returns a `gallery_info_df`, `is_wedding` flag, social circle/person records, and optionally an error.
6. **Enrichment Pipeline**:
   - `add_scenes_info`: merges Pic-Time gallery scene ordering (skips filenames containing underscores or non-numeric IDs).
   - `process_gallery_time`: normalizes timestamps, produces `is_artificial_time`.
   - `identify_kiss_ceremony`: marks ceremony spreads as `"may kiss bride"` when SAT rules hold (valid start/end, same day, officiant anchor within ±6 minutes).
   - `identify_parents`: re-labels portrait clusters as `"parents portrait"` when couples and age tolerances match social-circle data.
7. **Packing**: attaches `gallery_photos_info`, `is_wedding`, and `is_artificial_time` to `message.content`. Errors bubble up and abort the batch.

---

## 4. Selection Stage (`SelectionStage.get_selection`)

### Manual Path
- Triggered when `aiMetadata` is missing or lacks `photoIds`.
- `photos` list from the message is merged with `gallery_photos_info` to keep metadata in sync.
- Sets `manual_selection=True` and, if wedding, clones `wedding_lookup_table` while forcing `'Other'` and `'None'` to `(24, 4)` spreads.

### AI Path
1. Filters `gallery_photos_info` by `available_photos` when provided; stores a copy as `gallery_all_photos_info`.
2. Extracts AI hints:
   - `ten_photos`, `people_ids`, `focus`, `tags`, `density`, `is_artificial_time`.
3. LUT Handling:
   - For weddings, copies `wedding_lookup_table` and scales spread limits via `CONFIGS['density_factors'][density]`, capping at 24.
4. Calls `ai_selection`:
   - Loads tag embedding bins via `load_pre_queries_embeddings` (prefers blob paths, falls back to `/files/pre_queries/v1|v2`).
   - Routes to `smart_wedding_selection` (returns `(photo_ids, spreads_dict, errors)`) or `smart_non_wedding_selection`.
5. On success: shrinks `gallery_photos_info` to the selected IDs, stores `spreads_dict`, and fills `photos`. Bride/groom subsets are cached for first-page generation when requested.
6. Any errors attach `message.content['error']` for downstream reporting.

---

## 5. Processing Stage (`ProcessStage.process_message`)

1. **Preprocessing**:
   - Updates image ranks via `update_photos_ranks` (AI selections get `image_order=0`).
   - Launches a separate `multiprocessing.Process` running `process_crop_images`, which computes normalized crops based on segmentation, face masks, and aspect ratio heuristics.
2. **Temporal Ordering**:
   - Sorts by `image_order` descending, then calls `generate_time_clusters` to build coherent sequences.
   - `generate_first_last_pages` determines special spreads and returns `first_last_pages_data_dict`.
3. **Album Assembly (`album_processing`)**:
   - Chooses grouping strategy (`get_wedding_groups` vs `get_none_wedding_groups`), then maps each group to its image IDs.
   - Derives or reuses a lookup table (`get_lookup_table` or `modified_lut`), adjusts with `update_lookup_table_with_layouts_size`, and enforces spread limits (`update_lookup_table_with_limit`).
   - Wedding-specific cleanup: `process_wedding_illegal_groups` adjusts groups and LUT entries (respects `manual_selection`).
   - Iterates groups: each calls `process_group`, which sorts images, converts them to `Photo` objects, splits oversized groups (`get_group_photos_list`), and generates spreads via `generate_filtered_multi_spreads`.
     - If no spreads fit, the code retries with reduced parameters (0.8 → 0.2 multipliers) or injects a dummy `Photo` to force coverage.
     - `add_ranking_score` ranks candidate spreads; `assign_photos_order` assigns `left/right` box sets.
   - Results are time-sorted (`sort_groups_by_time`) for weddings; non-wedding groups remain as-is.
4. **Cropping Merge**:
   - Waits up to 200 seconds for the cropping process; failures terminate the subprocess and raise.
   - Merges cropping coordinates into `gallery_photos_info` and first/last page DataFrames.
5. **Output Construction (`assembly_output`)**:
   - Builds `result_dict` with `compositions`, `placementsImg`, `placementsTxt`, and metadata (`userJobId`, `projectId`, etc.).
   - Adds cover, first page, each group spread, and last page when applicable; mirrored designs (negative IDs) are handled via `get_mirrored_boxes`.
   - `customize_box` computes per-photo crops, distinguishing square boxes from adaptive aspect-ratio crops.
6. The final dict is stored on `message.album_doc` for the Report stage.

---

## 6. Reporting Stage (`ReportStage`)

- `push_report_msg` / `push_report_error` marshal JSON payloads, gzip + base64 encode them, and send to the reply queue named by `message.content['replyQueueName']`. Queue creation is attempted idempotently.
- After posting, the stage deletes the original queue message via `Message.delete()` and updates timing metrics.
- Every three reports (default), `print_time_summary` logs averages for read, processing, reporting, and overall durations calculated from the accumulated timing lists.

---

## 7. Key Decision Points & Conditions

| Area | Condition | Impact |
|------|-----------|--------|
| Stage Abort | Incoming object is `AbortRequested` | Stage stops processing the batch |
| Design Info | Missing `designInfo` but `designInfoTempLocation` provided | Fetches JSON via `PTFile`; failure aborts message |
| Mongo `isInVectorDatabase` | `True` | Triggers CLIP embedding fetch from Qdrant; failures mark message error |
| AI Metadata Presence | Missing `aiMetadata` or `photoIds` | Switches to manual selection, marks `manual_selection=True` |
| Wedding Flag | `is_wedding=True` | Enables LUT scaling, wedding group logic, parent detection tweaks, first/last page extras |
| Density Factor Missing | `density` not in `CONFIGS['density_factors']` | Defaults LUT multiplier to 1 |
| Cropping Timeout | Subprocess fails to return within 200s | Process terminated; exception raised to error queue |
| Spread Generation Failure | `generate_filtered_multi_spreads` yields `None` repeatedly | Retries with reduced params; may add dummy `Photo`; final failure skips group |
| Report Errors | `message.error` populated | Sends error payload via `push_report_error`; otherwise sends album document |

---

## 8. File Reference Summary

- `main.py`: Stage orchestration, queue setup, reporting helpers.
- `src/request_processing.py`: Message enrichment, layout ingestion, assembly output helpers.
- `src/selection/auto_selection.py`: Entry point for AI/manual selection paths.
- `src/album_processing.py`: Grouping, lookup table management, spread generation.
- `src/core/*.py`: Photo data structures, spread combinatorics, scoring heuristics.
- `src/smart_cropping.py`: Foreground/face-aware cropping executed in a subprocess.

This document should serve as the starting point for deeper dives into specialized modules (e.g., `src/selection/ai_wedding_selection.py`, `utils/time_processing.py`) whenever additional behavior details are required.

