# Album Processing Deep Dive

This guide focuses on `src/album_processing.py`—the stage that turns enriched gallery data into layout-ready spreads. It complements `docs/pipeline_overview.md` by drilling into inputs, lookup-table orchestration, grouping heuristics, and fallback strategies.

---

## 1. Entry Point: `album_processing`

```python
album_processing(df, designs_info, is_wedding, modified_lut, params, logger,
                 density=3, manual_selection=False)
```

| Parameter | Description | Notes |
|-----------|-------------|-------|
| `df` | `pandas.DataFrame` containing `gallery_photos_info` enriched in Read/Selection stages | Must include `image_id`, `cluster_context`, `cluster_label`, `image_time`, `image_as`, `general_time`, `image_color`, etc. |
| `designs_info` | Dict produced in `read_layouts_data` with layout DataFrames and lookup maps | Required keys: `anyPagelayouts_df`, `anyPagelayout_id2data`, `anyPagebox_id2data`, `maxPages` |
| `is_wedding` | Bool flag controlling grouping logic | Determines wedding-specific heuristics, parent detection, and group legality fixes |
| `modified_lut` | Optional LUT injected by Selection stage (manual path or wedding density scaling) | When `None`, LUT is generated from scratch |
| `params` | Tuple influencing spread search (e.g., `[0.01, 100, 1000, 100, 300, 12]`) | Passed downstream into layout ranking/greedy selection code |
| `logger` | PTInfra logger | Required for telemetry and exception traces |
| `density` | AI density hint (default 3) | Guides default LUT sizing when `modified_lut` absent |
| `manual_selection` | Flag toggled when Selection skipped AI | Bypasses some "illegal group" enforcement to respect user choices |

Return value: ordered `list` of spread dictionaries (one entry per logical group) later consumed by `assembly_output`.

---

## 2. Lookup Table (LUT) Pipeline

| Step | Function | Purpose |
|------|----------|---------|
| Derive/Reuse | `modified_lut` or `utils.lookup_table_tools.get_lookup_table(group2images, is_wedding, logger, density)` | Establish baseline `(max_spreads, min_photos_per_spread)` per cluster/context |
| Layout-Aware Limits | `update_lookup_table_with_layouts_size(look_up_table, designs_info['anyPagelayouts_df'])` | Ensures LUT respects available layout capacities (number of boxes per design) |
| Global Caps | `update_lookup_table_with_limit(group2images, is_wedding, look_up_table, max_total_spreads)` | Enforces absolute spread maximum (max of `CONFIGS['max_total_spreads']` and `designs_info['maxPages']`) |
| Post-Processing | Re-applied after group adjustments | Guarantees the LUT remains consistent after merges/splits |

Key considerations:
- LUT entries are context-driven (e.g., "ceremony", "bride and groom"). Wedding flows use `CONFIGS['density_factors']` (Selection stage) or `density` hints to scale target spreads.
- Manual selections with weddings often receive a pre-wired LUT (`modified_lut`) so album processing honors user-defined spread budgets.

---

## 3. Group Discovery & Enforcement

### Initial Grouping
- Weddings: `utils.album_tools.get_wedding_groups(df, manual_selection, logger)` groups by time cluster and context while tracking metadata for later sorting.
- Non-weddings: `get_none_wedding_groups` provides simpler clusters.
- `utils.album_tools.get_images_per_groups` converts group objects into `{group_name: image_ids}` for LUT steps.

### Illegal Group Handling (`process_wedding_illegal_groups`)
Only applied when `is_wedding=True`.

1. **Preconditions**: DataFrame must include `time_cluster`, `cluster_context`, `cluster_label`; missing columns raise `ValueError`.
2. **Metadata Columns**: Adds `group_sub_index`, `group_size`, `merge_allowed`, `original_context`, `groups_merged` to facilitate tracking.
3. **Splitting Large Groups**:
   - `handle_wedding_splitting` flags groups exceeding `CONFIGS['max_img_split']` or LUT-allowable spread sizes.
   - `split_illegal_group_by_time` splits by time; `split_illegal_group_in_certain_point` handles special contexts based on breakpoints returned by `check_time_based_split_needed`.
4. **Bride/Groom Balancing**:
   - `handle_wedding_bride_groom_merge` tries to merge asymmetric "bride" vs "groom" clusters (e.g., bride getting ready ≈ groom suit) while respecting `CONFIGS['max_imges_per_spread']`.
5. **General Merging Loop (`process_wedding_merging`)**:
   - Merges under-filled groups up to iteration cap (`max_iterations=500`), ensuring merged size ≤ `CONFIGS['max_imges_per_spread']`, total spreads ≤ ~2.1, and respecting merge attempt limits (`CONFIGS['merge_limit_times']`).
   - Uses `merge_illegal_group_by_time` to pick candidate mates based on temporal proximity.
6. **Output**: Updated `groups` (pandas GroupBy) and `group2images` map, plus possibly adjusted LUT for downstream steps.

Manual selection path bypasses special/regular splitting to avoid overriding curated sets.

---

## 4. Per-Group Processing (`process_group`)

### Sorting & Photo Extraction
- Weddings: dancing groups sort by `image_as` then `image_time`; others by time alone.
- Creates `Photo` objects via `src.core.photos.get_photos_from_db` (attributes: id, aspect ratio, color, rank, class, cluster label, general time).

### Group Fragmentation
- `get_group_photos_list(cur_group_photos, spread_params, largest_layout_size, logger)` splits large photo lists when `len/photos per spread threshold ≥ 4`:
  - Computes `optimal_spread_param = min(largest_layout_size, spread_params[0])`.
  - Splits into up to `number_of_splits = ceil(len / split_size)` pieces.
  - Each piece is processed as an independent sub-group (append index `*group_idx`).

### Spread Generation
1. `generate_filtered_multi_spreads(group_photos, layouts_df, spread_params, params, logger)` orchestrates combinatorial search:
   - `core.spreads.selectPartitions` enumerates viable partition sizes respecting layout box counts and portrait/landscape capacities.
   - `listSingleCombinations`, `partitions_with_swaps`, and `greedy_combination_search` find candidate photo allocations.
2. If `None` returned (no layout fits), fallback loop scales `spread_params[0]` by 0.8/0.6/0.4/0.2 and retries. Each iteration may re-split groups (`get_group_photos_list`) to reduce complexity.
3. If still failing, injects a dummy `Photo` with outlier rank/aspect ratio to enlarge combinations.

### Ranking & Ordering
- `add_ranking_score(filtered_spreads, sub_group_photos, layout_id2data)` computes layout suitability scores based on context, photo ranks, color, etc.
- Results sorted descending; top spread gets photo objects mapped into left/right boxes.
- `assign_photos_order(best_spread, layout_id2data, design_box_id2data, merge_pages=False)` finalizes per-box ordering and metadata.
- Keys format:
  - Weddings: `"{time_cluster}_{cluster_context}*{group_idx}"`.
  - Non-weddings: `"{time_cluster}*{group_idx}"`.

### Logging
- Each iteration logs processing time, split actions, and fallback usage. Exceptions capture traceback location and return `None` to skip the group.

---

## 5. Output Assembly & Ordering

- After collecting `result_list` entries per group, `album_processing` logs the overall processing duration.
- Weddings call `utils.time_processing.sort_groups_by_time(result_list, logger)` to ensure chronological spread order before returning; non-weddings keep original order.

### Downstream Interface
- `ProcessStage.process_message` receives the list and passes it into `assembly_output`, along with `message.designsInfo` and enriched image metadata.
- `assembly_output` iterates each group dictionary, adds compositions, resolves mirrored designs, and builds placement records.

---

## 6. Related Modules & Call Graph

```
album_processing
├── utils.album_tools.get_wedding_groups / get_none_wedding_groups
├── utils.album_tools.get_images_per_groups
├── utils.lookup_table_tools.{get_lookup_table, update_lookup_table_with_layouts_size, update_lookup_table_with_limit}
├── src.groups_operations.groups_management.process_wedding_illegal_groups
│   ├── handle_wedding_splitting
│   ├── handle_wedding_bride_groom_merge
│   └── process_wedding_merging
├── src.core.photos.get_photos_from_db
├── src.core.spreads.generate_filtered_multi_spreads
│   ├── selectPartitions / listSingleCombinations / greedy_combination_search
│   └── add_ranking_score / assign_photos_order
└── utils.time_processing.sort_groups_by_time
```

---

## 7. Operational Considerations

- **Parameter Tuning**: `params` tuple influences spread filtering thresholds; adjust with caution since `core.spreads` logic assumes ordering (`params[1]` used as divisor for partition weight pruning).
- **Performance**: Large weddings trigger expensive combinatorial searches. Early splitting (`handle_wedding_splitting`) and `largest_layout_size` capping help bound runtime.
- **Error Propagation**: Any raised exception bubbles to `ProcessStage`, triggering error reporting and queue visibility resets. Logging includes filename/line/function for triage.
- **Testing**: Unit coverage should mock `designs_info` and LUT operations to test group branching; integration tests can replay stored `gallery_photos_info` snapshots.

---

## 8. Next Steps

1. Diagram fallback loops (LUT constraints, split/merge, dummy photo injection) for onboarding documentation.
2. Instrument metrics around `handle_wedding_splitting` and `process_wedding_merging` iterations to quantify typical merge counts.
3. Consider exposing `params` via config to allow tuning without code changes.

