# Image Selection Deep Dive

This guide documents the **Selection stage** of AlbumDesigner — the step that turns a
whole gallery (hundreds to thousands of photos) into the ~100–200 photos that will
actually be laid out into the album, together with the per-category spread budget that
Album Processing consumes.

It complements `docs/pipeline_overview.md` (§4) and `docs/album_processing_deep_dive.md`
(the stage that runs immediately after).

---

## 1. Where the stage lives

| Layer | File | Role |
|-------|------|------|
| Stage wrapper | `main.py` → `SelectionStage.get_selection` (`main.py:159`) | Reads `aiMetadata`, decides manual vs AI, scales the LUT by density, writes results back onto the message |
| Router | `src/selection/auto_selection.py` → `ai_selection` | Loads tag embedding bins, dispatches wedding vs non-wedding |
| Wedding engine | `src/selection/ai_wedding_selection.py` → `smart_wedding_selection` | Budget allocation + scoring + per-category selection (the bulk of the logic) |
| Non-wedding engine | `src/selection/ai_non_wedding_selection.py` → `smart_non_wedding_selection` | People-cluster based proportional selection |
| Diversity / dedup | `utils/selection/refactoring.py` → `select_remove_similar` | Time/scene grouping + greedy cosine-diverse picking |
| People coverage | `src/selection/person_clustering.py` → `person_max_union_selection` | Max-coverage of distinct identities for portrait-like categories |
| Cluster round-robin | `utils/selection/wedding_selection_tools.py` → `get_clusters`, `select_non_similar_images` | Default fallback strategy |
| Temporal filtering | `utils/selection/time_orientation_selection.py` → `identify_temporal_clusters` | Drops temporally isolated photos |
| Non-wedding helpers | `utils/selection/non_wedding_selection_tools.py` | Quota math, per-cluster picking, fallbacks |
| Tunables | `utils/configs.py`, `utils/lookup_table_tools.py`, `files/focus_csv.csv` | Weights, thresholds, per-category priors |

### Dead / legacy modules

These are **not** reachable from the live path and should not be trusted as
documentation of current behaviour:

- `src/selection/image_selection_scores.py` — the old multiplicative score
  (`class × similarity × person × image_order × tags`). Still described by the top-level
  `README.md`, but nothing imports it. The live scorer is `get_scores` in
  `ai_wedding_selection.py` and it is a **weighted sum of min-max normalized scores**.
- `utils/selection/filtering_selection.py` — the previous home of `select_remove_similar`;
  the import at `ai_wedding_selection.py:16` is commented out in favour of
  `utils/selection/refactoring.py`.
- `utils/selection/time_orientation_selection.py` — only `identify_temporal_clusters`
  is used. `select_images_by_time_and_style`, `filter_similarity`, `allocate_by_size` are orphaned.
- `person_clustering.person_clustering_selection` (agglomerative/Jaccard) — superseded by
  `person_max_union_selection`.
- `wedding_selection_tools.get_possible_image_sums` / `allocate_images_to_categories`
  (plus `CONFIGS['total_target_images']`, `min_images_per_category`,
  `spreads_required_per_category`, `priority_categories`) — an older budget allocator,
  replaced by `calculate_optimal_selection`.

---

## 2. Inputs

### 2.1 The message

`SelectionStage` consumes the message produced by `ReadStage`:

| Key | Type | Meaning |
|-----|------|---------|
| `gallery_photos_info` | `DataFrame` | One row per gallery photo, fully enriched (schema below) |
| `photos` | `list[int]` | Photo ids the user made *available*. Empty ⇒ whole gallery |
| `aiMetadata` | `dict` or `None` | The AI hints. `None`, or `photoIds is None`, switches to the manual path |
| `rating` | `list[{photoId, rating}]` | Optional per-photo user rating |
| `is_wedding` | `bool` | Set by `check_gallery_type` during Read |
| `is_artificial_time` | `bool` | `True` when EXIF timestamps failed `check_time_correctness`; selection then groups by scene instead of time |
| `pagesInfo` | `dict` | Used only to decide whether to cache a "bride and groom" subset for the first page |

`aiMetadata` fields (`main.py:201`):

| Field | Default | Used for |
|-------|---------|----------|
| `photoIds` | — | The "ten photos" the user hand-picked. Drives the similarity and class scores, and are force-included per category |
| `personIds` | `[]` | Identities the user wants to see |
| `focus` | `['everyoneElse']` | Which column of `files/focus_csv.csv` / which `relations` table to use. One of `brideAndGroom`, `parents`, `everyoneElse` |
| `subjects` | a 26-tag default list | Tag cloud; each tag resolves to a `.bin` of pre-computed CLIP query embeddings |
| `density` | `3` | 1–5. Scales photos-per-spread via `CONFIGS['density_factors'] = {1:0.5, 2:0.75, 3:1, 4:1.5, 5:2.0}` |

### 2.2 `gallery_photos_info` schema (columns selection actually reads)

Built in `utils/read_protos_files.get_info_protobufs` by outer-joining five protobufs
(plus CLIP vectors from Qdrant when `isInVectorDatabase`), then enriched in
`src/request_processing.py`.

| Column | Source | Notes |
|--------|--------|-------|
| `image_id` | all | Join key |
| `embedding` | `ai_search_matrix.pai` or Qdrant | **L2-normalized** CLIP vector (512-d v1 / 768-d v2). Rows with NaN embedding are dropped |
| `model_version` | same | 1 or 2 — picks the `pre_queries` folder and the queries `.pkl` |
| `image_class`, `cluster_label`, `cluster_class`, `ranking`, `image_order` | `content_cluster.pb` | `ranking` = `selectionScore` (higher is better, already 0–1). `image_order` = `selectionOrder` (**lower is better**) |
| `cluster_context` | `map_cluster_label(cluster_class)` via `process_content` | The category name driving everything below. `-1`/out-of-range ⇒ `"None"`; `two brides`/`two grooms` collapse to `bride and groom` |
| `image_query_content`, `image_subquery_content` | `utils/image_queries.generate_query` | Best-matching tag / sub-query text for the image, searched **within** `cluster_context` when that context exists in the queries pkl |
| `persons_ids` | `persons_info.pb` | List of identity ids; coerced to `[]` when missing |
| `main_persons` | same | Used to resolve bride/groom |
| `bride_id`, `groom_id` | derived in `get_info_protobufs` | Most frequent identity in `cluster_context == 'bride'` / `'groom'`, preferring members of `main_persons`; several NaN fallbacks |
| `n_faces`, `faces_info` | `ai_face_vectors.pb` | |
| `number_bodies`, `bodies_info` | `ai_person_vectors.pb` | |
| `image_color` | `bg_segmentation.pb` (`colorEnum`) | `0` = grayscale |
| `image_orientation` | same | `landscape` if `aspectRatio >= 1` else `portrait` |
| `image_as`, `background_centroid`, `diameter` | same | Consumed by cropping, not selection |
| `image_time`, `general_time` | `process_gallery_time` | `image_time` is converted to `image_time_date` (a real timestamp) inside selection |
| `scene_order`, `scene_name` | `add_scenes_info` / `add_scene_info` | Pic-Time gallery scene ordering — the substitute for time when `is_artificial_time` |
| `parent_category` | `identify_parents` | `"bride and groom with parents"` / `"bride with her parents"` / `"groom with his parents"`. Rows that get one also have `cluster_context` rewritten to `"parents portrait"` |
| `user_rating` | optional | Alternative to the `rating` list |

---

## 3. Control flow

```
SelectionStage.get_selection
├── aiMetadata missing / photoIds None ──► MANUAL PATH
│     merge message['photos'] into gallery_photos_info
│     is_wedding ⇒ modified_lut = wedding_lookup_table with 'other'/'None' → (24, 4)
│     manual_selection = True                                    (no scoring at all)
│
└── AI PATH
      df ← gallery_photos_info restricted to message['photos'] (if any)
      gallery_all_photos_info ← df.copy()          # kept for later stages
      is_wedding ⇒ modified_lut = wedding_lookup_table with photos-per-spread
                    scaled by density_factor, clamped to [1, 24]
      ai_selection(...)
        ├── wedding      ► get_tags_bins  ►  smart_wedding_selection
        └── non-wedding  ►                   smart_non_wedding_selection
      gallery_photos_info ← df filtered to the selected ids
      photos, spreads_dict, min_total_spreads, max_total_spreads written to message
      pagesInfo['firstPage'] ⇒ cache message['bride and groom'] subset
```

Any non-`None` `errors` from `ai_selection` sets `message.content['error']` and the
message is passed straight to the Report stage.

### 3.1 Tag bins (`auto_selection.get_tags_bins` / `load_pre_queries_embeddings`)

Each `subjects` entry is mapped through `CONFIGS['bin_name_dictionary']`
(camelCase → file name, e.g. `weddingDress` → `wedding_dress`) and read as a binary blob:

- v1 model: `pictures/photostore/4/pre_queries/<tag>.bin`
- v2 model: `pictures/photostore/32/pre_queries/v2/<tag>.bin`
- On any blob failure it falls back to the local `files/pre_queries/v1|v2/` copy.

Layout: `<int32 dim><int32 n>` header then `dim*n` little-endian floats, reshaped to
`(n, dim)` — one row per phrasing of the tag.

Result is `{tag: matrix}`. An all-blank `subjects` list yields `[]`, which disables the
tag score entirely.

---

## 4. The wedding engine (`smart_wedding_selection`)

Signature:

```python
smart_wedding_selection(original_big_df, user_selected_photos, people_ids, focus,
                        tags_features, density, is_artificial_time, logger, rating=None)
  -> (selected_ids, spreads_allocation, min_total_spreads, max_total_spreads, error)
```

It runs in three phases: **budget → scoring → per-category picking**.

### 4.1 Category taxonomy used by the picker

```python
orientation_time_categories = {bride, groom, bride and groom, bride party, groom party,
                               full party, walking the aisle, first dance, cake cutting,
                               ceremony, dancing}
persons_categories          = {portrait, very large group, speech}
parents_categories          = {parents portrait}
# special-cased inline:      accessories, wedding dress,
#                            bride getting dressed, getting hair-makeup
# everything else            → default cluster round-robin
```

### 4.2 Phase 1 — Budget: how many photos per category

#### 4.2.1 Focus table

`load_event_mapping(CONFIGS['focus_csv_path'])` parses `files/focus_csv.csv` into
`{focus_column: {event: {'type', 'value'}}}`. Cell values become:

| Cell | `type` | `value` |
|------|--------|---------|
| `8%` | `percentage` | `8.0` |
| `yes` / `no` | `yes` / `no` | the string |
| numeric | `numeric` | float |

Sub-event names are stripped of surrounding quotes but **not** lower-cased, so they must
match `cluster_context` exactly.

The chosen column is `focus[0]`, defaulting to `brideAndGroom` when the focus is unknown
or `focus` is empty. `relations[focus[0]]` is looked up alongside it — but note
`calculate_optimal_selection` takes it as `image_lookup_table` and **never uses it**;
`relations` currently has no effect on selection.

#### 4.2.2 Total-spread envelope (`define_min_max_spreads`)

A coarse three-bucket rule over gallery size, identity count and how many of the focus
table's percentage events actually exist in the gallery:

```
events_ratio = (#percentage events present in the gallery)
             / (#percentage events absent or zero-valued)

images ≤ 600  and people ≤ 30  and ratio ≤ 0.5  →  (15, 18)
images ≥ 2000 and people ≥ 140 and ratio ≥ 0.8  →  (23, 26)
otherwise                                        →  (19, 22)
```

`min_total_spreads` becomes `TARGET_SPREADS`; both bounds ride along on the message for
Album Processing.

#### 4.2.3 Allocation (`calculate_optimal_selection`)

1. **Scale the LUT.** `modified_lut[event] = (clamp(photos_per_spread × density_factor, 1, 24), std)`
   from `wedding_lookup_table`. (Same scaling `SelectionStage` applies to the LUT it puts
   on the message — computed twice, independently.)

2. **Percentage events.**

   ```
   spreads = value / Σvalues × TARGET_SPREADS
   photos  = spreads × modified_lut[event].photos_per_spread
   miss        = max(0, photos − n_actual)          # demand we cannot satisfy
   miss_spreads = round(miss / photos_per_spread)
   if miss > 0:  photos = n_actual ; spreads = round(photos / photos_per_spread)
   over_photos  = max(0, n_actual − photos)         # surplus supply
   over_spreads = over_photos / photos_per_spread
   ```

3. **`yes` / `no` events.** Treated as "one token photo": `spreads = 0`, `photos = 1`,
   dropped to `0` if the gallery has none. Note `no` behaves identically to `yes` here —
   the `no` cells in the CSV are effectively inert.

4. **Rebalance.** While there is both unmet demand and surplus supply
   (`total_over_spreads > 1 and total_miss_spreads > 1`), walk the events and move one
   spread's worth of photos from surplus to the events that have it, one pass at a time,
   stopping when no event has `over_spreads > 1`.

5. **Emit.** For every category actually present in the gallery:
   `selections[event] = round(photos)` (the per-category **photo quota**, `need`) and
   `spreads[event] = spreads` (the per-category **spread budget**, returned as
   `spreads_dict`).

If any exception is raised the function returns `(None, None, None, None)` and
`smart_wedding_selection` aborts with `"No images got selected!"`.

### 4.3 Phase 2 — Scoring

Scoring is **per category** (each `cluster_context` group is scored and normalized
independently), and is skipped entirely when the user supplied no hints.

#### 4.3.1 Raw component scores (`calculate_scores`)

| Score | Rule | No-data fallback |
|-------|------|------------------|
| `person_score` | `1.0` if `people_ids` is empty; else `#selected_people_in_image / #people_in_image`, bumped to `0.99` for a solo shot of a selected person | `CONFIGS['person_score']` = `1e-7` |
| `similarity_score` | **max** cosine similarity to any of the user's `photoIds` (within the whole gallery, not just this category) | `5e-8` if the user picked nothing; `CONFIGS['similarity_score']` = `1e-5` if no embedding |
| `class_matching_score` | fraction of the user's picks sharing this image's `image_class`, bucketed: `≥0.8→1.0`, `≥0.5→0.9`, `≥0.3→0.7`, `>0→0.5`, `0→0.1` | `1.0` when the user picked nothing; `CONFIGS['class_matching_penalty']` = `1e-3` if the column is missing |
| `tags_score` | best `tag_matrix @ embedding` over all tags (max over phrasings, then max over tags) | `1` when `tags` is empty; `2e-7` when the dict is empty |

#### 4.3.2 Rating

`rating` arrives as `[{photoId, rating}, …]` and is turned into a dict. Precedence:
request `rating` → `user_rating` column rescaled by
`CONFIGS['user_rating_max_scale']` (`value / max_scale * 5`) → `0`.

#### 4.3.3 Normalization and the weighted sum (`get_scores`)

Every component is min-max normalized **within the category**; a flat column
(`max − min < 1e-9`) collapses to a constant `0.5`.

```python
total_score = w_class      × class_score_norm
            + w_similarity × similarity_score_norm
            + w_person     × person_score_norm
            + w_tags       × tags_score_norm
            + w_rank       × ranking                # raw protobuf selectionScore, NOT normalized here
            + w_rating     × rating_score_norm
```

Weights come from `CONFIGS['weights']`, plus a `rating` key injected at runtime, and are
divided by their sum:

| Key | Raw | Effective (÷ 1.7) | Used in the formula? |
|-----|-----|-------------------|----------------------|
| `person` | 0.4 | 23.5% | yes |
| `user_rating` | 0.3 | 17.6% | **no** |
| `rating` | 0.3 (injected) | 17.6% | yes |
| `class` | 0.2 | 11.8% | yes |
| `similarity` | 0.2 | 11.8% | yes |
| `rank` | 0.2 | 11.8% | yes |
| `tags` | 0.1 | 5.9% | yes |

Two footnotes:

- `CONFIGS['weights']` ships the rating key as **`user_rating`**, but `get_scores` reads
  `weights['rating']` and injects its own default of `0.3` when that key is absent — which
  it always is. Both keys therefore sit in `total_weight`, making the denominator `1.7`
  instead of `1.4`. The weights that actually contribute sum to `1.4/1.7 ≈ 0.82`, so
  `total_score` can never reach 1, and every weight is ~18% smaller than the config
  implies. (The table in the top-level `README.md` quotes the intended `1.4`
  denominator, not the real one.) Renaming the config key to `rating` changes all scores.
- `image_order_score_norm` (`1/(image_order+ε)`, normalized) is computed but unused;
  `ranking` is used instead.

#### 4.3.4 Candidate gating (`get_candidate_images`)

Two modes:

**Unscored mode** — `people_ids`, `photoIds` and `tags_features` are all empty.
`total_score` is set to `1.0` for every row, ordering falls back to `image_order`
ascending, and `no_selection = True` propagates that "sort by image_order, not score"
choice through the rest of the category.

**Scored mode** — keep images with `total_score > selection_threshold[cluster_name]`
(per-category floors in `utils/configs.py`, ranging from `0.005` for `groom` to `0.5`
for `accessories`). Then:

```python
if len(candidates) < len(scored_df):
    candidates = scored_df.head(need * 3)   # top-3×quota by score
```

Because the threshold filter almost always removes *something*, this fallback fires in
practice on nearly every category, and the threshold table mostly serves as a
"did anything at all pass?" check rather than a real filter.

If scoring fails or every score is `≤ 0`, the category is skipped.

### 4.4 Phase 3 — Per-category picking

`original_big_df.groupby('cluster_context')` (pandas sorts group keys, so categories are
visited alphabetically). Per category, with `need = images_allocation[cluster_name]`:

#### Early exits

1. `n_actual ≤ 2 and need == 1` → take the first row as-is, no scoring, no filtering.
2. `need == 0` → skip.
3. Empty candidate list → skip.

#### Forced inclusions

- **`accessories`, `wedding dress`**: user picks win outright (`user_selected_ids[:need]`);
  otherwise top-`need` candidates. No diversity pass.
- **All other categories**: every user-picked photo in the category is added
  unconditionally (not capped at `need`) and removed from the candidate pool.
  `images_allocation[cluster_name] -= len(user_selected_ids)` is executed, but `need` was
  already read into a local before that line — so **user picks are additive on top of the
  full quota**, not deducted from it.

#### Temporal narrowing

```python
min_keep = ceil(need * 3)
valid = scored_df                                       if len(scored_df) < min_keep
        identify_temporal_clusters(scored_df, 'image_time_date', 20, 4)   otherwise
```

`identify_temporal_clusters` builds a graph where photos within **20 minutes** are
connected, takes connected components, and **discards any component smaller than 4
photos** — i.e. it throws away temporally isolated one-offs. It can return an empty frame,
in which case the category is skipped.

`time_clusters_fixed_span(df, minutes=4)` then stamps `sub_group_time_cluster`: a new
cluster starts whenever a photo is more than 4 minutes after the *start* of the current
cluster, and singleton clusters are absorbed into their neighbour.

#### Color policy

The pool is split on `image_color`: `!= 0` is color, `== 0` is grayscale.

- **No color at all** → take at most 2 grayscale by `image_order` ascending, and stop.
- Otherwise the strategies below run **on color only**, and grayscale is used at the end
  purely to top up a shortfall (§4.5).

#### If supply ≤ demand

`has ≤ need` short-circuits the strategies: de-duplicate by
`(tuple(persons_ids), image_subquery_content)` keeping the best `image_order`, take
`head(need)`, done.

#### Strategy A — `bride getting dressed`, `getting hair-makeup`

1. Keep rows whose `image_subquery_content` contains `"bride"` (case-insensitive).
2. Find the most frequent identity across those rows and keep only images containing it.
3. If that leaves `≤ need` (or within 1 of it), take the top `need` by score /
   `image_order`; otherwise run `select_remove_similar`.
4. If step 1 matched nothing, fall back to the raw color pool ordered by `image_order`.

#### Strategy B — `orientation_time_categories`

Identity filters, then a safety net, then diversity:

| Category | Filter |
|----------|--------|
| `bride` | `persons_ids == [bride_id]` exactly (solo bride) |
| `groom` | `persons_ids == [groom_id]` exactly — **unless** the pool is smaller than `2 × need`, in which case no filter |
| `bride and groom` | contains both ids **and** (`n_faces == 2` or `number_bodies == 2`) |
| `bride party` | contains `bride_id` |
| `groom party` | contains `groom_id` |
| `walking the aisle` | see below |
| `full party`, `first dance`, `cake cutting`, `ceremony`, `dancing` | no identity filter |

`walking the aisle` is a scripted narrative: the **first** bride+groom photo is taken
outright, then the **latest** bride-without-groom photo (the bride arriving at the altar).
Each consumes one unit of `need`, and both are removed from the pool.

**Over-filter recovery** (important — it is what keeps sparse galleries from collapsing):

```
remaining == 0                    → take the first half of the unfiltered color pool
filtered_out > 0.8 × remaining    → add back the first  remaining//2  rejected rows
```

Then order by `total_score` desc (or `image_order` asc in unscored mode).

`dancing` gets one extra rule: prefer `image_orientation == 'landscape'`, and only pad
with portraits if landscapes cannot cover `need`.

Finally `select_remove_similar` picks the diverse subset.

#### Strategy C — `persons_categories` (`portrait`, `very large group`, `speech`)

For `portrait` only, first try the two "formal" sub-queries
(*formal studio-style wedding portrait…* / *formal family portrait…*) restricted to
images containing both bride and groom. If that yields nothing, fall back to the whole
color pool; if it yields fewer than `need`, top up with two "informal" sub-queries, and
failing that with any bride+groom image.

Then `person_max_union_selection`: greedily pick the image whose identity set adds the
most **new** people to the running union (resetting the union when nothing new can be
added), then drop any pick whose embedding has cosine `> 0.95` with an already-kept pick,
processing larger groups first.

> `bride_id` / `groom_id` are read here but are only ever *assigned* inside Strategy B.
> They survive because alphabetical group ordering puts `bride` before `portrait`. A
> gallery with no `orientation_time_categories` group reaching Strategy B will raise
> `NameError`, which the outer `try` converts into a whole-selection failure.

#### Strategy D — `parents portrait`

1. Take up to 2 (or 1 when `need < 2`) `"bride and groom with parents"` images,
   preferring *different* parent identity sets.
2. If the remaining need is **even**, pair `"bride with her parents"` with
   `"groom with his parents"` shots taken within **±10 minutes** of each other, greedily
   by nearest timestamp, so the album gets balanced facing pages.

> If the remaining need is **odd**, no branch assigns `preferred_color_ids`, and the
> variable retains its value from the *previous* category iteration. This is a live bug.

#### Strategy E — default

`get_clusters` buckets the color pool by `cluster_label` (content clustering), and
`select_non_similar_images` round-robins across buckets — buckets ordered by their best
image, images inside a bucket ordered by score — until `need` is met.

### 4.5 `select_remove_similar` (the shared diversity pass)

`utils/selection/refactoring.py`. Three steps:

1. **Group.**
   - `is_artificial_time == True` → group by `scene_order` × `cluster_label`
     (keys `SCN_<scene>__CL_<label>`), because timestamps are untrustworthy.
   - Otherwise → `build_time_merged_groups`: order the `sub_group_time_cluster` groups by
     time, drop singletons into a `_SINGLES_` bucket, and merge consecutive groups whose
     gap is `≤ 5 minutes`.

2. **Allocate** (`allocate_prefer_larger_artificial`): 1 photo per group first
   (largest-first if `need` < number of groups), then `+1` round-robin over the largest
   groups until `need` is met or every group is at capacity. `_SINGLES_` always gets 0.
   One special rule: **≤ 3 groups, all of size ≤ 4, and `need > 4`** ⇒ exactly one per
   group and stop, deliberately returning fewer than `need`.

3. **Pick with a cosine guard.** Within each group, walk images best-score-first
   (`total_score` desc, or `image_order` asc when the frame is unscored) and keep an image
   only if its cosine similarity to *every* already-selected image is `< 0.90`.

`allocate_prefer_larger` (the 2-per-group, half-capacity variant) is defined in the same
file but is not called.

### 4.6 Color top-up and finalisation

```python
if len(preferred_color_ids) >= need:
    selection = preferred_color_ids[:need]
else:
    selection = preferred_color_ids
    gap = need - len(preferred_color_ids)
    # fill from grayscale:
    #   len(grayscale) <= gap → take all
    #   gap == 1              → single best by total_score
    #   otherwise             → select_remove_similar(grayscale, gap)
```

Across the whole run, `category_picked` tracks `{category: {actual, selected}}` for
logging, and the final return de-duplicates while preserving order via
`list(dict.fromkeys(...))`.

---

## 5. The non-wedding engine (`smart_non_wedding_selection`)

Much simpler, and driven by **people composition** rather than event categories.

1. `len(df) ≤ CONFIGS['small_gallery_number']` (15) → return the whole gallery.
2. Build `people_cluster` per image via `generate_dict_key(persons_ids, number_bodies)`
   → `"No PEOPLE"`, or `"<count>_person_<ids>"` / `"<count>_pple_<ids>"`.
3. Compute each people-cluster's share of the gallery. Count how many exceed
   `CONFIGS['person_count_percentage']` (0.20).
4. Exactly one dominant cluster → `select_images_of_one_person`; otherwise
   `select_images_of_group`.

**Target count** (`calculate_required_images`) is a clamped linear ramp:
`10 + (30−10)/150 × total_images`, bounded to `[10, 30]`.

**`select_images_of_one_person`** samples from the head / middle / tail of the pool
(`select_random_image`), accepts at most one image per `cluster_label`, and refuses a
grayscale image when a color image of the same label is already in. Any shortfall is
filled by picking, per over-subscribed label, the image whose embedding is *farthest*
(cosine) from the centroid of what was already chosen from that label.

**`select_images_of_group`** first drops people-clusters that hold exactly one portrait
image (no layout exists for that), then runs
`proportional_selection_with_calculation`: per people-cluster quota
`ceil(share × required)`, `select_n_images` picks first/middle/last indices with unique
`cluster_label`s, with the same grayscale rule and the same farthest-from-centroid
fallback.

Non-wedding selection returns no spread allocation — `spreads_dict` stays `{}`.

---

## 6. Outputs

| Message key | Produced by | Consumed by |
|-------------|-------------|-------------|
| `photos` | selected ids | Process stage, `assembly_output` |
| `gallery_photos_info` | `df` narrowed to the selection | Album Processing |
| `gallery_all_photos_info` | pre-selection copy | first/last page generation, key pages |
| `spreads_dict` | `spreads_allocation` from `calculate_optimal_selection` | Album Processing spread budgeting |
| `min_total_spreads` / `max_total_spreads` | `define_min_max_spreads` | Album Processing global cap |
| `modified_lut` | `SelectionStage` (density-scaled LUT) | `album_processing` |
| `manual_selection` | manual path only | disables some illegal-group enforcement downstream |
| `bride and groom` | color subset when `pagesInfo['firstPage']` | first-page generation |
| `error` | any failure | Report stage |

---

## 7. Configuration reference

All in `utils/configs.py` unless noted.

| Key | Value | Effect |
|-----|-------|--------|
| `weights` | see §4.3.3 | Score blend (note the `user_rating` vs `rating` key mismatch) |
| `user_rating_max_scale` | `5` | Divisor for the `user_rating` column |
| `density_factors` | `{1:0.5, 2:0.75, 3:1, 4:1.5, 5:2.0}` | Multiplies photos-per-spread |
| `selection_threshold` | per-category, `0.005`–`0.5` | Candidate floor (largely bypassed, §4.3.4) |
| `person_score` / `similarity_score` / `class_matching_penalty` | `1e-7` / `1e-5` / `1e-3` | Missing-data penalties |
| `small_gallery_number` | `15` | Non-wedding "take everything" cutoff |
| `person_count_percentage` | `0.20` | Non-wedding "dominant person" threshold |
| `ε` | `1e-9` | Flat-column guard in normalization |
| `focus_csv_path` | `files/focus_csv.csv` | Per-focus category percentages |
| `bin_name_dictionary` | camelCase → file name | Tag `.bin` resolution |
| `wedding_lookup_table` | `utils/lookup_table_tools.py` | `(photos_per_spread, std)` per category |
| `relations` | `utils/configs.py` | Loaded per focus but **currently unused** |
| Hard-coded | `20 min` / `min 4` (temporal clusters), `4 min` (fixed span), `5 min` (merge gap), `0.90` (dedup cosine), `0.95` (portrait dedup cosine), `±10 min` (parent pairing), `need × 3` (candidate fallback and `min_keep`) | Not exposed in config |

---

## 8. Known quirks and failure modes

| # | Where | Behaviour |
|---|-------|-----------|
| 1 | `get_scores` | Weight key mismatch (`user_rating` in config vs `rating` in code) inflates the normalization denominator to 1.7; only 82% of the weight mass is actually applied |
| 2 | `smart_wedding_selection` main loop | User picks are added *on top of* `need`, not deducted, so a category can overshoot its quota |
| 3 | Strategy C | `bride_id`/`groom_id` leak from Strategy B via loop-variable scope; `NameError` if no Strategy B category runs first |
| 4 | Strategy D | Odd `remaining_need` leaves `preferred_color_ids` holding the previous category's result |
| 5 | `get_candidate_images` | The `len(candidates) < len(scored_df)` fallback fires whenever any image is below threshold, silently replacing threshold filtering with "top 3×need" |
| 6 | `calculate_optimal_selection` | `relation_table` is accepted and ignored; `no` cells in `focus_csv.csv` behave the same as `yes` |
| 7 | `n_actual ≤ 2 and need == 1` | Takes `cluster_df` row order, which is the DataFrame's arbitrary order — not score, not `image_order` |
| 8 | LUT density scaling | Done independently in both `SelectionStage` and `calculate_optimal_selection` — they can drift |
| 9 | `identify_temporal_clusters` | Silently returns an empty frame on error, which reads downstream as "no images for this category" |
| 10 | `allocate_prefer_larger_artificial` | The "≤3 groups, all ≤4, need>4" rule intentionally under-delivers |
| 11 | Exception handling | Almost every failure is caught, logged, and turned into an empty/partial result rather than an error — a broken category simply contributes nothing |
| 12 | `README.md` | Still documents the dead multiplicative scorer; §4.3 here supersedes it |

---

## 9. Worked example

Wedding, 1,400 photos, 60 identities, `focus = ['brideAndGroom']`, `density = 3`,
6 user picks, 3 `personIds`, default tag list, EXIF time valid.

1. `define_min_max_spreads` → not small, not large ⇒ `(19, 22)`; `TARGET_SPREADS = 19`.
2. `bride and groom` has `12%` of the `brideAndGroom` column, whose values sum to `107`
   ⇒ `spreads = 12/107 × 19 ≈ 2.13`; the LUT says 4 photos/spread at density 3
   ⇒ `photos ≈ 8.5`. The gallery has 140 such photos, so `miss = 0`,
   `over_photos ≈ 131.5`, `over_spreads ≈ 32.9` — a big surplus that the rebalance loop
   can lend to starved categories.
3. `need = round(8.5) = 8`. Scoring runs (hints are present); the 140 photos are scored and
   min-max normalized within the category; the top 24 (`need × 3`) survive gating.
4. Temporal clustering drops any isolated shot; `sub_group_time_cluster` splits the rest
   into 4-minute bins, then `build_time_merged_groups` merges bins less than 5 minutes apart.
5. Strategy B keeps only frames where both bride and groom are present with exactly two
   faces or bodies. If that removes more than 80% of the pool, half of the rejects come back.
6. `select_remove_similar` spreads the 9 picks across the merged time groups, refusing any
   image with cosine `≥ 0.90` to an earlier pick.
7. If only 7 color images survive, the 2-photo gap is filled from grayscale via a second
   `select_remove_similar` call.
8. Repeat for every category; de-duplicate; return with
   `spreads_dict = {'bride and groom': 2.28, …}`, `min/max = 19/22`.
