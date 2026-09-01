# The Substage Pipeline

Branch `image_selection_2` breaks the Read and Selection stages into named,
replaceable substages that share one data-transfer object. This document is the
map: what the pieces are, what contract they honour, and how to swap one out.

It supersedes §3 and §4 of `docs/pipeline_overview.md`.
`docs/image_selection_deep_dive.md` still describes what the selection
*algorithm* does — that behaviour is unchanged.

---

## 1. Why

Two of the four stages were monoliths:

- **Read** did not only read. Inside `get_info_protobufs` and `read_messages`,
  protobuf decoding was interleaved with content classification, a CLIP
  projection against a query bank, bride/groom identity resolution, timestamp
  normalisation, and two event detectors. You could not replace the classifier
  without editing the file that fetches blobs.
- **Selection** was one ~600-line loop that scored, thresholded, narrowed,
  branched eleven ways per category, deduplicated and allocated spreads. You
  could not change the dancing rule without reading all of it.

Neither could be tested or replaced a piece at a time. That is the problem this
branch fixes; the algorithms themselves are untouched.

---

## 2. The unified data transfer

Everything moves inside one object, `src/pipeline/contracts.AlbumContext`.

| Field | What it holds |
|-------|---------------|
| `photos` | The canonical photo table, one row per image |
| `request`, `project_url`, `project_id`, `available_photo_ids` | The parsed request |
| `hints` (`AiHints`) | `aiMetadata`: picked photos, people, focus, subjects, density |
| `designs` (`DesignSpec`) | Layout data |
| `facts` (`GalleryFacts`) | `is_wedding`, `is_artificial_time`, `model_version`, `bride_id`, `groom_id` |
| `clip_embeddings`, `ratings`, `social_circles`, `person_details`, `is_in_vector_db` | Ingest sidecars |
| `selection_inputs`, `selection_plan` | Selection working state |
| `selection` (`SelectionOutcome`) | Chosen photo ids, spread budget, lookup table |
| `services` (`Services`) | Injected Mongo/Qdrant clients |
| `error`, `diagnostics` | Outcome and the execution trace |
| `message` | The `ptinfra` transport object — the boundary, see §6 |

Photo-table column names are constants on `contracts.Col`, so a rename is one
edit and substages never spell a column inline.

### Requirement tokens

A substage declares its contract as a set of strings in two namespaces:

- `photo:<column>` — must exist on `ctx.photos`
- `ctx:<field>` — must be set on the context

The runner checks `requires` before a substage runs and `provides` after. A
replacement that forgets to produce something fails at the boundary with a clear
message instead of corrupting a later stage.

```python
class SemanticTagsSubStage(SubStage):
    name = "enrich.semantic_tags"
    requires = frozenset({photo(Col.EMBEDDING), photo(Col.MODEL_VERSION)})
    provides = frozenset({photo(Col.IMAGE_QUERY_CONTENT), photo(Col.IMAGE_SUBQUERY_CONTENT)})

    def execute(self, context):
        context.photos = add_semantic_tags(context.photos, context.logger)
        return context
```

The base class handles timing, requirement checks, error containment and the
diagnostics record. Subclasses implement `execute`, plus `applies_to` when the
substage is conditional. `optional = True` downgrades a failure to a warning.

---

## 3. The substages

### Ingest — reads bytes, organises them, never infers

| Name | Does | Provides |
|------|------|----------|
| `ingest.request` | Validate the envelope, parse `aiMetadata` | `ctx:project_url` |
| `ingest.design` | Design/layout data from the request or its blob | — |
| `ingest.rating` | The user's rating list | — |
| `ingest.project_registry` | Mongo: `isInVectorDatabase`, `imageModelVersion` | — |
| `ingest.embeddings` | CLIP vectors from Qdrant (skipped when not vectorised) | — |
| `ingest.gallery_assets` | Decode six protobufs into the photo table | `image_id`, `embedding`, `model_version`, `image_class`, `cluster_label`, `cluster_class`, `ranking`, `image_order`, `persons_ids`, `image_time`, `image_color`, `image_orientation`, `scene_order` |
| `ingest.merge_ratings` | Join ratings onto the table | — |
| `ingest.scenes` | Gallery scene ordering | — |

### Enrich — everything the read stage used to derive

| Name | Infers | Provides |
|------|--------|----------|
| `enrich.gallery_type` | Wedding or not | — (`facts.is_wedding`) |
| `enrich.content_class` | `cluster_class` int → category name | `cluster_context` |
| `enrich.identities` | Which identity is the bride, which the groom | `bride_id`, `groom_id` |
| `enrich.semantic_tags` | CLIP projection against the query bank | `image_query_content`, `image_subquery_content` |
| `enrich.require_cluster_data` | (hygiene gate) drop rows without cluster data | — |
| `enrich.people_cluster` | People-composition key | `people_cluster` |
| `enrich.temporal` | Usable timeline, artificial-time detection | `image_time_date`, `general_time` |
| `enrich.ceremony_kiss` | The "may kiss bride" moment | — |
| `enrich.parents` | Couple-with-parents portraits | `parent_category` |

`enrich.content_class` and `enrich.identities` are wedding-only, matching the
original: non-wedding galleries never get a `cluster_context` column.

### Select

| Name | Does |
|------|------|
| `select.route` | Manual vs AI; resolves the pool, lookup table, tag bins and ratings |
| `select.budget` | Focus profile → per-category photo and spread allowance |
| `select.pick` | The category loop; delegates each category to a strategy |
| `select.publish` | Narrow the photo table, finalise the outcome |

### Category strategies — the second level inside `select.pick`

The driver owns what is common to every category: scoring, threshold gating,
honouring the user's own picks, temporal narrowing, the colour split and the
greyscale top-up. What differs per category is a `CategoryStrategy`.

| Strategy | Categories |
|----------|-----------|
| `BridePrepStrategy` | `bride getting dressed`, `getting hair-makeup` |
| `CoupleTimelineStrategy` | `bride`, `groom`, `bride and groom`, `bride party`, `groom party`, `full party`, `walking the aisle`, `first dance`, `cake cutting`, `ceremony`, `dancing` |
| `PersonCoverageStrategy` | `portrait`, `very large group`, `speech` |
| `ParentsPortraitStrategy` | `parents portrait` |
| `ContentClusterStrategy` | everything else (fallback) |

A strategy receives a fully prepared `CategoryRequest` and returns a
`CategoryPicks`. `accessories` and `wedding dress` are resolved in the driver
rather than as strategies, because they are decided before the pool is narrowed.

---

## 4. Replacing a substage

Three ways, in increasing scope.

**One category's rule**, leaving everything else alone:

```python
from src.pipeline import build_select
from src.pipeline.select.strategies import default_registry

registry = default_registry().set("dancing", MyDancingStrategy())
pipeline = build_select(logger, options={"select.pick": {"strategies": registry}})
```

**One substage, at a call site**:

```python
pipeline = build_read(logger)
pipeline.replace("enrich.semantic_tags", MyTagger())
```

**One substage, globally** — by code or by config:

```python
from src.pipeline import override
override("enrich.semantic_tags", MyTagger())
```

```python
CONFIGS['pipeline_overrides'] = {
    'enrich.semantic_tags': 'my_package.taggers:MyTagger',
}
```

`Pipeline` also supports `insert_after`, `remove`, `describe()` (prints every
slot's contract) and `unsatisfied()` (static ordering check).

---

## 5. Running one

```python
from src.pipeline import AlbumContext, Services, build_read, build_select

services = Services(project_status_collection=collection, qdrant_client=qdrant)
context  = build_read(logger).run(AlbumContext.from_message(msg, logger, services))
context  = build_select(logger).run(context)
message  = context.sync_to_message()
```

Both pipelines log a trace of every substage — status, duration, row count in
and out — which is the fastest way to see where a gallery lost its photos:

```
read trace:
  [ok ] ingest.request                 0.000s
  [ok ] ingest.gallery_assets          4.118s 0->2431
  [ok ] enrich.semantic_tags          11.402s 2431->2402
  [ok ] enrich.require_cluster_data    0.031s 2402->2388
  ...
```

---

## 6. The `message` boundary

`ProcessStage` and `ReportStage` are not decomposed yet and still read
`message.content[...]`. `AlbumContext.sync_to_message()` writes exactly the keys
they expect, so they are untouched by this branch.

Between Read and Selection the context is parked on the message and picked back
up by `AlbumContext.for_message()`, so nothing is serialised or rebuilt. When a
message arrives without one — a hand-built message, a test, older code — the
context is rehydrated from `message.content` instead, so a stage can still run
on its own.

When Process/Report are migrated, the `message` field goes away and nothing else
changes.

---

## 7. What changed in existing files

| File | Change |
|------|--------|
| `utils/read_protos_files.py` | `get_info_protobufs` split into `load_gallery_assets` (pure read) + `classify_gallery_type`, `add_content_class`, `resolve_bride_groom`, `add_semantic_tags`, `require_cluster_data`, `add_people_cluster`. `get_info_protobufs` remains as their composition, in the original order, with its original signature |
| `src/request_processing.py` | `read_messages` is now a thin driver over the ingest + enrich pipeline. Same signature, same `(messages, error)` contract. Helper functions unchanged |
| `main.py` | `SelectionStage.get_selection` is a thin driver over the select pipeline |
| `process_gallery.py` | `get_selection` — previously a drifted near-copy of the service's — now drives the same pipeline |

Nothing was deleted. `src/selection/ai_wedding_selection.py` is untouched and
serves as the reference implementation the equivalence tests compare against.

---

## 8. Tests

```
python tests/test_selection_equivalence.py   # or: python -m pytest tests/ -v
python tests/test_pipeline_contracts.py
```

`test_selection_equivalence.py` runs the untouched monolith
(`smart_wedding_selection`) and the decomposed substages over the same synthetic
gallery and asserts the chosen photos, their order, and the spread budget are
identical — across five request shapes (no hints, people + picked photos,
ratings, artificial time, low density) and four gallery seeds.

`test_pipeline_contracts.py` guards the structure: every named substage
resolves, no pipeline has an ordering violation, unmet requirements and broken
`provides` contracts fail loudly, the context round-trips through a message,
and — the one that matters most over time — **no ingest substage may declare a
derived column**, so the reading/inferring split cannot quietly erode.

---

## 9. Deliberate behaviour differences

The refactor is behaviour-preserving except in three places, all of which turn a
crash or a divergence into the sane result:

1. **`portrait` no longer depends on a leaked variable.** In the monolith
   `bride_id`/`groom_id` were assigned inside the couple-timeline branch and
   read by the portrait branch, working only because pandas visits `bride`
   before `portrait` alphabetically. A gallery with no couple-timeline category
   raised `NameError` and failed the whole selection. Strategies read the ids
   from their own pool, which is the same value in every non-crashing case.
2. **`process_gallery.py` manual path** used `'Other'` where the service used
   `'other'` when widening the lookup table. Both now use the service's spelling.
3. **`process_gallery.py` first-page couple subset** was taken from the
   pre-selection frame where the service used the post-selection one. Both now
   use the service's.
4. **An empty photo table reports instead of raising.** `SelectionStage` used to
   `raise` if the table was empty before the availability filter, but set
   `content['error']` and continue if it was empty after it. Both paths now
   report the error, which is the treatment the second case already had. The
   first case is unreachable in practice — the read pipeline already fails a
   message whose photo table is empty.

One quirk was deliberately **preserved**: when `parents portrait` has an odd
remaining need, the monolith fell through without assigning its ranked list and
silently reused the previous category's. `CategoryPicks(preferred=None)` plus
`WeddingPicker._carry_over` reproduces that exactly. It is commented at both
ends; deleting the carry-over is the fix when you want it.

The other quirks catalogued in `docs/image_selection_deep_dive.md` §8 are all
still present and still behave the same way.
