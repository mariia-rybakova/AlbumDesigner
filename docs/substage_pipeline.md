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
| `key_pages` (`KeyPages`) | The photos that open and close the album |
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

Ingest decodes through **ptinfra** rather than its own parsing — see §3.1.

### 3.1 What ingest takes from ptinfra

`ptinfra.proto` is the shared protobuf kernel: its `pb/` modules are copied
verbatim from `pic-time/protobufs` at a recorded commit
(`ptinfra.proto.PROTOBUFS_COMMIT`), so every service parses the same schema.
AlbumDesigner used to carry its own copy under `utils/protos/` and its own
decode loop for each file. Both are gone.

| Was | Now |
|-----|-----|
| `utils/protos/*_pb2.py` (7 vendored schema modules) | `ptinfra.proto.pb` |
| `PTFile(...).read_blob()` + `WhichOneof("versions")` + `.v1`, repeated 6× | `ptinfra.read_stage.load_versioned(url, WrapperCls, missing_ok=...)` |
| `PTFile(...).exists()` guards | `missing_ok=True` |
| Hand-rolled `pai2`/`pai3` binary reader in `get_image_embeddings` | `ptinfra.exporter.pai_writer.parse_pai` — the codec the exporter writes with |

That is 411 lines deleted for 40 added, and the DataFrames the readers return
are unchanged (see `tests/test_ingest_readers.py`).

**The vendored copies had already drifted.** Field-by-field, ptinfra's schemas
are ahead by two fields AlbumDesigner never saw:

- `PersonInfo.Identity.PersonInfo.bibNumber`
- `PersonVector.Photo.Body.backboneEmbedding` (the 384-d DINOv3 backbone
  embedding, present from `bodyModelVersion` 3)

**A local copy cannot coexist with ptinfra's.** protobuf keys its global
descriptor pool by `.proto` file name and rejects a second registration whose
serialized bytes differ. Importing both copies of `PersonInfo_pb2` in one
process raises `TypeError: duplicate file name PersonInfo.proto`. So the
migration had to remove `utils/protos/` outright — there is no gradual path.

#### Not adopted yet: the canonical dataclasses

`ptinfra.proto.data_convert` also offers plain dataclasses per blob type
(`FaceVectors`, `BgSegmentation`, `ContentCluster`, ...), reachable through
`BaseReadStage.load_*`, with embeddings already decoded to numpy and bboxes as
`(x1, y1, x2, y2)` tuples. ptinfra's intent is that "consumers never touch
protobuf".

AlbumDesigner still stores raw protobuf objects in the `faces_info` and
`bodies_info` columns and a proto `Point` in `background_centroid`, so adopting
them would touch:

- `src/smart_cropping.py` — `face.bbox.x1..y2` (2 sites), `centroid.x/.y`
  (3 sites)
- `utils/selection/time_orientation_selection.py::_determine_shot_style` —
  `face.bbox.x2 * face.bbox.y2` (dead code, see the deep dive §1)

Worth doing — the dataclasses pickle cleanly across the cropping subprocess
boundary, where protobuf objects currently round-trip through serialization,
and `Body.embedding` gets the right dtype for its `bodyModelVersion`. It is
left out of this change because it alters cropping arithmetic, which the
synthetic-blob tests do not cover.

### Enrich — everything the read stage used to derive

| Name | Infers | Provides |
|------|--------|----------|
| `enrich.duplicate_shots` | One copy of each shot uploaded more than once | — |
| `enrich.gallery_type` | Wedding or not | — (`facts.is_wedding`) |
| `enrich.content_class` | `cluster_class` int → category name | `cluster_context` |
| `enrich.identities` | Which identity is the bride, which the groom | `bride_id`, `groom_id` |
| `enrich.same_sex_couple` | Gives each partner of a same-sex couple their own solo class | — (`facts.same_sex_couple`) |
| `enrich.semantic_tags` | CLIP projection against the query bank | `image_query_content`, `image_subquery_content` |
| `enrich.require_cluster_data` | (hygiene gate) drop rows without cluster data | — |
| `enrich.people_cluster` | People-composition key | `people_cluster` |
| `enrich.temporal` | Usable timeline, artificial-time detection | `image_time_date`, `general_time` |
| `enrich.parents` | Couple-with-parents portraits | `parent_category` |
| `enrich.ceremony_anchor` | The kiss and the send-off, from one shared anchor | `send_off_score` |
| `enrich.key_pages` | Which photo opens the album and which closes it | `key_page` |

`enrich.content_class`, `enrich.identities` and `enrich.key_pages` are
wedding-only, matching the original: non-wedding galleries never get a
`cluster_context` column.

#### `enrich.same_sex_couple`

`map_cluster_label` folds the content model's `two brides` / `two grooms` into
`bride and groom`. That is right for the couple shots and wrong for the solo
ones: the model puts **both** partners' portraits into a single class and
leaves the other empty.

On gallery 52894932 the `bride` class held 68 photos that split exactly
**32 / 32** between the two brides, and `groom` held **none**. That loses a
partner outright, because `CoupleTimelineStrategy` filters the `bride` category
to `persons_ids == [bride_id]` and `groom` to `persons_ids == [groom_id]` — so
the 32 portraits of the second bride sat in a class whose filter rejects them,
while the category that would have taken them had nothing to pick from. Her 12%
of the album bought nothing at all.

The substage moves the second partner's exact solo frames into the empty
counterpart class. Everything downstream then works untouched and already
correctly — two 12% shares in `focus_csv.csv`, the identity filter in the
strategy, the lookup table, the per-class thresholds — and nothing else needs
to know the couple is same-sex, which is the point of doing it here.

Measured on that gallery:

| | before | after |
|---|---|---|
| `bride` / `groom` classes | 68 / **0** | 36 / **32** |
| photos picked for those two categories | 18 / **0** | 15 / **12** |
| solo shots of the first bride in the album | 9 | **16** |
| album length | 95 | **104** |

`resolve_bride_groom` was hardened for the same case. It names the first
partner from the populated solo context, then looked to `main_persons` for the
second — and that list is the model's most-frequent identities and can be
**empty**, as it is on this gallery. When it is, the second partner came out
`NaN`, which is not a loud failure either: the category filter
`persons_ids == [nan]` simply matches nothing, so a partner vanishes silently.
It now falls back to the next most common identity of the context that named
the first, which is by construction the other partner. `main_persons` is still
preferred when it has an answer, and a gallery with only one identity is left
unresolved rather than having a guest promoted into the couple.

Detection is the model's own output — `cluster_class` for `two brides` or
`two grooms`, which is 99 photos here and **zero across the other six
validation galleries**, so there are no false positives to guard against. It
never guesses: an empty counterpart class is the whole signature, so a gallery
where the model filled both is left alone, and a partner whose identity could
not be resolved is recorded but not moved.

**What it does not fix.** A quarter to a third of *every* album's budget weight
is allocated to categories the gallery has none of — 23–33% on the six
opposite-sex galleries, and 51% here. That is a separate, general problem in
`budget_each`, where `total_value` sums the whole profile rather than the
present categories. It cannot be fixed by simply dropping the absent share,
though: on a same-sex gallery `groom` is exactly such a category, and
redistributing its 12% elsewhere is what would leave the second partner
unrepresented.

#### `enrich.duplicate_shots`

A photographer often uploads a second copy of their best frames with a
different treatment — black and white, or a blue or brown tone. Gallery
53273032 is **1028 photos that are really 514 shots, each uploaded twice**, and
one album spread showed the same dance frame in colour and in black and white.

**It is a judgement about the gallery, not about a pair of photos.** Nothing in
the photo table tells a re-export from the next frame of a burst:

| | |
|---|---|
| CLIP cosine, same shot colour vs grey | 0.777 – 0.930 |
| CLIP cosine, *different* shots, same treatment | 0.870 – 0.954 |
| composition (centroid, diameter, `n_faces`) | not pixel-deterministic — a confirmed twin differs as much as a burst pair, and one same-second pair gave `n_faces` 37 vs 67 |

An identical capture second plus aspect ratio does hold for all 512 pairs — a
re-export preserves the EXIF — but on its own it is far too eager: a camera at
three frames a second makes several genuinely different photos in one second,
and pair-by-pair that rule wanted to drop **46, 32 and 11 real frames** from the
other galleries.

What marks a duplicated gallery is the regularity. Photos sitting in duplicate
`(capture second, aspect ratio)` groups:

| gallery | share | dropped |
|---|---|---|
| 53273032 | **~100%** | 517 |
| 49994361 | 3.8% | 0 |
| 49995684 | 3.1% | 0 |
| 53496523 | 2.1% | 0 |
| 47981912 | 0% | 0 |
| 53147741 | <1% | 0 |

Two orders of magnitude of daylight, so `min_gallery_share` is 0.5 and the rule
fires only on a systematically duplicated gallery. The colour flag is never
consulted, so a toned copy is caught as readily as a grey one; the copy kept is
the best-ranked (`image_order`, where 0 is best). A gallery with unusable EXIF
needs no special case — everything lands in one enormous group, and only groups
small enough to be a re-upload set are counted (`max_copies_per_shot`).

**Why it runs first, before anything counts the gallery.** Tried inside
`select.pick` instead, the budget sizes the album against a supply twice as
large as it really is, every category then runs out of distinct photos, and the
album fell from 100 photos to 86. Removed up front, the counts are simply right:
`select.budget` allocates against 511, and the layout cannot pad a group with a
twin either. The album goes 100 → 85 photos, but 15 of those 100 were second
copies, so the distinct content is unchanged.

The limitation worth knowing: a photographer who re-uploads only a handful of
favourites in black and white is **not** caught. That is the direction to err in
— doing nothing leaves one redundant spread, while guessing wrong deletes
photos the album should have had.

#### `enrich.ceremony_anchor`

Two detectors sharing one reading of the ceremony, because two detectors
deriving the climax independently can disagree on the same gallery.

The anchor is the **median** of the climax frames (vows, rings, kiss) inside the
ceremony core (p5–p95 of ceremony positions). Median, not last: the subquery
classifier scatters stray "exchanging vows" labels minutes late — on the
validation galleries the last climax frame sits 85, 171 and 72 positions after
the median, far enough to put the search window past the event it is meant to
find.

Everything works in **positions** (rank by `general_time`), not wall-clock
minutes. That is the only axis that survives galleries with unusable EXIF — one
validation gallery has 2 distinct `image_time` values across 528 photos.

| | reads the anchor as | evidence |
|---|---|---|
| `may kiss bride` | a **centre** — the kiss is itself a climax signal, so it cannot anchor on itself | subqueries the query bank already carries |
| `bride walking the aisle` / `groom walking the aisle` | an **upper bound** — the processional is before the ceremony *starts*, so the bound is the ceremony core start, not the anchor | identity is mandatory; subquery and concept only rank |
| `send off` | a **lower bound** — guests shower the couple as they *leave* | a CLIP concept bank, because nothing else sees it |

A `walking the aisle` photo that sits **after the ceremony centre** is
reclassified as `other` before any of this runs. Walking in happens before the
ceremony begins, so the class cannot be right once it is under way — what the
classifier is looking at there is the recessional, the couple walking back out.
The cut is the anchor rather than the ceremony's end, which is the conservative
line: it leaves alone anything between the ceremony starting and its climax,
where a late arrival really might still be walking in. Only `cluster_context` is
rewritten; `image_class` is the model's own output and enrich does not edit it,
which is also what keeps the send-off detector working, since it reads the
per-photo label. Measured across six galleries it fires on two, demoting 2 and 5
photos, with the kept and demoted position ranges cleanly separated (148–167
against 382–383, and 75–90 against 138–316).

The processional deliberately does **not** key on the `walking the aisle` class.
That label is sparse — 1, 9, 5 and 24 photos on the validation galleries — and
the groom almost never gets it, because he is waiting at the altar rather than
walking. Of the photos the detector tags, only 6 of 16 carried it; the rest came
from `groom party`, `ceremony`, `groom`, `bride and groom` and `other`. What
does the discriminating is **identity**: the bride present and the groom absent,
or the reverse. The couple walking in together is neither of them walking in.

Indications rank rather than gate, because the query bank has no phrase at all
for the groom walking to the altar — but a floor still applies, or a gallery
with no processional tags its prep portraits instead. Real processionals score
0.44–0.60; the false positives that floor removes scored 0.19–0.29.

The send-off needs a burst of ≥5 **and** visual confirmation; sequence alone
cannot separate it from the plain recessional, and visual evidence alone picks
the wrong event (on one gallery the highest-scoring photo is a couple portrait
session with bubbles two hours later).

Supersedes `enrich.ceremony_kiss`, which anchored on the last "officiant leading
wedding ceremony" frame within a real-timestamp window, ±6 minutes. That
combination tagged **1 photo across 3,361 in four galleries**: two were rejected
outright by its SAT gate for unusable EXIF, and on a third the kiss frames sat
11 minutes from the officiant anchor. The same four galleries now yield 3, 5, 0
and 1.

Shared machinery lives in `src/pipeline/enrich/timeline.py` — ordering, the
ceremony core, the anchor, concept scoring and burst grouping — so the
walking-the-aisle detector can reuse it rather than fork it.

#### `enrich.key_pages`

Which photo opens the album and which closes it. Choosing a cover is an
inference about photos, not a step of page layout, but it has always run inside
ProcessStage — `src/core/key_pages.py::generate_first_last_pages`, wedged
between time clustering and the layout search.

Weddings only, for now, alongside the other substages built around the couple.
ProcessStage keeps handling non-wedding galleries.

**This is the first half of moving it.** The substage calls the same function
ProcessStage calls, so there is one implementation of the rule and no chance of
the two drifting. The only difference is the pool: enrich runs before selection,
so it sees the whole gallery instead of the few hundred frames selection kept.
ProcessStage still runs its own copy and still decides the covers — nothing
downstream reads `key_page` or `ctx.key_pages` yet.

Each cover is drawn from **a quarter of the candidates by count**, taken in
time order — the opening from the earliest quarter of them, the closing from the
latest — and the best photo in that quarter wins. Three rules have been tried
and the first two both failed on real albums:

- **`time_cluster` min/max** (what ProcessStage used). The couple frames a
  wedding actually yields are often bunched into one part of the day, so
  `min(time_cluster) == max(time_cluster)` and the album opened and closed on
  two shots of the same moment. On the reviewed album all ten landscape couple
  photos sat in **cluster 1 of 2**.
- **The first and last ten photos** (what enrich used, having no
  `time_cluster`). Closer, but still biased to the extreme edge, and on one
  gallery it put the closing photo at 30% of the day against an opening at 23%.
- **A quarter of the elapsed time.** Breaks whenever the gallery is not one
  continuous session. Gallery 52894932 holds two shoots **five days apart** — a
  single **114.7-hour gap** between consecutive photos — so a quarter of its
  time span contained **88% of the photos**, the window was effectively the
  whole gallery, and the opening cover came from **79% of the way through the
  day** while good frames sat in the first fifth.

Counting is immune to all three. It is also the rule the rest of the pipeline
already follows: `enrich/timeline.py` works in **positions** rather than
wall-clock minutes for exactly this reason — ordering by time is trustworthy,
measuring distances along it is not. Nothing is trimmed by anything but time
order inside the quarter; which frame is *good* is for the ranking to say.
`COVER_FRACTION` is the knob.

The window is taken along `general_time`, not `image_time`: the two are the same
seconds when the EXIF is trustworthy, but when it is not, `general_time` has
been rebuilt from scene order into a synthetic monotonic day while `image_time`
still holds the unusable original — two validation galleries carry 2 distinct
`image_time` values across 528 and 582 photos.

Both callers now take the same path, so enrich and ProcessStage agree exactly:

| gallery | opening | closing |
|---|---|---|
| 52894932 | 18% *(was 79%)* | 90% |
| 53147741 | 48% | 81% |
| 49994361 | 28% | 99% |
| 49995684 | 23% | 71% |
| 47981912 | 24% | 95% |
| 53496523 | 47% | 64% |

**The ranking direction was also inverted.** `image_order` is the content
model's `selectionOrder`, a rank where **0 is best** — `update_photos_ranks`
sets a hand-picked photo to 0, and the selection stage sorts it ascending for
the same reason. Both cover rules sorted it *descending*
(`_select_by_priority_from_subset`) and took an `argmax` over the normalised
rank (`_pick_most_dissimilar`), so each was choosing the worst-ranked candidate
of every tie it broke. Fixed in both places.

The remaining property of the wider pool, still true:

- **The pool is not quality-filtered.** Selection is what removes the weak
  frames. Over the full gallery a mediocre early couple shot competes on equal
  terms with a good one, separated only by `image_order` and only after the
  subquery priority has already tied. That is why `KeyPages` holds ranked lists
  rather than two ids — the consumer resolves them against the photos it has.

One thing the wider pool exposed that is the rule's own, not the wiring's:
`_pick_most_dissimilar` ranks the closing candidates by rating (0.6) and
dissimilarity to the opening (0.4), with **no recency term at all**. On
49995684 the last-ten window correctly spans to 75% of the day, and the rule
still picks a frame at 30% because it rates higher. Worth revisiting when the
rule itself is, rather than here.

One thing found while testing the rule, not addressed here because the
non-wedding path is out of scope for now: `choose_good_non_wedding_images`
splits its picks down the middle and is asked for one, so the opening half is
always empty — **non-wedding albums get a closing photo and never an opening
one.**

### Select

| Name | Does |
|------|------|
| `select.route` | Manual vs AI; resolves the pool, lookup table, tag bins and ratings |
| `select.budget` | Focus profile → per-category photo and spread allowance; settles the ceremony's `yes` classes |
| `select.preselect` | The constraints: photos the album is committed to before any ranking |
| `select.pick` | The category loop; delegates each category to a strategy |
| `select.publish` | Narrow the photo table, finalise the outcome |

#### `select.preselect` — the constraints

Some photos are in the album because something decided so before any ranking
happened. Running them through the picker either adds nothing or risks losing
them, so they are settled between `select.budget` and `select.pick`. Four kinds,
honoured in this order — the user's own intent first, so a later rule finds its
slot filled rather than competing for it:

| constraint | what it commits | why the picker was the wrong place |
|---|---|---|
| `user_picks` | every `aiMetadata.photoIds` photo still in the pool | the picker honoured only the ones that survived scoring and the candidate cut first; a hand-picked photo could score below the floor and be dropped |
| `identities` | the best photo of each `aiMetadata.personIds` identity | `personIds` only fed `person_score`, so a requested person could be ranked up everywhere and appear in nothing |
| `key_pages` | the opening and closing photos | they were chosen over the whole gallery, and ProcessStage takes its covers from the *selected* pool — nothing guaranteed they survived |
| `yes_categories` | each `yes` category's whole allowance | a `yes` category is promised one photo if the thing happened: no allowance to divide, nothing to weigh |

Each is switchable under `CONFIGS['preselect']`, and
`photos_per_identity` (default 1) sets how deep identity coverage goes —
guaranteeing a named person appears, not saturating the album with them.

**Only `yes` categories are charged.** There the commitment *is* the allowance,
so it is zeroed and the picker skips the category. The other three are added to
the budget. Charging them was tried first and made the album *shorter* rather
than more certain: a committed photo is usually one the picker would have chosen
anyway, so charging its category costs a second photo for nothing — on the
equivalence fixture, charging identity coverage lost a `walking the aisle` frame
both paths had already selected. Uncharged, a constraint costs a slot only when
it actually adds a photo, which is the rule the monolith already applied to
hand-picked photos.

Where a choice remains, the ranking is the one the picker would have used: the
request's own scoring, falling back to `image_order` ascending. Note the
direction — `image_order` is the content model's `selectionOrder`, a rank where
**0 is best**, which is why `update_photos_ranks` sets a hand-picked photo to 0.

Measured against the monolith on the equivalence fixture's five scenarios:

| constraints | effect |
|---|---|
| all off | **identical on all five** — the substage is a true no-op, so every departure below is the constraint and not drift |
| `user_picks` only | 4–7 photos swapped, album length unchanged |
| `identities` only | +0 to +1 photos — it adds one exactly when a requested identity had none |
| `yes_categories` only | 1–4 photos swapped, +0 to +2 net where a `yes` category had been getting nothing |
| all on | +0 to +3 net |

The swaps are inherent to hoisting a decision earlier: a committed photo leaves
the frame the strategy diversifies over, so the remaining slots fill differently.

#### `select.budget` and the ceremony's `yes` classes

The arithmetic is a step-for-step port of `calculate_optimal_selection`, now in
`src/pipeline/select/allocation.py` as named steps rather than one pass. The
original stays where it is, untouched, as the oracle: a test asserts the two
agree exactly on galleries where the new rule cannot fire, so the rest of the
budget tests measure the rule and not a drifting reimplementation.

Three of the four classes `enrich.ceremony_anchor` produces carry `yes` in
`focus_csv.csv` rather than a percentage. In the original arithmetic that means
one photo if the moment happened, no page of its own — and a *surplus* the
fill-up loop may draw on. That last part is the problem. A send-off is a burst,
fifteen to twenty-one frames against a lookup-table base of two, so its surplus
reads as seven to ten spare pages. On the validation galleries the loop happens
to exhaust the shortfall before reaching it — `send off` is near the bottom of
`focus_csv.csv` and the loop walks the file in order — but nothing was holding
it there.

So the group is settled before redistribution and then taken out of it:

| | the album is short of pages | the album is full |
|---|---|---|
| **2 or more present** | they supply exactly **one** of the missing pages, one frame each; the shortfall drops by one and the loop fills the rest from elsewhere | no page; their photos come out of **ceremony's** own allowance, so the album does not quietly grow |
| **fewer than 2** | no page — one special moment is a photo, not a spread | as above |

`CONFIGS['ceremony_yes_min_classes']` is the threshold. Membership of the group
is read from the focus profile rather than fixed in code: give `send off` a
percentage in `focus_csv.csv` and it becomes an ordinary category again. That is
also why `may kiss bride` is *not* in the group — it already carries 2–3%.

##### Two rounds of filling

The fill loop hands one page to each category with a spare page's worth,
walking `focus_csv.csv`, and goes round again until the album is full. **A
`yes` category sits out the first round.** `yes` means one photo if the thing
happened, and that is what it is worth while the album can still be built from
the categories the profile actually weighted; only once a full walk of those
has failed to fill the album is a `yes` category worth a page of its own.

Without that it competed on the first walk like anything else, and on a gallery
short of material the album filled with whatever sat high in the file. Measured
on 53273032 after de-duplication took it to 511 photos:

| | before | after |
|---|---|---|
| `settings` | 5 photos (a full page, granted on pass 1) | 1 |
| `food` | 5 photos of the 6 it had | 1 |
| pass 1 grants | 9, two of them to `yes` categories | 7, all weighted |

The ceremony highlights are out of *both* rounds — `settle_ceremony_yes` zeroes
their surplus, because a send-off burst is large enough to fill several pages on
its own.

**And a page of a `yes` category is now a small page.** The second round still
grants one at whatever size `wedding_lookup_table` says, and those sizes were
written for spreads the profile actually weights — `food` and `settings` at 4
photos a page, `pet` at 4, `rings` and `suit` at 3. All fourteen string-valued
categories now carry **(2, 1)**, matching what `send off` and the two
processionals already had. On the degenerate case where nothing weighted can
fill, the same twelve granted pages cost **49 photos before and 25 after**. A
structural test pins the pair for every `yes` row in the profile, so adding one
without a matching lookup entry fails.

**What this exposes rather than fixes:** the walk order is the row order of
`focus_csv.csv`, which is not a priority. `other` (242 photos available,
budgeted **0%**) and `None` (15 available, **0%**) sit at rows 3 and 4, ahead of
`ceremony`, `bride and groom` and `speech`, and each took two pages of that
shortfall. Holding the `yes` categories back sends *more* of the shortfall their
way. Two categories the profile gives zero weight should not be near the front
of the queue; fixing that means walking the table in a deliberate order, and is
a separate change.

Measured on the four validation galleries, all of which are 5–7 pages short, so
all of which take the fill-up branch:

| gallery | highlights present | photos before → after |
|---|---|---|
| 49994361 | bride aisle, groom aisle, send off | 113 → 109 |
| 49995684 | bride aisle, groom aisle, send off | 115 → 111 |
| 47981912 | bride aisle, groom aisle | 83 → 81 |
| 53496523 | bride aisle, groom aisle | 109 → 107 |

The album stays the same length — one page still unfilled in both — and the
count drops because the page the group now occupies holds one frame of each
moment where the category it displaced would have held four.

Worth knowing: **the "charge it to the ceremony" branch fired on none of the
four.** Real weddings are structurally short against this profile, because it
budgets a percentage to categories a gallery often has none of (`couple`,
`kiss`, `entertainment`, the three parent groupings). If that is not intended,
it is a bigger question than this substage.

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
| `utils/read_protos_files.py` | `get_info_protobufs` split into `load_gallery_assets` (pure read) + `classify_gallery_type`, `add_content_class`, `resolve_bride_groom`, `add_semantic_tags`, `require_cluster_data`, `add_people_cluster`. `get_info_protobufs` remains as their composition, in the original order, with its original signature. Every reader now decodes through ptinfra (§3.1) |
| `utils/protos/` | Deleted — replaced by `ptinfra.proto.pb` (§3.1) |
| `src/smart_cropping.py` | Its two `utils.protos` imports repointed at `ptinfra.proto.pb` |
| `src/request_processing.py` | `read_messages` is now a thin driver over the ingest + enrich pipeline. Same signature, same `(messages, error)` contract. Helper functions unchanged |
| `main.py` | `SelectionStage.get_selection` is a thin driver over the select pipeline |
| `process_gallery.py` | `get_selection` — previously a drifted near-copy of the service's — now drives the same pipeline |

Nothing was deleted. `src/selection/ai_wedding_selection.py` is untouched and
serves as the reference implementation the equivalence tests compare against.

---

## 8. Tests

```
python -m pytest tests/ -v          # or run each file directly
```

`test_selection_equivalence.py` runs the untouched monolith
(`smart_wedding_selection`) and the decomposed substages over the same synthetic
gallery, **with `select.preselect` switched off**, and asserts the chosen photos,
their order, and the spread budget are identical — across five request shapes (no hints, people + picked photos,
ratings, artificial time, low density) and four gallery seeds.

`test_pipeline_contracts.py` guards the structure: every named substage
resolves, no pipeline has an ordering violation, unmet requirements and broken
`provides` contracts fail loudly, the context round-trips through a message,
and — the one that matters most over time — **no ingest substage may declare a
derived column**, so the reading/inferring split cannot quietly erode.

`test_dedupe.py` covers `enrich.duplicate_shots`, and most of it is about when
the rule declines to act: a few same-second bursts, unusable timestamps, a
missing capture time, and a differently framed frame in the same second all
have to leave the gallery untouched.

`test_preselect.py` covers the other side of that: what each constraint
commits, that a hand pick settles its own `yes` category rather than doubling
it, that one frame of the couple covers both of them, and that with every
switch off nothing is committed and no allowance moves.

`test_ingest_readers.py` builds synthetic protobuf and PAI blobs with known
content, serves them through a patched `PTFile`, and asserts each reader
returns exactly the DataFrame the rest of the pipeline expects — columns,
dtypes and the proto-shaped `faces_info` / `bodies_info` / `background_centroid`
cells cropping depends on. It covers the missing-blob paths and runs
`load_gallery_assets` and the registered `ingest.gallery_assets` substage
end to end.

---

## 9. Deliberate behaviour differences

`select.preselect` is the one substage that is **not** behaviour-preserving, and
deliberately so — see its section above for what each constraint changes and by
how much. `tests/test_selection_equivalence.py` switches it off, because holding
it to the monolith would be asserting the change had not been made; with it off
the pipeline still reproduces the monolith exactly, which is what makes the
measured departures attributable to the constraints rather than to drift.

Everything else is behaviour-preserving except the places below, all of which
turn a crash or a divergence into the sane result:

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
