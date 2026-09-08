# Multiple albums per gallery — a plan for review

One request, one gallery read, **N** albums. Selection and layout run per album;
read/enrich and report run once.

The saving is real but not N-fold: the read dominates, and layout does not.

| stage | runs | measured cost |
|---|---|---|
| read + ingest + enrich | **once** | 42 s (496 photos), 67 s (938), 96 s (955 in manifest) |
| `select.*` | **per album** | 0.5–1.1 s cp-sat; 1.9 s narrator (268 photos), 12.5 s (938) |
| layout + crop + assembly (ProcessStage) | **per album** | 5.0–7.7 s |
| report | **once** | negligible |

So N=5 costs roughly `read + 5 x 13 s` ≈ 105–160 s against ≈ 250–500 s for five
independent runs — about 2.5–3x, not 5x. Worth stating plainly up front so the
mode is not sold as more than it is.

## 1. The problem is not the loop, it is the shared state

Adding a `for` loop around selection+layout would appear to work and would
silently produce albums contaminated by their predecessors. Measured on a
60-photo gallery, running the SELECT pipeline twice against one message:

```
pass 1: saw  60 photos -> selected 29 | message frame now 29 | context reused: True
pass 2: saw  29 photos -> selected 24 | message frame now 24 | context reused: True
```

The second album is composed from the first album's output. No error, no
warning, and a plausible-looking album at the end of it.

`message` is used as mutable shared state throughout, and this session hit three
separate instances of the same failure — each silent, each found only by
measuring output that looked reasonable:

* **`select.publish` narrows `content['gallery_photos_info']` to the selection.**
  A second pass over the same message starts from the first pass's chosen
  photos. Observed: a picker comparison where the second picker began from the
  first's 137 photos instead of the gallery's 559 and looked like it had chosen
  117.
* **`AlbumContext` is cached on the message** (`_MESSAGE_SLOT`) and
  `AlbumContext.for_message` reuses the attached object rather than rebuilding
  from content. Observed: an `is_wedding` override written to `content` was
  ignored entirely — budget and preselect still ran and the narrator still
  declined — because selection reads `facts` off the cached context.
* **ProcessStage writes the *laid-out* frame back**: `message.content
  ['gallery_photos_info'] = df` after adding `time_cluster`,
  `group_sub_index` and the `cropped_*` columns, so the next pass inherits
  layout state as if it were gallery state.

Everything else that a second pass would inherit:

| written by | keys |
|---|---|
| `select.publish` / `sync_to_message` | `photos`, `spreads_dict`, `min_total_spreads`, `max_total_spreads`, `modified_lut`, `manual_selection`, `bride and groom`, `predefined_layout` |
| ProcessStage | `gallery_photos_info`, `album_doc`, `error` |
| `update_photos_ranks(df, chosen)` | mutates `image_order` from the chosen set |
| `allocate()` | mutates `focus_table` (a fresh copy per request today — per *request*, not per album) |
| `CONFIGS` | process-global; any variant that changes config changes it for everything |

**The design consequence.** Do not enumerate what to reset — that is how all
three bugs happened. Invert it: freeze an immutable base after the read and
build every album from a fresh copy of it, so inheriting state is impossible by
construction rather than by discipline.

## 2. Mechanism — the fan-out already exists

`Worker` posts whatever a stage returns with `out_q.postMessage(resMsg)` and
**does not iterate it**, so a returned list travels the pipeline as one queue
item. Every stage already unpacks it:

| stage | already handles a list |
|---|---|
| `ReadStage.read_messages` | returns a list today |
| `SelectionStage.get_selection` | `msgs if isinstance(msgs, list) else [msgs]`, then `for _msg in messages` |
| `ProcessStage.process_message` | `msgs if isinstance(msgs, list) else [msgs]` |
| `ReportStage.report_message` | `if isinstance(msgs, Message) ... elif isinstance(msgs, list): for one_msg in msgs` |

So the inner loop is not something to add — it is present four times over, and
multi-album is **`ReadStage` returning N messages instead of one**.

**N is the payload's length, never `batch_size`.** The dispatch is
`work_in_single(stage) if stage.batch_size < 2 else work_in_batch(stage)`, and
all four stages use `batch_size=1`, so `work_in_single` hands the *whole* queue
item to `task_fn` and posts the result whole. A downstream stage therefore never
needs to be told N in advance — it reads `len(messages)` on arrival, which is
what the existing loops already do. This is the reason one code path serves both
single and multiple: if N were `batch_size` it would be fixed at construction
for the service's lifetime and could not vary per request.

⚠️ **The design requires `batch_size` to stay 1.** At 2 or more,
`work_in_batch` collects several queue items into a list and then iterates
`resBatch`, posting each element separately and inspecting `msg.error`. A
sibling list would become a list-of-lists and the fan-in would break. Anyone
raising `batch_size` for throughput has to reconsider this design first.

```
ReadStage      read + enrich ONCE -> GalleryBase
               return [AlbumMessage(base, variant_0), ... AlbumMessage(base, variant_N-1)]
                                |
                    one queue item, a list of N siblings
                                v
SelectionStage  for _msg in messages:      # already loops
ProcessStage    for message in messages:   # already loops
ReportStage     collect N album_docs -> ONE report, delete the ONE source message
```

**Single and multiple are the same code path.** N=1 returns a one-element list,
which is exactly what `ReadStage` returns today. There is no `if multi:` branch
anywhere.

### N is decided by enrich, not by the request

How many albums are worth making is a **derived fact about the gallery**, so it
cannot come off the request. `read_messages` runs `build_read()` (INGEST +
ENRICH) and only then does `ReadStage` return, so the fan-out point already sits
after enrich — but the count has to be *computed* there, from things only enrich
knows:

| fact | what it decides |
|---|---|
| `is_wedding` | whether a narrator variant is a candidate at all |
| `model_version` | a v1 (512-d) gallery cannot produce a narrator variant |
| which `cluster_context` classes exist | a focus or density variant is pointless if the gallery lacks the classes it would emphasise |
| gallery size | a small gallery may support only one viable album |

So the planner is a substage at the end of ENRICH -- `enrich.variants` --
producing `context.variants: list[AlbumVariant]`, with declared `requires` so it
is contract-checked and unit-testable like every other derived fact. **`len(variants)`
is N**, and a single album is `len(variants) == 1`: the same code path, now for a
principled reason rather than a convenient one.

### The three changes this needs

1. **`ReadStage` fans out**, one sibling per planned variant. Build
   `GalleryBase` once, emit `len(context.variants)` sibling messages.
   Siblings share the base **by reference**; each takes its working copy of the
   frame lazily when its selection starts, so one copy is live at a time rather
   than N. This matters — a 955-photo frame is tens of MB once embeddings and
   face protos are in it.
2. **`ReportStage` fans in.** The only genuinely new logic, and today's list
   branch does the *wrong* thing for this mode: it calls `report_one_message`
   per element, which would send N queue messages and delete N source messages.
   It must collect the N `album_doc`s, send one report, and delete the single
   source message once. Error semantics get decided here too: one album failing
   should not necessarily fail the request.
3. **Isolation, which the loops do not provide.** The stages iterate happily
   over siblings that share state — that is precisely the 60 -> 29 -> 24 result
   in §1. Siblings must not share `content['gallery_photos_info']`, and
   `AlbumContext.for_message` must not hand album 2 the context cached for
   album 1.

### The alternative, and why not

Collapsing `SelectionStage` and `ProcessStage` into one stage that loops over
variants internally would avoid the ReportStage join. Rejected: it throws away
list handling that already exists in all four stages and fuses two cleanly
separated stages, and the join it avoids is small precisely because the siblings
arrive **together in one list** rather than as N independent queue items needing
correlation ids, completion tracking and a timeout. That property is the reason
this design is cheap, and it comes from the framework for free.

## 3. What differs between albums

The mode is pointless without a definition of *variant*, and this is the part I
have least evidence for — it is a product question, not a mechanical one.
Candidates, cheapest first:

| variant axis | why | cost |
|---|---|---|
| `density` (1–5) | already a request field; changes budget and photo count | free |
| picker (cp-sat / loop) | the two disagree by 3–7 photos per gallery | free |
| narrator `seed` / `sample_k` | the policy is stochastic; different rollouts are genuinely different albums | free |
| `focus` | changes the whole budget shape | free |

Proposal: a variant is a **named config overlay plus hint overrides**, so the
axis set is data rather than code, and the request can carry either a count
(`albumCount: 3` → a default variant set) or an explicit list. Which axes ship
by default should be decided by whoever wants the feature; the machinery does
not care.

## 4. Response

Measured, from a real 27-spread / 72-photo album:

```
json                39,108 bytes
gzip + base64        4,888 bytes    <- what goes on the queue
Azure queue cap     65,536 bytes    -> ~13 albums of this size
```

Recommended shape, and the reasoning is in §5 of the earlier discussion:

```json
{"requestId": "...", "error": null,
 "composition": { ...first album... },
 "albums": [ { ...album 1... }, { ...album 2... } ]}
```

`albums` as the new plural key, `composition` kept pointing at the first album so
an unchanged consumer still works. **Not** by extending `compositions`:
`compositionId` and `placementImgId` both restart at 0 per `assembly_output`
call, so merging albums into that list makes `placementsImg` ambiguous about
which album a placement belongs to.

Two things this depends on that are outside this repo:

* **`userJobId` and `compositionPackageId` are per *request*.** N albums share
  them. If the consumer keys a saved album by `userJobId`, N albums collide on
  write no matter what the JSON says. This must be settled with the consumer
  owner before the shape is fixed.
* **There is no size guard on the outgoing report today** —
  `json.dumps` → gzip → base64 → `send_message`, unchecked. At 13 albums it
  fails at the Azure boundary. The fix has a precedent on the *incoming* side:
  `designInfoTempLocation` / `ratingTempLocation` put an oversized payload in
  blob and pass a location. The response should do the same above a threshold.

## 5. Constraints to respect

* **Message lease.** `visibility_timeout` is 1200 s. `read + N x (select+layout)`
  must fit or the queue redelivers and the work is done twice. At 96 s + 13 s per
  album, N=10 is ~226 s — fine — but a pathological gallery could approach it.
  Needs a cap on N and a budget check, not just hope.
* **Cropping is duplicated.** ProcessStage spawns a crop subprocess per album
  over the selected frame. Crops are a property of the *photo*, not of the
  album, so the union of selected photos can be cropped once and shared. This is
  the one real optimisation available beyond the shared read.
* **Determinism — a prerequisite for Phase 2, discovered in Phase 0.**
  `smart_non_wedding_selection` is **not deterministic**:
  `select_random_image` (utils/selection/non_wedding_selection_tools.py) calls
  `random.choice` on Python's global RNG, which the `np.random.seed(42)` in
  main.py does not touch. So two runs of the same non-wedding album differ,
  and Phase 2's "N identical variants must produce identical albums" is
  unachievable on that path until the RNG is seeded per album. cp-sat is
  deterministic and the narrator already takes an explicit seed; this is the
  loop path only. Phase 0's tests seed it themselves to stay sensitive to
  isolation alone -- which is a workaround for a test, not a fix for the
  service.
* **Sequential first.** Parallel album passes are tempting but unsafe today:
  `CONFIGS` is process-global (so a config-overlay variant would leak across
  threads) and the crop subprocess uses a single `self.q`. Sequential, then
  revisit.

## 6. Phases

Each phase ends with something measurable, in the order that de-risks the most
first.

* **Phase 0 — pin the contamination as a test.** Turn the measurement in §1
  into a regression test: run selection twice against one message, assert the
  second pass sees the whole gallery. It **fails today** (60 -> 29 -> 24, shown
  above), and it is the test that must go green in Phase 1 and stay green.
  Without it the rest is unverifiable, and every later phase is scored against
  it.
* **Phase 1 — `GalleryBase` + `AlbumRun`, N=1.** Extract the immutable base and
  the per-album scope; one album still produced. Acceptance: the wedding
  regression is byte-identical (same cp-sat objective, photos, covers, spread
  count) — the same check used when the narrator landed.
* **Phase 2 — fan out to N identical variants.** Acceptance: N albums that are
  **identical to each other and to the N=1 output**. This is the phase that
  catches contamination; if albums differ with identical inputs, state is
  leaking.
* **Phase 3 — `enrich.variants` + real variants.** The planner substage that
  derives the variant set from the enriched gallery (§2), plus the config
  overlay and hint overrides each variant carries. Until this phase N is 1 or a
  test fixture; after it, N is whatever the gallery supports. Acceptance:
  albums differ, each one individually matches what a single run with those
  settings produces, and a gallery that supports only one album yields exactly
  one -- no empty or duplicate variants.
* **Phase 4 — response shape and size guard.** `albums` key, `composition` kept
  for compatibility, a measured size check with blob offload above the
  threshold.
* **Phase 5 — crop once.** Union of selected photos, one subprocess. Acceptance:
  identical crops, N-1 fewer crop passes.

## 7. Risks, and how each is caught

| risk | how it shows | caught by |
|---|---|---|
| album N contaminated by album N-1 | plausible but wrong albums; no error | Phase 2's identical-variants assertion |
| lease expiry on a slow gallery | duplicated work, two reports | N cap + budget check (Phase 1) |
| oversized queue message | failure at the Azure boundary, not in our code | Phase 4 size guard |
| `userJobId` collision downstream | albums overwrite each other after we return | consumer owner; blocks Phase 4's final shape |
| `CONFIGS` leakage between variants | wrong variant applied, silently | sequential-only until proven |

## 8. Not doing

* **Parallel album passes** — see §5. Sequential until the config and crop
  paths are per-run.
* **Extending `compositions`** — ruled out in §4 on id collisions.
* **Reusing selection across albums** — the whole point is that selection
  differs; only the read, and later the crops, are shared.
