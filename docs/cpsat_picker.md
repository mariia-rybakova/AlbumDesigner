# The CP-SAT picker

A reference for `src/pipeline/select/cpsat.py`: what the model is, every
mechanism in it and why it is there, and — at least as usefully — the things
that were built, measured and switched off.

Companion documents: `docs/cpsat_scoring_plan.md` is the argument and the
phase-by-phase record; this is the description. `docs/substage_pipeline.md`
covers the pipeline it plugs into.

**Status: off by default** (`CONFIGS['pick_cpsat']['enabled']`). Any failure —
`ortools` missing, no solution inside the time limit, a modelling mistake —
logs and hands back to `WeddingPicker`, so switching it on cannot cost an
album. That safety has a cost of its own: a mis-modelled constraint is
indistinguishable from a working one from outside, which is why
`tests/test_cpsat.py` exists.

---

## 1. What it replaces, and the one idea

`WeddingPicker` decides one category at a time: score it, gate it, narrow it,
hand it to a strategy. Each category therefore chooses without knowing what
any other chose, and the only thing spreading picks across the day is the
diversity pass *inside* a category.

`CpSatPicker` states the whole album as one constrained optimisation over the
whole gallery, so coverage of the day, the per-class quotas, the identity rules
and the ranks are traded off against each other rather than in sequence. Same
signature — `run()` returns `(chosen_ids, per_category)` — so the driver swaps
between them and nothing downstream notices.

The formulation follows *Algorithms for Constrained Sequence Selection*, with
the sequence being the day (`general_time` rank, not the row index — on an
artificial-time gallery those are not the same thing).

```
maximise   Σ (rank_i − cost_i + identity_i) · x_i        per-photo value
         + Σ coverage rewards                            breadth
         + Σ cohesion rewards                            runs that read as one moment
         − Σ shortage penalties                          only where a floor is real
         − Σ repeat penalties  − Σ greyscale
subject to per-class ceilings, identity exclusions, distinct-shot and
           near-duplicate exclusions, and x_i = 1 for everything committed
```

Ranks are integers, so every score is scaled by `SCORE_SCALE = 1000`. Weights
throughout are calibrated against that: a term of 1200 outranks any rank
difference, a term of 60 only breaks ties.

---

## 2. Building the pool

### `_pool` — what is even a variable

Orders the gallery on the day's axis, flags what `select.preselect` committed,
scores each class, and drops everything the loop would never have offered a
strategy. Three decisions live here.

**Committed photos are pinned, not removed.** `x_i = 1` for each. Keeping them
in the frame means they still count towards windows and cohesion pairs, so the
solver treats them as anchors and fills around them rather than ignoring them.

**Scores are computed over the free rows only.** `get_scores` min-max
normalises within the frame it is handed, and the loop scores a class *after*
dropping committed photos. Scoring the whole class ranked it against a
different population: on 53459898, with 42 photos committed, this shifted
`total_score` in all 20 affected classes and flipped the ranking order in 10.

**`CandidateGate` decides eligibility.** The same gate the loop uses: it drops
a class whose photos do not score at all, and otherwise caps the field at
`allocation × 3`. A class it declines stays empty, as in the loop. Ineligible
rows are *dropped* rather than constrained to zero, so no variable is created
and they count towards no window — which is why window edges are cut from
`self.positions` (the whole day) rather than `len(frame)`.

---

## 3. The mechanisms

### 3.1 Quotas — a ceiling, not a target
`_add_quotas`, `CONFIGS['pick_cpsat']['quota_ceiling']`

`sum(class) ≤ need`, with a penalised floor kept **only** for the `yes`
classes, where coming back empty is a failure rather than restraint.

Measured over 60 classes, the loop comes in **11 photos short of its allowance
and never once over**, because its diversity passes return fewer items than
they were asked for. An equality with a penalised slack cannot express that:
`shortage_weight` at 4000 sits above every other term, so the model always
fills.

### 3.2 Admission cost — what a page is worth
`_admission_costs`, `admission_quantile: 0.2`

A ceiling alone changes nothing: while every admitted photo earns a flat
positive rank, more photos is always better and the solve fills to whatever it
is allowed. So a photo must clear a bar, *or* bring coverage that makes up the
difference.

**The bar is a quantile of the photo's own class, not a constant.**
Normalisation is per class — every class has a photo at 1.0 and one at 0.0 — so
a flat cost means something different in each. Sweeping one over 53459898 left
the count at 143 for every value from 0 to 300 and then dropped it to 120 at
400: a cliff where whole classes fall under the bar together. Against the
class's own distribution both galleries grade smoothly.

**Waived where nothing is ranked.** A class whose free pool is at or below its
allowance is not ranked at all in the loop — `_take_all_distinct` takes
everything bar the repeats without consulting a score — so charging a page bar
there drops photos the loop keeps. It cost `entertainment` and `kiss` one
distinct shot each before this was added.

### 3.3 Identity rules — who each class is about
`_identity_bonus`, `contradictions`, `IDENTITY_RULES`, `identity_preference`

`bride` means the bride on her own, `bride and groom` means the two of them and
nobody else, the getting-ready classes mean the bride. `CoupleTimelineStrategy`
and `BridePrepStrategy` carry these as hard filters; the model said nothing at
all, and it showed in the album — a hair-and-makeup spread of someone
unrelated to the couple, `groom` frames with no groom in them.

**Three-valued, because a wrong identity is not a missing one:**

| verdict | meaning | treatment |
|---|---|---|
| match | the class's rule is satisfied | bonus of 1200, above `SCORE_SCALE`, so a match beats any non-match on rank |
| unknown | `persons_ids` empty | neutral — a detection that did not happen; rank decides |
| contradicted | identities present, none of them the subject | **excluded** where the class is exclusive |

The unknown case is deliberate: it is the loop's `_recover_over_filtering`
without the special case. A hard rule there empties a class on a gallery where
detection missed the couple.

**Excluded, not charged.** As a penalty it was outvoted — a contradicted photo
still collects the time-coverage rewards for its class and window, +300 and
+150, which beat a 1200 charge once the rank is in. It only looked sufficient
on the validation galleries because those classes had unknown-identity frames
to fall back on.

**Only for `exclusive` classes.** `bride`, `groom`, `bride and groom` and the
getting-ready classes are about one person. The party classes and the
processional are not — parents and flower girls walk the aisle — and penalising
a non-couple face there emptied `walking the aisle` outright on 53459898,
losing a scripted moment to a detection gap.

The couple is read from the **photo table** (`bride_id` / `groom_id` columns,
as `CategoryRequest.bride_id` does), not from `context.facts`. Reading facts
first made the whole rule a silent no-op, since a SELECT-only driver never sets
them.

Solver picks with a wrong identity in an exclusive class: **5 → 0** on
53459898, 0 on 53147741. What remains there is photos the *user* hand-picked,
which are pinned and deliberately not second-guessed.

### 3.4 Distinct shots — the same people doing the same thing
`_add_distinct_shots`, `distinct_shots`

At most one photo per `(persons_ids, subquery)` in a class whose free pool is
at or below its allowance. `_take_all_distinct` as a constraint.

**The first mechanism of five to improve the objective** — restraint went 3/4
to 4/4 — and it had to be a hard rule. In a three-photo pool every photo is its
own bucket in every coverage dimension, so coverage *rewards* taking all three;
only an exclusion can say two of them are one shot.

**The condition is the whole safety of it.** Applied to every class it would
cap `dancing` at one photo, since every frame there holds the same couple and
carries the same subquery. Committed photos are left out, as in the loop — two
of them sharing a key would make the constraint infeasible.

### 3.5 Coverage — reaching a new part of something
`_add_coverage`, `_cover`, `coverage`

A boolean per `(class, bucket)`, raisable only when something in that bucket is
picked, rewarded once. The first pick in a bucket earns it and the second earns
nothing, so the day gets covered and the remaining slots sit where the ranks
are best — diminishing returns rather than a target to hit.

Four dimensions share one mechanism, differing only in a bucketing function:

| dimension | bucket | replaces | shipped weight |
|---|---|---|---|
| `time` | window of the day | `_add_windows`, `_add_spacing` | **300** / 150 global |
| `people` | identity | `person_max_union_selection` | **0** |
| `content` | `image_subquery_content` | `select_non_similar_images` | **0** |
| `visual` | cosine cluster at 0.9 | `select_remove_similar` | **0** |

`class = ALL` is the album-wide statement — one more application of the same
mechanism, not a separate one. `max_buckets` caps model size by keeping only
the most populated buckets; `per_class` is the beginning of the
`w[class][dimension]` table, carrying zeros for the classes where the user's
own pick decides and for the single-moment events.

Replacing deviation penalties with coverage rewards **halved the constraint
count**, 2356 → 1148 on 53459898: two constraints per class-window plus the
spacing pairs became one boolean per bucket.

`content` buckets on `image_subquery_content` (116 values over 1069 photos)
rather than `cluster_label` (447, about two photos each) — at that granularity
covering a bucket is barely different from rewarding every photo.

### 3.6 Near-duplicate exclusion
`_add_exclusions`, `duplicate_similarity: 0.97`

Two frames of a class within `cohesion_max_gap` positions and above 0.97 cosine
cannot both be picked. Cohesion rewards neighbours, and neighbours are exactly
where the second copy of a shot lives, so without this the reward is collected
by duplicates. Also covers the committed photos, which is what
`select_remove_similar`'s `already_selected` does for the loop.

### 3.7 Cohesion
`_add_cohesion`, `cohesion_weight: 60`

Rewards picking consecutive photos of the same class, on the theory that a run
of neighbours reads as one moment. **Nothing in the loop rewards adjacency** —
`select_remove_similar` actively avoids it — so this is inherited from the paper
rather than from the behaviour being matched. At 60 against coverage's 300 it
only breaks ties. Retiring it, or justifying it on its own merits, is open.

### 3.8 Greyscale penalty
`grayscale_penalty: 200`

A flat per-photo charge, standing in for the loop's two-pool discipline: the
driver picks from colour and tops up from greyscale only on a shortfall,
capping at two when a class has no colour at all. One model has one pool, so
this is an approximation and a known one — a strong greyscale frame can still
displace a colour one.

### 3.9 Diagnostics
`_report`, `model_size`

Logs status, wall time, objective, per-class need against got, the per-window
spread, and the model's variable and constraint counts. Model size is recorded
before the solve, because "how big did that get" is the question a queue worker
cares about. `per_category['bound_by']` names which decision settled each class;
`tools/pick_attribution.py` tabulates it.

---

## 4. Things tried and let go

The useful half of the record. Five mechanisms were built, measured and left
inert; three attractive explanations were disproved.

### Mechanisms that did not pay

| mechanism | what it did | why it was dropped |
|---|---|---|
| **`people` coverage** | reward covering a new identity within a class, and album-wide | No weight from ×0 to ×1 improved the score. Agreement 81 → 78–80 of 137. |
| **`content` coverage** | reward covering a new subquery | Same sweep, same result. |
| **`visual` coverage** | reward covering a new appearance cluster | Trades one photo of headroom for one of restraint; scores an identical 8 of 11 either way. Ships at the safer corner (0) because breaking restraint puts a rejected photo in the album while missing headroom only leaves a page unfilled. |
| **person-repeat penalty** | charge for each extra photo of the same person in a class | 8 of 10 right at zero weight and at a tenth of the intended values; 7 at a quarter and above, where it starts costing headroom. |
| **flat admission cost** | one bar for every class | Cliffs instead of grading, because normalisation is per class. Replaced by the quantile; the config key survives as a floor and is 0. |

Their intended weights are kept in comments in `utils/configs.py` so a later
attempt starts from them rather than inventing new ones.

**The pattern is the finding.** Four coverage-shaped mechanisms in a row scored
identically across twenty configurations of weight and quantile. Coverage is a
statement about *breadth*, and the errors that remain live in pools of three or
four photos where every photo is its own bucket in every dimension — so
coverage rewards taking all of them. What moved the objective was hard rules:
the distinct-shot exclusion and the identity exclusion. **Per-class coverage
weights do not subsume the strategies**, which was the thesis
`docs/cpsat_scoring_plan.md` was built on, and fitting a weight table (that
plan's Phase 5) is hard to justify when twenty points of the weight space score
the same.

### Explanations that were wrong

Recorded so nobody spends time on them again.

| hypothesis | measurement | verdict |
|---|---|---|
| No identity filters, so `bride` admits non-solo frames | picks failing the loop's own filter: 2 vs 2, 8 vs 8 | Not a *live* difference in aggregate — but wrong all the same, and only visible by looking at a spread rather than a count. §3.3 exists because of what the album showed. |
| Cohesion clusters picks where the loop spreads them | median same-class gap 14 vs 7 — **without request hints**; 4 vs 5 with the real request | An artefact. With nothing committed there are no anchors and the window terms had free rein. |
| No candidate gate, so it picks photos the loop never sees | **98/98 and 91/91** picks already inside the loop's shortlist | Not a difference at all. The gate was added anyway, as insurance and to shrink the model. |

### Measurement mistakes worth not repeating

* **Agreement with the loop is the wrong scoreboard.** Every setting that pulled
  the model's count toward the loop's reduced the photos the two shared — which
  is correct, because disagreeing on a class the loop under-filled out of
  timidity is the entire point. Score the two halves apart: *headroom taken* of
  the similarity shortfall, *restraint kept* of the scarcity and orphan
  shortfall.
* **`restraint kept` only counts taking more than the loop.** It scored a win
  while the distinct-shot rule was throwing away a genuinely distinct shot.
  *photos dropped that the loop kept* is the third line, added for that reason —
  and a non-zero value there needs the class named before it is read as a fault,
  since a deliberate decline looks the same.
* **Instruments must not re-implement the pipeline.** `hints_from_request`
  hand-parsed `aiMetadata` and forced `present=True`, silently turning a manual
  request into an AI one and reporting a healthy 58-photo selection where the
  pipeline lays out the whole gallery. `why_failed.py` reused one message
  object for both pickers, so the second started from the first's output. Both
  now defer to the pipeline's own code.
* **Replay only requests whose `aiMetadata.photoIds` is not null.** A null one
  is a manual album: `select.route` marks the message manual and every substage
  after it skips, so the whole gallery reaches layouting. Deliberate, and
  indistinguishable from a catastrophic selection bug from outside.

---

## 5. Where it stands

Scored on the two halves that matter, over the validation galleries:

| gallery | density | headroom | restraint | dropped |
|---|---|---|---|---|
| 53459898 + 53147741 | 3 | 6/6 | 4/4 | 0 |
| 53520346 | 2 | 2/2 | 3/3 | 0 |
| 52989013 | 5 | 6/6 | 10/10 | **6** |

Cost: pick-stage wall time runs **1.2–1.6× the loop** throughout, and the
model stays around 1100–1500 variables on a 1000-photo gallery.

**Open.** The six dropped photos on 52989013 are unexplained — the metric
cannot tell six correctly-declined near-duplicates from six real losses, and
that gallery's density-5 allowance is tight enough against supply
(1 photo in 4, against 1 in 8–9 elsewhere) that the ceiling genuinely binds
there. `take_all_distinct` accounts for 10 of that gallery's 16-photo
shortfall across 5 classes, far more than anywhere else, so it is also the
gallery on which to sanity-check that rule's threshold.

## 6. Reference

Code `src/pipeline/select/cpsat.py`; the loop it is measured against is
`WeddingPicker` in `src/pipeline/select/pick.py` with
`src/pipeline/select/strategies/`. Config `CONFIGS['pick_cpsat']`. Tests
`tests/test_cpsat.py`, plus `tests/test_pick_attribution.py` for the
attribution the measurements rest on.

Run it with `process_gallery.py … --cp-sat`. Tabulate a comparison with
`tools/pick_attribution.py`; the recorded baseline is
`tools/baselines/pick_attribution.json`.

> Every local run of `process_gallery.py` leaks its process. `ptinfra.intialize`
> starts a non-daemon `ElasticQueueThread` that posts every ~90 seconds and
> never exits, so the interpreter cannot shut down after `main` returns — and
> stdout stays block-buffered, which is why the log file looks empty. Use
> `python -u`, and kill the process once the PDF is written.
