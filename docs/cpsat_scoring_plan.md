# Generalised per-class scoring for the CP-SAT picker

A plan to replace the per-category strategies *and* the model's separate
coverage penalties with one thing: a **per-class weighted score whose terms
include coverage**. Written after measuring where `src/pipeline/select/cpsat.py`
actually diverges from `WeddingPicker`, because three plausible explanations
turned out to be wrong and the real one changes what the fix should be.

---

## 1. What the divergence actually is

Measured on 53459898 and 53147741, driving SELECT with the real requests'
`aiMetadata` (50 user picks each, 42 and 64 photos committed by
`select.preselect`).

**Three hypotheses, all disproved.** Worth recording so nobody spends time on
them:

| hypothesis | measurement | verdict |
|---|---|---|
| No identity filters, so `bride` admits non-solo frames | picks failing the loop's own filter: 2 vs 2, 8 vs 8 | not a live difference on these galleries |
| Cohesion clusters picks where `select_remove_similar` spreads them | median gap between same-class picks 14 (cp-sat) vs 7 (loop) — but *without request hints*; with the real request it is 4 vs 5 and 3 vs 2 | not a difference. The first reading was an artefact: with nothing committed there are no anchors, and the window terms had free rein |
| No candidate gate, so it picks photos the loop never sees | **98/98 and 91/91** picks already inside the loop's shortlist | not a difference at all |

**The real one: the loop's quota is a ceiling it routinely leaves unmet, and
the model's is a target it fills.** Four separate mechanisms under-deliver, and
none of them is a constraint the model is missing — each is a *saturation*
effect:

| class | need | loop takes | why |
|---|---|---|---|
| `cake cutting` (53459898) | 1 | **0** | temporal narrowing empties the class |
| `may kiss bride` (53147741) | 1 | **0** | temporal narrowing empties the class |
| `bride` (53147741) | 4 | **1** | `_take_all_distinct` deduplicates on `(persons_ids, subquery)` |
| `bride getting dressed` | 5 | **3** | same |
| `kiss` | 3 | **2** | same |
| `speech` (53459898) | 3 | **2** | `person_max_union_selection` stops when a slot adds no new guest |
| `groom` (53459898) | 10 | **9** | `select_remove_similar` returns fewer than asked |

`_add_quotas` states `sum(class) + shortage == need` with the shortage
penalised at 4000 — far above every other term — so the model always fills.
That is the whole of "cp-sat selects more" (98 vs 96, 91 vs 84; 142 vs 137 and
136 vs 135 with real hints).

**So the parity fix is not another constraint.** Every mechanism in that table
is the same statement: *a photo that duplicates what is already picked is worth
nothing*. The loop expresses it four times, in four incompatible ways, each
inside a different strategy. Express it once, as a scoring term, and the
saturation falls out — the model stops early on its own and the quota can go
back to being a ceiling.

---

## 2. Done already

Two fixes landed (see the module docstring):

1. **Score over the free rows only.** `get_scores` min-max normalises within
   the frame it is given, and the loop scores a class after dropping committed
   photos. On 53459898 the old behaviour shifted `total_score` in all 20
   classes holding a committed photo, and **flipped the ranking order in 10**.
2. **`CandidateGate` gates the variables.** A class it declines stays empty;
   only its shortlist becomes variables. A no-op on the two validation
   galleries (the gate keeps everything there) but it removes a whole class of
   silent divergence, and it shrinks the model.

Deliberately *not* done, because both would have hard-coded the loop's ad-hoc
structure into the model rather than generalising it: temporal-orphan
narrowing, and the `accessories`/`wedding dress` special case. Both are §3-§4
below.

---

## 3. The shape: coverage as a scoring term

Today's objective is a sum over *photos*, with coverage bolted on as deviation
penalties over fixed windows:

```
maximise  Σ rank_i·x_i  +  Σ cohesion  −  Σ shortage  −  Σ window deviation  −  Σ grey
```

The proposal is a sum over **buckets covered**, per class, per dimension:

```
maximise  Σ_i rank_i·x_i                          quality, unchanged
        + Σ_c Σ_d w[c][d] · Σ_b covered[c][d][b]  coverage, the new part
        − Σ_c shortage_c                          only where a floor is real
```

with `covered[c][d][b] ≤ Σ{ x_i : photo i in class c, bucket b of dimension d }`
and `covered ≤ 1`. A boolean per bucket, so a second photo in a bucket earns
nothing. That is maximum-coverage — linear, and something CP-SAT is good at.

**The dimensions, and what each one replaces.** They are the same shape;
they differ only in what "the same thing" means:

| dimension `d` | bucket | replaces |
|---|---|---|
| `time` | position window of the day | `_add_windows`, `_add_spacing`, and the global album spread |
| `people` | identity (capped to the top-K by appearance) | `person_max_union_selection` |
| `content` | content-cluster label / `image_subquery_content` | `select_non_similar_images`, and `_take_all_distinct`'s `(persons_ids, subquery)` dedup |
| `visual` | embedding cluster | `select_remove_similar`, `_add_exclusions` |
| `colour` | `image_color` | the flat `grayscale_penalty`, and `GRAYSCALE_ONLY_CAP` |
| `orientation` | landscape / portrait | `_prefer_landscape` for `dancing` |

The global coverage the current model gets from `window_weight` is just the
`time` dimension with `c = ALL` — one row in the same table, not a separate
mechanism.

**What coverage cannot express, and must stay separate.** "The `bride` class
means solo-bride frames" is a *hard eligibility predicate*, not a weight. Keep
`CoupleTimelineStrategy`'s identity filters — including the over-filter
recovery, without which a gallery with poor face detection loses whole classes
— as a per-class predicate table beside the weight table. Same for `walking the
aisle`'s two scripted beats, which are forced picks rather than preferences.

---

## 4. The weight table

`w[class][dimension]` is the artefact this plan is really about. It subsumes
the strategy registry (which class gets which rule becomes which dimension is
weighted), and it is the natural home for `selection_threshold[category]` and
the per-class LUT entries. Grounded in what the strategies do today:

| class | time | people | content | visual | notes |
|---|---|---|---|---|---|
| `portrait`, `very large group`, `speech` | low | **high** | low | med | people coverage is the entire point |
| `detail`, `settings`, `food`, `rings` | low | — | **high** | med | round-robin across content clusters |
| `bride`, `groom`, `bride and groom` | med | low | low | **high** | plus the identity predicate |
| `dancing` | med | med | low | high | plus the orientation preference |
| `ceremony`, `bride party`, `groom party` | **high** | med | low | med | spread across the event |
| `accessories`, `wedding dress` | 0 | 0 | 0 | 0 | pure rank — this *is* `USER_PREFERENCE_CATEGORIES` |
| `cake cutting`, `kiss`, `first dance`, `send off` | 0 | 0 | 0 | high | one moment; no spread wanted |

Two things this buys immediately. The `accessories` / `wedding dress` special
case stops being a special case — it is a row of zeros. And the `yes` classes
stop needing `_add_spacing`'s separate sparse branch, because zero time weight
already says "do not spread this".

**Set the weights by fitting, not by taste.** The objective is measurable:
minimise disagreement with the loop's selection across the validation
galleries, per class. That gives a defensible starting point *and* makes "as
similar as possible" a number rather than an opinion. Deliberate departures
from the loop then start from a known baseline instead of being tangled up with
calibration error. `scratchpad/diff_cpsat.py` already reports the per-class
deltas the fit would minimise.

---

## 5. Phases

Each is independently verifiable, and each keeps the fallback to the loop.

**Phase 0 — instrument. Done; see §7.** The attribution lives in
`WeddingPicker._note` and `CpSatPicker._pool`, not in a script: an instrument
that re-implemented the decision tree would drift from the tree.
`tools/pick_attribution.py` runs both pickers and tabulates, and
`tools/baselines/pick_attribution.json` is the recorded result.

**Phase 1 — the `time` dimension. Done; see §7.** `_add_time_coverage`
replaces `_add_windows` and `_add_spacing`, behind
`CONFIGS['pick_cpsat']['coverage']['enabled']` so the two can be measured
against each other. Two mechanisms become one, sparse classes stop needing a
separate branch, and `coverage_weight()` is the first row of the
`w[class][dimension]` table.

**Phase 2 — `people` and `content`. Implemented, shipped at zero weight; see
§7.** Both dimensions exist and are measurable, and neither improves agreement
with the loop at any weight tried. The reason is structural and it reorders the
remaining phases: **coverage cannot make the model stop.** While every extra
photo earns a flat positive rank, more photos is always better, so the solve
fills whatever the quota allows; the loop stops because its diversity passes
return fewer items than asked for. `speech` cannot fall to 2 of 3 on its own
until the quota is a ceiling *and* rank is a marginal value rather than a flat
per-photo bonus. So Phase 4 is the prerequisite for Phase 2 paying off, not the
other way round.

**Phase 3 — `visual`. Done, and it settles the question the plan was built
on; see §7.** `_by_appearance` groups a class greedily by cosine, O(n x buckets)
rather than the pairwise exclusions' O(n^2). It works, it is a real knob -- and
like people and content it does not improve the objective. Across twenty
configurations of visual weight, people+content weight and admission quantile,
the score is **8 of 11 in every single cell**. Coverage weighting is not the
lever.

**Phase 4 — the quota becomes a ceiling, and rank becomes marginal.**
`sum(class) ≤ need`, keeping the penalised shortage only for classes with a
genuine floor (the `yes` events). **A ceiling alone changes nothing** — Phase 2
established that, the hard way. With `Σ rank·x` as a flat positive term the
model still fills to the ceiling, because every admitted photo pays. One of two
things has to go with it:

* an **admission cost** per photo, so a pick has to clear a bar rather than
  merely be positive; or
* **rank net of redundancy** — a photo's value discounted by how much of it the
  already-picked set covers, which is coverage applied to the quality term
  instead of beside it.

**The first is done; see §7.** The admission cost is a **quantile of the
class**, not a constant: `get_scores` min-max normalises within each class, so
a flat cost means something different in every one, and sweeping one produced a
cliff rather than a gradient -- 143 photos at every value from 0 to 300, then
120 at 400. Against the class's own distribution it reads the same everywhere:
better than a fifth of your class, or bring something new.

The second remains the truer statement, and is still open.

**Phase 5 — fit the weight table**, then tune deliberately from the fitted
baseline.

**Phase 6 — retire cohesion, or justify it.** Nothing in the loop rewards
adjacency; `select_remove_similar` actively avoids it. If cohesion survives, it
should be because a spread of one moment reads better, argued and measured on
its own — not inherited from the paper.

---

## 6. Risks

**Model size.** Coverage variables are |classes| × |dimensions| × |buckets|.
Bound each: ≤ 12 time windows, identities capped to the top-K by appearance,
content clusters as they come, embedding buckets fixed by k. Without caps the
`people` dimension alone is 68 identities × 27 classes on 53459898.

**Fitting to the loop bakes in its bugs.** The loop's `_carry_over` quirk — a
category silently reusing the previous one's ranked list — is reproduced
behaviour, not intended behaviour. Fit against the loop's *output*, but exclude
the classes where `_carry_over` fired, or the weight table inherits an accident.

**Coverage rewards can outrun rank.** If `w[c][d]` is large, the model buys a
bucket with a bad photo. The rank term is capped at 1000 per photo
(`SCORE_SCALE`); coverage weights need to be scaled against that explicitly,
and the fit should be constrained so no single bucket is worth more than the
quality range within a class.

**The two-pool colour discipline is not quite a coverage dimension.** The loop
picks from colour and tops up from greyscale only on a shortfall — an ordering,
not a weighting. A `colour` weight approximates it; the exact rule needs a cap
(`greyscale ≤ max(0, need − colour_supply)`, and ≤ 2 when a class has no colour
at all).

---

## 7. The Phase 0 baseline

Recorded 2026-09-06 over 53459898 and 53147741 with their real requests'
`aiMetadata` — 60 classes, 42 and 64 photos committed. Refresh with:

```
python tools/pick_attribution.py --request 53459898_ai --request 53147741_ai --out
```

**Always with real hints.** An earlier pass of this with no hints put
`cake cutting` and `may kiss bride` at zero from temporal narrowing; with the
real request both are settled by their strategy instead, because the hints
change the budget *and* what `select.preselect` commits. A baseline taken
without hints describes a request nobody makes.

| mechanism | classes | short of need | over | \|cp-sat − loop\| |
|---|---|---|---|---|
| `strategy` | 27 | **7** | 0 | 9 |
| `no_allowance` | 19 | 0 | 0 | 0 |
| `take_all_distinct` | 6 | **4** | 0 | 2 |
| `all_committed` | 3 | 0 | 0 | 0 |
| `greyscale_only` | 1 | 0 | 0 | 1 |
| `temporal_narrowing` | 1 | **1** | 0 | 1 |

**The loop is 12 photos short of its budgeted allowance and never once over.**
That is the thesis of §1 in one line: the quota is a ceiling, and three
different saturation mechanisms leave it unmet. cp-sat differs from the loop by
13 photos in total (137 → 142 and 135 → 136 per gallery), which is the same
order as the shortfall — consistent with the difference being *the model
filling what the loop declines to*, not the model choosing differently in bulk.

**Do not watch agreement with the loop.** That was the wrong scoreboard, and
Phase 4 is where it showed: every setting that pulled the model's count toward
the loop's *reduced* the photos the two shared. Which is correct --
disagreeing on a class the loop under-filled out of timidity is the entire
point. Score the two halves separately instead, which
`tools/pick_attribution.py` now does:

* **headroom taken** -- of the similarity shortfall, how much the model
  filled. Higher is better.
* **restraint kept** -- of the scarcity and orphan shortfall, how much it
  left alone. Higher is better.

### The shortfall is not all virtue

Splitting the 11 photos by *why* the loop stopped, which decides whether it is
a target or a floor:

| kind | photos | reading |
|---|---|---|
| `orphan` | 1 | a quality filter dropped a temporally isolated photo. The loop is right; match it. |
| `scarcity` | 4 | the pool was never big enough. Match it. |
| **`similarity`** | **6** | photos were there and a diversity pass declined them — `53459898/groom` alone accounts for 4. **Budgeted pages left unfilled: headroom, not a target.** |

So parity is the floor, not the goal. Five of the eleven are the loop being
right; the other six are pages the album was budgeted and did not get, and a
model that trades coverage off globally should be able to fill them with
something better than a near-duplicate. `tools/pick_attribution.py` prints this
split, keyed on the *mechanism* rather than the pool size — an orphan-emptied
class has plenty of photos and is still not headroom.

### Phase 1 result

Coverage-as-reward against the deviation penalty, same galleries, real hints:

| | median same-class gap (loop 5 / 2) | photos | shared with the loop |
|---|---|---|---|
| deviation penalty | 4 / 3 | 142 / 136 | 77/137, 119/135 |
| **coverage reward** | **3 / 2** | 143 / 139 | 81/137, 117/135 |

Total divergence from the loop **12 → 10 photos**. Modest, and honestly
reported: with real hints the two were already close, because 42 and 64
committed photos anchor the day before either mechanism runs. The structural
win is the larger one — two mechanisms collapse into one, sparse classes lose
their special branch, and the weight table now has a call site.

### Phase 2 result, and what it cost

Agreement with the loop, spread, and price. Medians of repeated runs, real
hints. `people+content` scaled from 0 to their intended weights:

| people+content weight | agree 53459898 | agree 53147741 | same-class gap | model (vars/constraints) |
|---|---|---|---|---|
| deviation penalties *(pre-Phase 1)* | 77/137 | 119/135 | 4 / 3 | 1115 / **2356** |
| **×0 — time only, shipped** | **81/137** | 117/135 | 3 / 2 | 1138 / **1148** |
| ×0.1 | 80/137 | **119/135** | 3 / 2 | — |
| ×0.25 | 78/137 | 118/135 | 3 / 2 | — |
| ×0.5 | 80/137 | 116/135 | 8 / 2 | — |
| ×1.0 | 79/137 | 112/135 | 10 / 2 | 1482 / 1492 |

**No weight beats zero.** So people and content ship at zero: implemented,
measurable, and inert until Phase 4 gives them something to do. The intended
weights are recorded in `utils/configs.py` for the Phase 5 fit to start from
rather than invent.

Cost is the good news. Phase 1 **halved the constraints** (2356 → 1148) by
replacing two deviation constraints per class-window plus the spacing pairs
with one boolean per bucket. Adding people and content puts variables up ~30%
and still lands under the deviation baseline on the larger gallery. Pick-stage
wall time runs **1.2–1.5× the loop** throughout — 0.85s against 0.70s, 0.9s
against 0.61s — and does not move measurably between the three configurations
at this size. Watch it again at Phase 3: embedding buckets are the first
dimension whose bucketing is not a groupby.

### Phase 4 result

The quota is a ceiling (`sum(class) <= need`), with a penalised floor kept only
for the `yes` classes, where coming back empty is a failure rather than
restraint. The admission cost is a per-class quantile.

Scored on the two halves that matter, over 60 classes:

| admission quantile | headroom taken | restraint kept | cp-sat photos (loop 137 / 135) |
|---|---|---|---|
| 0.0 *(ceiling only)* | **6/6** | 1/5 | 143 / 139 |
| **0.2 — shipped** | 5/6 | **3/5** | 142 / **135** |
| 0.5 | 5/6 | 3/5 | 142 / 133 |

A ceiling on its own fills every page the loop left on the table — and also
overruns four of the five classes it was right to stop on, because with a flat
positive rank every admitted photo still pays. The cost is what buys the
restraint back, at one photo of headroom.

Two things this cost me, recorded because they were both my own mistakes.
A **flat** admission cost does not work: normalisation is per class, so it
cliffs instead of grading. And I spent a sweep optimising **agreement with the
loop** before noticing it is the wrong objective — every setting that improved
the count made agreement worse, which is exactly what should happen when the
model correctly declines to imitate a class the loop under-filled.

Still imperfect: restraint is 3 of 5, so two classes get filled that should not
have been. Both are scarcity cases with small pools, where the ceiling permits
taking what little is there. Phase 3 is the next lever.

### Phase 3 result, and what it means for the plan

Visual coverage against the two knobs that could interact with it, scored on
headroom and restraint:

| visual | people+content | headroom | restraint | right |
|---|---|---|---|---|
| 0 | ×0 … ×1.0 | 5/6 | 3/5 | **8** |
| 250 | ×0 … ×1.0 | 6/6 | 2/5 | **8** |

Twenty cells including the admission-quantile sweep, and **every one scores 8
of 11.** Visual weight trades one headroom photo for one restraint; nothing
else moves at all. People and content do not move it either -- and that is now
tested the right way, on this scoreboard and with the ceiling in place, which
the Phase 2 write-off was not.

**Why, and it is not a tuning problem.** The three remaining errors are all
restraint failures, and all three sit in tiny pools:

| class | pool | need | loop | cp-sat | the loop's mechanism |
|---|---|---|---|---|---|
| `53459898/entertainment` | 3 | 3 | 2 | 3 | `take_all_distinct` |
| `53147741/bride` | 4 | 3 | 1 | 2 | `take_all_distinct` |
| `53147741/may kiss bride` | 3 | 1 | 0 | 1 | `temporal_narrowing` |

In a three-photo pool every photo is its own bucket in every dimension, so
coverage rewards taking all of them. Coverage is a statement about *breadth*,
and in a tiny pool everything is broad. No weight can express "there are three
photos here and two of them are the same shot".

**So the plan's central thesis is not supported.** Per-class coverage weights
do not subsume the strategies. What the remaining errors need is what §3
already identified as the thing coverage cannot express -- **hard eligibility
rules**, and specifically two:

* **At most one photo per `(persons_ids, subquery)` in a class, when that
  class's free pool is at or below its need.** This is `_take_all_distinct`
  exactly, conditional included. It must stay conditional: applied
  universally it would cap `dancing` at one photo, since every frame there
  shares people and subquery.
* **Temporal-orphan eligibility** — `drop_temporal_orphans` per class, which
  is what `may kiss bride` turns on.

Both are small, both are hard constraints rather than weights, and between them
they address all three failures. That is the recommended next step, and it
supersedes Phase 5: fitting a weight table is unlikely to pay when twenty
points of the weight space score identically.

`bound_by` is diagnostic only; nothing downstream reads `per_category`. Three
tests in `tests/test_pick_attribution.py` keep it honest — every category names
a mechanism, every mechanism is one `tools/pick_attribution.py` knows, and every
`return` in `_run_category` is preceded by a `_note`. The last one is
mutation-checked: deleting a single `_note` fails it.

---

## 8. Reference

Code: `src/pipeline/select/cpsat.py`; the loop it is measured against is
`WeddingPicker` in `src/pipeline/select/pick.py` with
`src/pipeline/select/strategies/`. Config: `CONFIGS['pick_cpsat']`.

Harnesses, all in the session scratchpad and all needing a frame that carries
`general_time` (a SELECT-only driver never runs `enrich.temporal`, and the
model declines without that axis):

- `smoke_cpsat.py` — does the solver run at all, or silently fall back
- `diff_cpsat.py` — per-class deltas, greyscale, orphans, identity violations
- `spread_cpsat.py` — same-class gap distribution, the clustering signature
- `gate_cpsat.py` — how much of a pick set the loop's shortlist contains
- `why_empty.py` — which decision point bound a class's count
- `verify_fixes.py` — end-to-end with the real request hints
