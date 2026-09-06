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
| Cohesion clusters picks where `select_remove_similar` spreads them | median gap between same-class picks **14 (cp-sat) vs 7 (loop)** | **backwards** — the window terms (300/150) dominate cohesion (60), so it spreads *more* |
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

**Phase 1 — the `time` dimension.** Replace `_add_windows` and `_add_spacing`
with bucket-coverage on time. Fewest moving parts, and it directly tests
whether "coverage as reward" behaves like "deviation as penalty". Expected: the
median same-class gap moves from 14 back toward the loop's 7.

**Phase 2 — `people` and `content`.** Retire the comparison against
`person_max_union_selection` and `select_non_similar_images`. This is where the
under-delivery in §1's table should start reproducing itself: `speech` should
fall to 2 of 3 on its own, because the third photo covers no new identity.

**Phase 3 — `visual`.** Embedding buckets from the existing clustering rather
than the current pairwise `_add_exclusions`, which is O(n²) within a class and
only reaches `cohesion_max_gap` positions.

**Phase 4 — the quota becomes a ceiling.** `sum(class) ≤ need`. Keep the
penalised shortage only for classes with a genuine floor (the `yes` events).
This is the change that stops the model over-filling, and it is safe only once
Phases 1–3 give it a reason to stop.

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

The two numbers to watch as later phases land: **short → should stay near 12**
as the model learns to saturate, and **|cp-sat − loop| → should fall toward 0**.

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
