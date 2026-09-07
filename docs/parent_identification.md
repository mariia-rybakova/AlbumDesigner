# Parent identification

`enrich.parents` answers one question: **which identities are the couple's
parents?** The answer is three-valued — the bride's parents, the groom's
parents, or *inconclusive* — and inconclusive is a normal outcome, not a
failure.

This replaces a rule that classified *photos* and never named anyone. Both of
that rule's discriminating tests turned out to be broken; it is still reachable
via `CONFIGS['parents']['by_identity'] = False`, for the equivalence tests.

---

## 1. Why the old rule had to go

`identify_parents` accepted a `portrait` row with 3–4 distinct people including
at least one partner, whose two others were of different genders, were "a
couple in the social circles", and were roughly `couple_age + 15`.

**The age window selected the wrong generation.** `AGE_TOLERANCE = 10.0`
around an offset of `+15` accepts a candidate **+3 to +27** years older than
the couple and rejects **+28 and up**. The median parent–child gap is ~28–32
years, which lands just outside. Measured, with the real parents in the
couple's own social circle:

| line-up | ages | verdict |
|---|---|---|
| bride + groom + her mother and father | 58, 60 | **rejected** |
| bride + groom + two friends | 40, 42 | **accepted** |
| bride + groom + one parent | 58 | rejected (needs exactly two others) |

**The "must be a couple" test was not one.** The comment claimed
`couple_pairs` held `frozenset({id1, id2})`, but the loop was
`couple_pairs.update(ids)` — a flat set of individual ids — and the test was a
plain intersection. The rule reduced to *"at least one of the two appears in at
least one circle anywhere"*:

| the two candidates are… | verdict |
|---|---|
| in one circle together | accepted |
| in two **separate** circles | accepted |
| **only one** of them in any circle | accepted |
| neither in any circle | rejected |

The commented-out `if len(ids) == 2:` shows the intent. As written the social
graph contributed nothing, which is a shame, because it carries a lot — see §4.

**Consequences on the validation galleries.** 53147741 was given eleven
"parents"; 53459898 six. Three of eight galleries returned early and never
created the `parent_category` column at all. Where the rule did find real
parents it was by luck: on 53459898 the groom's estimated age of 42 shifted the
window up into the parent band. For a couple both aged 26 the same six
identities would have been reduced to one.

---

## 2. What changed

**The unit of decision is an identity, not a photo.** Once the parents have
names, labelling is exact and needs no guessing: a portrait's `parent_category`
follows from which named parents are in it. Three of the old rule's structural
blind spots close for free, because nothing counts heads any more:

- one parent (widowed, divorced, or separately photographed) now works
- three (a step-parent) now works
- two parents of the same gender now work — the old rule rejected them on
  `parent_gender_1 == parent_gender_2`

**Every gate fails toward silence.** An unmarked parent costs a category that
would have been budgeted anyway; a marked stranger puts a stranger on the
family spread. The thresholds are set accordingly.

---

## 3. The hard part: parent vs. wedding party

Not parent vs. guest. The bridal party shares almost every indicator with the
parents — both are in the prep scenes, both are at the ceremony, both are in
the posed portraits — and is far more numerous. Three things separate them.

**The party classes.** Measured per identity:

| gallery | wedding party, `*party* frames` | parents |
|---|---|---|
| 47981912 | 32, 39, 42, 42 | 0, 0, 1, 3 |
| 53459898 | 45, 47, 51, 52, 53 | 0, 2, 2 |

The strongest single signal, and a *negative* one.

**But the penalty must be relative to the gallery.** On 53459898 the groom's
father has **10** `groom party` frames: in a suit he is not visually separable
from the groomsmen, and the content model groups him with them. Ten is
disqualifying on an absolute scale and obviously not against the groomsmen's
45–53. So the penalty is scaled by the most party-heavy candidate in the same
gallery. An absolute threshold of 8 sank him; the relative one puts him at
0.18 and he is correctly named.

**Side skew.** Frames alone with one partner, the other absent. Family are
photographed with their own child. On 47981912 the six candidates split
cleanly — 21/1, 15/0, 10/2 bride-side against 0/13, 3/23, 0/10 groom-side.
This is both a scored positive *and* how the side is assigned. The old rule
took the side from `main_persons[0]`/`[1]` — most-photographed order — which
says nothing about who is whose, and is a different notion of "the couple" from
the `bride_id`/`groom_id` that `enrich.identities` already resolved.

**Age, as a rank.** Never as an offset. Face-age estimators regress toward the
mean: a 60-year-old reads as ~50 and a 30-year-old as ~32, so the *gap*
compresses while the *ordering* survives. Same lesson as `enrich.timeline` —
position is trustworthy, measured distance is not. `min_age_rank` is a hard
requirement that nothing overrides.

> The rank is relative to the gallery's own identity population, so it is
> weakest on a gallery with very few identities. The real ones carry 27–68.

---

## 4. The rarer signals

Near-conclusive when they fire, absent otherwise, so they enter as bonuses
rather than requirements.

**The parent dance.** A dance frame whose people are *exactly* one partner and
one candidate. Nothing else looks like that. This is what identified the
groom's mother on 53459898 (id 16, two frames) — the mother-son dance, one of
the indicators worth having precisely because it is unambiguous.

**A small social circle shared with another old candidate.** How the two
parents of one side corroborate each other. 53459898's circles are `[19,21]`,
`[8,27]`, `[9,11]`, `[8,28,39]`, `[16,39,15]`, `[17,25,33]` — pairs and family
triples, exactly the structure the old rule flattened away. A circle shared
with a *young* candidate says nothing; that is a couple of guests.

**The officiant, as a negative.** Old, at the ceremony, on neither side, and
confined to a narrow band of the day. Id 26 on 53459898 is 61 with 28 ceremony
frames inside 8% of the gallery, and ranks well on age alone.

---

## 5. The queries augment age — they do not replace it

`parents_of_couple` minus `wedding_party_member`, scored only on photos where
the candidate is one of at most three identities, so the image-level cosine is
actually about them. Concepts built by `tools/build_concept_bin.py`.

On 47981912 the delta orders correctly: parents +0.04…+0.09, party −0.06. On
53459898 the *raw* delta is dominated by sample size — the known bride's
mother, with 31 attributable photos, ranked sixteenth behind a dozen
identities with three or four. So it is shrunk by `n/(n + query_prior)` and
capped at a weight of 0.15. It is a second opinion on age, not a detector, and
`test_the_query_term_cannot_name_a_parent_on_its_own` pins that.

A missing `.bin` drops the term and leaves the structural indicators standing;
it does not fail the gallery.

> **Deployment.** The three new bins exist locally in both CLIP spaces. The
> production service reads them from blob storage, so they must be published
> (`tools/build_concept_bin.py <name> --upload-only`) before this runs outside
> a local replay. That is a production write and is deliberately not done
> automatically.

---

## 6. Results

| gallery | old rule named | new: bride's | new: groom's |
|---|---|---|---|
| 53459898 | 8, 15, 16, 17, 33, 39 | **8** (age 66, prep 6, aisle 4) | **16** (parent-dance 2), **15** (age 63) |
| 53147741 | 10, 13, 28, 32, 38, 39, 46, 52, 54, 57, 60 | **8** (aisle 10) | **28** (circle 13, 57) |
| 52894932 | 3, 4, 10, 11, 18, 24 | *inconclusive* | *inconclusive* |

53459898's three match a hand reading of the gallery. The bride's father is not
named — plausibly id 39, which shares her mother's circle but has no side
skew — and that is the intended behaviour rather than a miss to fix.

52894932 has no prep, no party, no dances and no aisle coverage, so every
candidate is an age and a co-occurrence count. Two identities aged 73 and 74
share a circle and are almost certainly a parent couple, but they appear evenly
with both partners, so *whose* they are is unknowable. The stage declines, and
the reason is logged: `candidates [13, 12, 16] are within 0.10 of each other`.

---

## 7. Known gaps

**Side-less parents are the main recall loss.** A candidate photographed mostly
with both partners or with neither gets no side, and the contract requires one.
On 53147741 that leaves `[14, 60, 11, 57]` — four identities aged 60–68 sharing
circles — unresolved. Adding a *side unknown* outcome would recover them, but
the caller would have nothing to do with it: `parent_category` is defined by
side.

**Dead configuration, unrelated to this stage but adjacent to it.**
`cluster_context` only ever becomes `parents portrait`; the three-way split
lives in `parent_category`. Yet `bride with her parents`, `groom with his
parents` and `bride and groom with parents` are configured as if they were
categories — in `files/focus_csv.csv` (3% / 2% / 2% under the `parents` focus),
in three threshold blocks in `utils/configs.py`, and in
`utils/lookup_table_tools.py`. None is reachable
(`parents_categories = {"parents portrait"}`). A `focus: parents` request
therefore gets **3%**, not the 10% the CSV reads as.
`budget_normalise_present_only` makes the dead rows cost nothing, so this is a
documentation trap rather than a live leak — but now that the parents have
names, promoting the three labels to real categories is a genuine option, and
it would quadruple the parents focus. That is a budget change and is not part
of this one.

**The `suit` class is the groom's only prep signal.** The content model has
`bride getting dressed` and `getting hair-makeup` for her side and nothing
equivalent for his, so groom-side prep counts are systematically the weaker of
the two.

---

## 8. Reference

Code: `src/pipeline/enrich/parents.py` — `measure` (evidence), `score`
(weighting), `resolve` (the decision), `label` (the photo table).
Wiring: `ParentsSubStage` in `src/pipeline/enrich/identities.py`.
Config: `CONFIGS['parents']`. Tests: `tests/test_parents.py`.
Diagnostics: `scratchpad/diag_parents.py`, `scratchpad/eval_parents.py`.

Named parents land on `context.facts.bride_parents` / `.groom_parents`, so
anything downstream — a strategy, `select.preselect`'s identity coverage — can
use them without re-deriving.
