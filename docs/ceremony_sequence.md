# The ceremony sequence reads forward

Why `src/groups_operations/ceremony.py` exists, what was tried before it, and
which parts of it are load-bearing. Written after the work in November 2026, for
whoever has to change it next.

## The problem

A wedding album is not uniformly chronological and should not be. On a dance
floor or in a couple session the design decides the order of a page, and two
frames a minute apart may sit either way round because one is the better
opening. The ceremony is the exception: groom walks in, bride walks in, vows,
kiss, send-off. That is a narrative, and printing it out of order reads as a
mistake rather than as an edit.

What was observed, on a real album:

```
page 11  20:15:13  ceremony
page 12  20:15:19  ceremony
page 13  20:16:09  ceremony
page 14  20:18:23  ceremony
page 15  20:15:38  bride and groom     <- the bride entering, after the vows
```

## Why ordering alone cannot fix it

A group is an **atomic block** downstream. `sort_groups_by_time` orders whole
groups by their median and no spread may pass one belonging to another group.
The couple's frames *at the altar* are filed `bride and groom` — the content
model cannot tell them from the ones shot at sunset two hours later, only the
clock can — so they sit in a group whose median falls after the vows.

`reorder_disjoint_spreads` cannot rescue it either. It moves a spread only when
every photo of it precedes every photo of the one in front, and that couple
group's single spread spanned the whole ceremony. It is disjoint from nothing.

So the boundary is **removed**, not ordered across: the groups inside the
ceremony are folded into one, and its spreads are cut from the photos in the
order they happened.

## What was tried first, and why it was not enough

Three earlier attempts are worth knowing about, because each fixed something
real and none fixed the reported bug.

1. **A chronology-first class list** (`CHRONOLOGY_FIRST_CLASSES`, predating this
   work). Narrows what stage 2 may emit and relaxes the page penalties, for
   `ceremony` only. It never reached the couple's frames at the altar, because
   those are not classed `ceremony`.
2. **Within-page ordering** (`_reorder_by_time`, `spreads_layout/spreads/spread.py`).
   A genuine latent defect: area tiers hand the biggest box the best-*ranked*
   photo and sort only within a tier, so a page of unequal boxes reads in rank
   order. Fixed and committed — but replayed over all 34 pages of the album
   above it changes none of them, because every multi-photo page there came from
   boxes inside the 0.01 area tolerance. It was not the bug.
3. **A window as an ordering rule.** Making a group chronology-first when it
   *overlapped* the ceremony window. This did fix a real within-group crossing,
   but left the group an atomic block, so it still printed after the ceremony.

The lesson worth keeping: **reconstruct the final page order from the log before
deciding which stage is at fault.** Two of the three above were diagnosed from
recorder JSON that showed a different, real defect in the same group.

## The two decisions that carry the risk

### Where the sequence is

`CEREMONY_SEQUENCE_CLASSES` names the classes whose photos mark the sequence
out. The window is the span they cover — **read from the whole gallery, not from
the photos that survived selection**.

That last point is not a detail. Selection is not deterministic here: three runs
of one commit produced three different group sets. A window read off the
selection moves run to run, so whether a group was folded in would be luck. It
also produced a concrete failure — on one run the earliest *selected*
processional frame was 20:15:19 while the couple group opened at 20:10:50,
inside the processional but before any frame of it was kept. The group had a
foot outside a window its own photos helped define. Where the ceremony happened
is a fact about the gallery, not about what was printed from it.

Two guards on the span, both deliberate:

- `CEREMONY_ANCHOR_CLASSES` — the processional, the kiss and the send-off are
  tagged by `enrich.ceremony_anchor` reading outward from the ceremony's climax,
  bounded by `aisle_lead_in` and `send_off_horizon`, so they cannot land far
  from it. `ceremony` is **not** among them: it comes from the content model,
  which also puts it on frames shot before the ceremony began. It may fill the
  span; it may not define an edge alone.
- `CEREMONY_WINDOW_MAX_GAP` (30 min) — marks are cut into runs and only one
  survives, so a single stray label cannot stretch the window across half the
  evening. The threshold has to clear the real gap between the vows ending and
  the guests gathering for the send-off: 17.3 minutes on the gallery this was
  measured against.

### Who is in it

**Containment, not overlap.** Only a group lying wholly inside the window is
folded in. A couple session that begins at the altar and runs on to sunset keeps
its own block, because merging it would drag photos from another part of the
evening into the ceremony. This was the explicit condition the work was accepted
under.

## Order of operations, and why

The merge runs inside `process_wedding_illegal_groups`, **before**
`handle_wedding_splitting`. Two things depend on that:

- the split's own rules apply to the merged group like any other — in
  particular the ceiling that keeps groups to two spreads, which the merge would
  otherwise bypass by assembling a long group after splitting had its say.
- `original_context` is assigned **before** the merge, not after. The merge
  rewrites `cluster_context` for the groups it folds together; taken afterwards,
  every photo of the sequence would read as `ceremony` and the page would lose
  the context mix it is scored on. Nothing between the two points touches the
  context.

`group_size` is recomputed by the merge, or the split rules would read the old
groups' numbers.

## What the merged group then gets

Its class is `ceremony`, which is in `CHRONOLOGY_FIRST_CLASSES` — aliased to
`CEREMONY_SEQUENCE_CLASSES`, one list, so the two cannot drift. That turns on:

- stage 2: `filter_time_disjoint` drops any split where two spreads straddle
  each other in time;
- stage 3: `CHRONOLOGY_FIRST_PENALTIES` relaxes `class_mix` and
  `context_mix_penalty` to 1.0, so a page holding the processional *and* the
  vows is not punished for the mix — without this the scorer fights the merge;
- stage 4: `_reorder_by_time` reads the page back chronologically among boxes of
  one orientation.

`bride and groom` is deliberately **not** in that list and must not be. It
appears all over an album, and outside the ceremony the design has more to say
about a page than the clock does. The couple's frames at the altar get the rule
by being folded into the sequence, never by their label.

## Turning it off

`CONFIGS['merge_ceremony_sequence'] = False` restores the old behaviour for an
A/B, and has a test.

## How to check a run

`files/stages_info/combinations/*.json` records, per group, the split that
shipped. Two spreads of one group should not straddle each other in time. The
log line `Ceremony sequence runs ...` names the window the run found, and
`Ceremony sequence: merged N group(s)` names what was folded in.

## Open questions

None at the time of writing; `docs/order_issues.md` carries the four that were
raised and closed, with the reasoning for each.
