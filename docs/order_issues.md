# Order issues — open list

Things found while fixing the order photos are read in. One per entry, to be
taken one at a time. Closed entries keep their note; they are the record of why
the code looks the way it does.

The design note that came out of all four is `docs/ceremony_sequence.md`.

---

## 1. The ceremony merge sits outside `groups_operations`

**Status:** closed 2026-11-12. Moved to `src/groups_operations/ceremony.py`,
called from `process_wedding_illegal_groups`. `src/core/` is back as it was;
nothing outside the tree needs the window any more.

`merge_ceremony_sequence` reshapes groups that already exist, which is exactly
what `src/groups_operations/` is for — but it lives in `src/core/` and is called
from `album_processing`, because the tree rule sends anything imported from
outside the package to `src/core/`.

The clean split:

- `src/core/ceremony_window.py` keeps the vocabulary — `ceremony_window`,
  `within_ceremony`, `groups_inside`. `spreads_layout` needs the window to decide
  what reads forward, so it has to be importable from anywhere.
- the regrouping moves into `groups_operations`, called at the end of
  `process_wedding_illegal_groups` rather than from `album_processing`. Timing
  works: that entrance already runs before the LUT strikes the spread budget.

The wrinkle: it merges *legal* groups for narrative reasons, while that entrance
is named for illegal ones. So it would be a new policy inside a tree named for a
different job.

---

## 2. The overlap branch of `is_chronology_first` may be dead weight

**Status:** closed 2026-11-12. Dropped. `is_chronology_first` takes a group name
again, and `CHRONOLOGY_FIRST_CLASSES` is aliased to `CEREMONY_SEQUENCE_CLASSES`
so an unmerged processional or send-off group still reads forward.

`is_chronology_first` is true when the group's class is in
`CHRONOLOGY_FIRST_CLASSES`, **or** when the group's photos overlap the ceremony
window. After the merge in step 2, those two nearly always coincide: every group
wholly inside the window has become one group whose class is `ceremony`, which
the class test already catches.

So the overlap branch fires for exactly one leftover case — a group that
*straddles* an edge of the window, say 20:30 to 21:30 against a window ending at
20:40. Not contained, so not merged; overlapping, so ordered forward internally.
No group on the gallery this was measured against straddles, so the branch never
ran there.

Decide between:

- keep it, as insurance for a gallery where a group does straddle; all it does is
  order that group internally.
- drop it and rely on class + merge alone. One rule fewer, and closer to the
  stated intent that outside the sequence the design decides.

Leaning to dropping it.

---

## 3. The ceremony merge can hand layout a group longer than two spreads

**Status:** closed 2026-11-12, by issue 1 as predicted. The merge now runs before
`handle_wedding_splitting`, so the split's own rules apply to the merged group.

`is_split_needed` runs inside `process_wedding_illegal_groups`; the ceremony
merge runs after it, in `album_processing`. So the merge can reassemble a group
that splitting has already decided about, and the merge-split stage's intent --
that no group longer than two spreads reaches layout -- is not enforced on it.

Measured: on the gallery this was built against the merged group came to 8
photos over 2 spreads, inside the limit. Nothing guarantees that; a gallery with
a longer processional or send-off would exceed it.

The layout stage's own fallback does hold up if it happens:
`split_group_if_needed` cuts the group into slices by position, and the merged
group is time-sorted, so the slices are time-contiguous and chronology survives.
It is still the fallback, not the intent.

The fix falls out of issue 1: with the merge inside
`process_wedding_illegal_groups` it can run *before* the split, and the existing
2-spread rule applies to the merged group with no new code.

---

## 4. The window moved with the selection, not with the gallery

**Status:** closed 2026-11-12.

Found by a test written while closing issue 2. The window was read from the
photos that survived selection, so its edges were an artifact of what selection
kept. On one run the earliest processional frame kept was 20:15:19 while the
couple group opened at 20:10:50 -- inside the processional, before any frame of
it was selected. The group therefore had a foot outside a window its own photos
helped define, containment excluded it, and it stayed the atomic block that
printed the bride entering after the vows.

Worse, selection is not deterministic here: three runs of one commit produced
three different group sets, so whether a group was folded in was luck.

Fixed by reading the window from `all_gallery_df`. Where the ceremony happened
is a fact about the gallery, not about what was printed from it, and the gallery
does not move between runs. Containment, the anchored-class rule and the 30
minute gap guard are unchanged, so a wider window still cannot pull in a group
that is not wholly inside it.

The alternatives, for the record: growing the window by whatever overlaps it
needs a fixpoint loop and a threshold, and a long group overlapping by a second
would drag its whole span in; putting the overlap branch back would have ordered
the straddling group internally while still printing it after the ceremony,
which is the bug.
