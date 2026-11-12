"""The one stretch of a wedding that must read forward, folded into one group.

A wedding album is not uniformly chronological, and should not be. Most of it is
edited: a couple session, a dance floor, a table of details -- there the design
decides, and two frames a minute apart may sit either way round because one is a
better opening than the other. The ceremony is the exception. The groom walks
in, the bride walks in, the vows, the kiss, the send-off: that is a narrative,
and printing it out of order reads as a mistake rather than as an edit.

Ordering it is not enough, because a group is an atomic block downstream.
`sort_groups_by_time` orders whole groups by their median and no spread may pass
one belonging to another group, so the couple's frames *at the altar* -- filed
'bride and groom', indistinguishable by class from the ones taken at sunset two
hours later -- sit in a group whose median falls after the vows, and the bride
entering prints after the ceremony she is walking into. No reordering pass can
repair that: the group's spread spans the whole ceremony and is disjoint from
nothing.

So the boundary is removed rather than ordered across. The sequence becomes one
group, and its spreads are cut from the photos in the order they happened.

Two decisions carry the risk, and both are deliberately conservative:

*Where the sequence is.* `CEREMONY_SEQUENCE_CLASSES` names the classes whose
photos mark it out, and the window is the span they cover -- read from the whole
gallery, not from the photos that survived selection. Selection is not
deterministic here, and a window read off it moves from run to run, so whether a
group was folded in would be luck. The gallery does not move.

*Who is in it.* Containment, not overlap: only a group lying wholly inside the
window is folded in. A couple session that begins at the altar and runs on to
sunset keeps its own block, because merging it would drag photos from another
part of the evening into the ceremony.

This runs before `handle_wedding_splitting`, so the split's own rules -- and the
two-spread ceiling in particular -- apply to the merged group like any other.

`docs/ceremony_sequence.md` carries the rest: the three earlier attempts that
each fixed something real and none of them this, and why the guards are shaped
the way they are.
"""

from typing import Any, List, Optional, Tuple

import pandas as pd

from src.groups_operations.split import update_groups_size
from utils.configs import (CONFIGS, CEREMONY_ANCHOR_CLASSES, CEREMONY_SEQUENCE_CLASSES,
                           CEREMONY_WINDOW_MAX_GAP, SPECIAL_GROUP_SEP)

#: The columns a wedding group is keyed by, in order.
GROUP_KEY = ['time_cluster', 'cluster_context', 'group_sub_index']

#: What the merged group is called. It is a ceremony, and naming it so also puts
#: it in `CHRONOLOGY_FIRST_CLASSES`, which is what makes its spreads read
#: forward once `spreads_layout` has it.
MERGED_CLASS = 'ceremony'


def _content_class(context: Any) -> Optional[str]:
    """A group's class with any special 'class|idx' tag stripped."""
    if not isinstance(context, str):
        return None
    return context.split(SPECIAL_GROUP_SEP)[0]


def _longest_run(marks: List[Tuple[float, bool]]) -> List[Tuple[float, bool]]:
    """The one stretch of the sequence the window is allowed to span.

    A single mislabelled frame must not be able to stretch the window across
    half the evening and swallow the groups in between -- that is the whole risk
    of defining a span as a hull. So the marks are cut into runs wherever they
    are more than `CEREMONY_WINDOW_MAX_GAP` apart, and only one run survives.

    The run that wins is the one holding the most *anchored* marks.
    `enrich.ceremony_anchor` tags those classes by reading outward from the
    ceremony's climax, so they cannot be far from it; a run of nothing but the
    content model's 'ceremony' label is the one that might be anywhere. Photo
    count breaks a tie, and the earlier run breaks that -- the ceremony comes
    before the party, and a late run of look-alike frames should not win on
    volume alone.
    """
    if not marks:
        return []

    runs = [[marks[0]]]
    for mark in marks[1:]:
        if mark[0] - runs[-1][-1][0] > CEREMONY_WINDOW_MAX_GAP:
            runs.append([])
        runs[-1].append(mark)

    return max(runs, key=lambda run: (sum(1 for _, anchored in run if anchored),
                                      len(run), -run[0][0]))


def ceremony_window(photos_df, logger=None) -> Optional[Tuple[float, float]]:
    """`(first, last)` general_time of the ceremony sequence, or None.

    None means the frame gave no sign of a ceremony -- a non-wedding gallery, or
    one the content model found no processional, vows or send-off in. Nothing is
    then folded together, which is the right failure: a window guessed from no
    evidence would impose chronology on an album that never asked for it.

    Args:
        photos_df: Photos carrying `cluster_context` and `general_time`. The
            whole gallery, normally -- see the module docstring on why.
        logger: Optional; one line naming the window that was found.
    """
    try:
        frame = photos_df[['cluster_context', 'general_time']].dropna()
    except (TypeError, KeyError, IndexError, AttributeError):
        return None
    if len(frame) == 0:
        return None

    marks = []
    for context, moment in frame.itertuples(index=False):
        content = _content_class(context)
        if content in CEREMONY_SEQUENCE_CLASSES:
            marks.append((float(moment), content in CEREMONY_ANCHOR_CLASSES))

    run = _longest_run(sorted(marks))
    if not run:
        return None

    window = (run[0][0], run[-1][0])
    if logger:
        dropped = len(marks) - len(run)
        logger.info(f"Ceremony sequence runs {window[0]:.0f}-{window[1]:.0f} "
                    f"({(window[1] - window[0]) / 60:.0f} min) over {len(run)} gallery photo(s)"
                    + (f", {dropped} outlier(s) ignored" if dropped else ""))
    return window


def groups_inside(photos_df, window: Optional[Tuple[float, float]]) -> List[Any]:
    """The groups lying *wholly* inside the window, earliest first.

    Containment, not overlap, and the difference is the whole safety of the
    merge: a group with one foot outside the window would bring photos from
    another part of the evening into the ceremony with it.
    """
    if window is None or photos_df is None or len(photos_df) == 0:
        return []

    inside = []
    for name, frame in photos_df.groupby(GROUP_KEY):
        times = frame['general_time'].dropna()
        if len(times) and times.min() >= window[0] and times.max() <= window[1]:
            inside.append((float(times.min()), name))
    return [name for _, name in sorted(inside)]


def _free_key(merged: List[Any], every: List[Any]) -> Tuple[Any, str, int]:
    """A group key for the merged sequence that no surviving group already has.

    Keyed on the earliest cluster the sequence touches and named for what it is.
    The sub-index walks up until the key is unused, so a stray ceremony group the
    window excluded is never silently absorbed by colliding with it.
    """
    cluster = min(name[0] for name in merged)
    taken = set(every) - set(merged)
    sub = -1
    while (cluster, MERGED_CLASS, sub) in taken:
        sub += 1
    return (cluster, MERGED_CLASS, sub)


def merge_ceremony_sequence(photos_df: pd.DataFrame, all_gallery_df=None,
                            logger=None) -> pd.DataFrame:
    """Fold the groups lying inside the ceremony window into one group.

    Only the three key columns are rewritten, and only for photos inside the
    window. `original_context` is left alone, so every photo still carries what
    it actually shows -- which is what the page's context penalty reads, and
    what `CHRONOLOGY_FIRST_PENALTIES` then relaxes for this group precisely so
    the processional and the vows may share a page. That is why the caller sets
    `original_context` before calling this and not after.

    `group_size` is recomputed, since the groups it describes have changed.

    Args:
        photos_df: The album's photos, already grouped by `_get_groups`.
        all_gallery_df: The whole gallery, which is where the window is read
            from. Falls back to `photos_df` when absent.
        logger: Optional.

    Returns:
        `photos_df`, rebuilt when a merge happened and handed straight back when
        it did not.
    """
    if not CONFIGS.get('merge_ceremony_sequence', True):
        return photos_df

    source = all_gallery_df if all_gallery_df is not None and len(all_gallery_df) else photos_df
    window = ceremony_window(source, logger)
    merged = groups_inside(photos_df, window)

    if len(merged) < 2:
        if logger and window is not None:
            logger.info("Ceremony sequence: nothing to merge -- "
                        f"{len(merged)} group(s) lie wholly inside the window.")
        return photos_df

    every = [name for name, _ in photos_df.groupby(GROUP_KEY)]
    key = _free_key(merged, every)

    photos_df = photos_df.copy()
    rows = pd.Series(False, index=photos_df.index)
    for name in merged:
        match = pd.Series(True, index=photos_df.index)
        for column, value in zip(GROUP_KEY, name):
            match &= photos_df[column] == value
        rows |= match

    for column, value in zip(GROUP_KEY, key):
        photos_df.loc[rows, column] = value
    update_groups_size(photos_df, clusters=['time_cluster', 'cluster_context'])

    if logger:
        logger.info(f"Ceremony sequence: merged {len(merged)} group(s) into {key} "
                    f"over {int(rows.sum())} photos: "
                    + ", ".join(str(name) for name in merged))
    return photos_df
