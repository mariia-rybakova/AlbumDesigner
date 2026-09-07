"""Opening and closing "hero" pages: a single photo each, bookending the album.

Spec:
  * first page = one photo drawn from the earliest 25% of the gallery (by time)
  * last page  = one photo drawn from the latest 25%
  * both should feature the group of people that appears most often in the gallery

The choice is fully determined by the data, so it is made deterministically at reset rather
than left to the policy: the two photos are withheld from the assignable pool and the env
prepends/appends their pages, so the policy composes only the album body. This follows the
project's "structure via construction, not via penalty" principle.
"""
from __future__ import annotations

import numpy as np

from ..data.schema import PackedGallery

FIRST_WINDOW = 0.25   # fraction of the timeline the opening photo is drawn from
LAST_WINDOW = 0.25


def dominant_group(pg: PackedGallery) -> frozenset[int]:
    """The most frequently occurring set of identities across the gallery.

    Exact identity-set match, so "Alice+Bob" and "Alice" are different groups -- the intent is
    a recurring *group*, not a popular individual. Falls back to the single most frequent
    identity when no multi-photo group exists, and to the empty set when nobody is tagged.
    """
    counts: dict[frozenset[int], int] = {}
    for s in pg.cast:
        if s:
            counts[frozenset(s)] = counts.get(frozenset(s), 0) + 1
    if counts:
        best = max(counts.items(), key=lambda kv: (kv[1], len(kv[0])))
        if best[1] > 1:
            return best[0]
    if pg.n_persons:
        per_id = pg.cast_multihot.sum(0)
        if per_id.max() > 0:
            return frozenset({int(pg.person_ids[int(np.argmax(per_id))])})
    return frozenset()


def _group_score(pg: PackedGallery, i: int, group: frozenset[int]) -> float:
    """Jaccard overlap between photo i's cast and the dominant group (1.0 = exact match)."""
    if not group:
        return 0.0
    s = pg.cast[i]
    if not s:
        return 0.0
    inter = len(s & group)
    union = len(s | group)
    return inter / union if union else 0.0


def _window_candidates(pg: PackedGallery, first: bool) -> np.ndarray:
    """Indices inside the leading / trailing time window, by time_norm when times are usable."""
    n = pg.n
    # Needs actual SPREAD, not merely valid timestamps: ~8% of real galleries carry one
    # identical timestamp on every photo, which normalizes to time_norm == 0 everywhere. That
    # makes the leading window match the whole gallery and the trailing window match nothing,
    # so both ends must fall back to position instead.
    if pg.time_valid.any() and np.unique(pg.time_norm[pg.time_valid]).size > 1:
        t = pg.time_norm
        sel = ((t <= FIRST_WINDOW) if first else (t >= 1.0 - LAST_WINDOW)) & pg.time_valid
        if sel.any():
            return np.nonzero(sel)[0]
    # no usable time spread: fall back to position in the gallery (which ingest keeps in
    # photoId order, the closest available proxy for capture order)
    k = max(1, int(round(n * (FIRST_WINDOW if first else LAST_WINDOW))))
    return np.arange(0, k) if first else np.arange(n - k, n)


def pick_hero_photos(pg: PackedGallery) -> tuple[int, int]:
    """(opening, closing) photo indices. Returns (-1, -1) if the gallery is too small.

    Ranked by dominant-group overlap first, then per-photo ``selection`` score as the
    tiebreak, so among equally on-cast photos the better one opens the album.
    """
    if pg.n < 3:
        return -1, -1
    group = dominant_group(pg)

    def best_of(cands: np.ndarray, exclude: int) -> int:
        cands = np.array([i for i in cands if i != exclude], dtype=np.int64)
        if cands.size == 0:
            return -1
        keys = [(_group_score(pg, int(i), group), float(pg.selection[int(i)])) for i in cands]
        return int(cands[int(np.argmax([k[0] * 1000.0 + k[1] for k in keys]))])

    first = best_of(_window_candidates(pg, first=True), exclude=-1)
    last = best_of(_window_candidates(pg, first=False), exclude=first)
    if first < 0 or last < 0 or first == last:
        return -1, -1
    return first, last
