"""The opening and closing photos of a narrator album, chosen on our terms.

The vendored policy picks its two "hero" bookends in `narrator.env.hero`: the
earliest and latest quarter of the timeline, ranked by how closely a photo's
people match the gallery's most frequent group and then by the content model's
score. Nothing else counts -- no orientation, no quality of the photograph, no
check that a subject is cut by the border -- and since nearly every frame of a
couple or family session shows that group, the content score alone decides.
53952231 (an engagement session) opened on a portrait with landscapes beside
it in the window, and a near-copy of each cover sat in the body.

**Swapped after composing, not chosen before it.** The policy picks its heroes
at every reset and withholds them from the pages it builds, so the obvious
change -- let it use ours from the start -- reaches into the composition, and
it does not survive contact: on 53952231 the policy given our closing ended
the album early, 7-9 spreads where its own heroes give 10-12, at best-of-1, 4
and 8 alike. It was trained with its own bookends. So it composes with them,
undisturbed, and the covers are replaced in the finished block: ours come out
of the body if they are in it, the narrator's two leave the album, and body
photos repeating the new covers are pruned as on the wedding path.

`src/narrator` itself is untouched: it is copied verbatim from albumNarrator
so it can be diffed against its source.

What is kept from the narrator's rule: its time windows, and its dominant-group
overlap, as the presence term. What is added is the wedding covers' scoring
minus everything about a bride and groom (`key_pages.rank_general_covers`),
and the same widening of the opening window for small galleries.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.pipeline.contracts import Col

#: The narrator's own closing window (`hero.LAST_WINDOW`).
CLOSING_WINDOW = 0.25


def choose_covers(pg, photos: pd.DataFrame, logger=None) -> Optional[Tuple[int, int]]:
    """(opening, closing) photo ids by `rank_general_covers`, or None to keep the narrator's.

    A cover is not worth the album: any failure answers None.
    """
    if logger is None:
        # The cover scoring logs as it goes; without a logger it would raise,
        # and every gallery would quietly keep the narrator's choice.
        logger = logging.getLogger(__name__)
    rows = photos.drop_duplicates(Col.IMAGE_ID).set_index(Col.IMAGE_ID, drop=False)
    try:
        chosen = _choose(pg, rows, logger)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"narrator covers: declined ({type(exc).__name__}: {exc}); "
                       f"keeping the narrator's own bookends")
        return None
    if chosen is None or min(chosen) < 0:
        return None
    return int(pg.photo_ids[chosen[0]]), int(pg.photo_ids[chosen[1]])


def apply_covers(block: dict, covers: Tuple[int, int], photos: pd.DataFrame,
                 logger=None) -> dict:
    """The block with ``covers`` as its opening and closing.

    The new covers leave the body; the narrator's own two leave the album --
    they were never in its body, and putting a photo back into a page the
    policy composed without it is a decision this module does not make. Body
    photos repeating a new cover are then pruned (`prune_cover_duplicates`).
    """
    opening, closing = covers
    previous = (block.get('firstPagePhotoIds') or [None])[0],         (block.get('lastPagePhotoIds') or [None])[0]
    spreads = []
    for spread in block.get('spreads', []):
        ids = [pid for pid in spread.get('photoIds', []) if pid not in (opening, closing)]
        if ids:
            spreads.append({**spread, 'photoIds': ids})
    out = {**block, 'spreads': spreads,
           'firstPagePhotoIds': [opening], 'lastPagePhotoIds': [closing]}
    if logger and (opening, closing) != previous:
        logger.info(f"narrator covers: opening {previous[0]} -> {opening}, "
                    f"closing {previous[1]} -> {closing}")
    return prune_cover_duplicates(out, photos, logger)


def _choose(pg, rows: pd.DataFrame, logger) -> Optional[Tuple[int, int]]:
    from src.core.key_pages import _opening_fraction, rank_general_covers
    from src.narrator.env import hero

    if pg.n < 3:
        return -1, -1
    group = hero.dominant_group(pg)

    def ranked(indices: np.ndarray, exclude: int) -> List[int]:
        indices = [int(i) for i in indices if int(i) != exclude
                   and int(pg.photo_ids[int(i)]) in rows.index]
        if not indices:
            return []
        frame = rows.loc[[int(pg.photo_ids[i]) for i in indices]].reset_index(drop=True)
        presence = [hero._group_score(pg, i, group) for i in indices]
        by_id = {int(pg.photo_ids[i]): i for i in indices}
        return [by_id[int(pid)] for pid in rank_general_covers(frame, logger, presence)]

    opening = ranked(_window(pg, _opening_fraction(pg.n), first=True), exclude=-1)
    if not opening:
        return None
    first = opening[0]
    closing = ranked(_window(pg, CLOSING_WINDOW, first=False), exclude=first)
    if not closing:
        return None
    return first, closing[0]


def _window(pg, fraction: float, first: bool) -> np.ndarray:
    """`hero._window_candidates` with the fraction as a parameter."""
    if pg.time_valid.any() and np.unique(pg.time_norm[pg.time_valid]).size > 1:
        t = pg.time_norm
        selected = ((t <= fraction) if first else (t >= 1.0 - fraction)) & pg.time_valid
        if selected.any():
            return np.nonzero(selected)[0]
    k = max(1, int(round(pg.n * fraction)))
    return np.arange(0, k) if first else np.arange(pg.n - k, pg.n)


def prune_cover_duplicates(block: dict, photos: pd.DataFrame, logger=None) -> dict:
    """The block without body photos that repeat a cover.

    The wedding path's `_drop_cover_duplicates`, on the narrator's block: the
    burst neighbour of a cover is a different photo and survives in the body.
    A spread left empty is dropped.
    """
    from src.core.key_pages import _drop_cover_duplicates

    covers = [pid for field in ('firstPagePhotoIds', 'lastPagePhotoIds')
              for pid in block.get(field, [])]
    if not covers:
        return block
    body = [pid for spread in block.get('spreads', []) for pid in spread.get('photoIds', [])]
    frame = photos[photos[Col.IMAGE_ID].isin(body)]
    cover_rows = photos[photos[Col.IMAGE_ID].isin(covers)]
    kept = set(_drop_cover_duplicates(frame, cover_rows, logger)[Col.IMAGE_ID])
    if len(kept) == len(set(body)):
        return block
    spreads = []
    for spread in block.get('spreads', []):
        ids = [pid for pid in spread.get('photoIds', []) if pid in kept]
        if ids:
            spreads.append({**spread, 'photoIds': ids})
    return {**block, 'spreads': spreads}
