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

What is kept from the narrator's rule: its time windows. What is added is the
wedding covers' scoring minus everything about a bride and groom
(`key_pages.rank_general_covers`), and the same widening of the opening window
for small galleries.

**Presence is the gallery's main people, not its most frequent group.** The
narrator's `hero.dominant_group` is the most frequent *exact* set of
identities, and a family's most frequent set is rarely the family: whoever
holds the camera is missing from most frames. On 54117560 (a family trip,
mother, son and father) mother-and-son was in 70 photos and all three in 12,
so every frame of all three scored 2/3 of a pair's presence and both covers
went to the pair. The main people are every identity in at least
`cover_main_share` of the identified photos (`main_people`) -- there, all
three -- and the overlap with them is the presence term, with a bonus for a
frame that has every one of them (`_presence`).
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import List, Optional, Tuple

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
    from src.covers.choosing import opening_fraction
    from src.covers.scoring import rank_general_covers
    from src.narrator.env import hero

    if pg.n < 3:
        return -1, -1
    group = main_people(pg) or hero.dominant_group(pg)
    subjects = max(2, len(group))

    def ranked(indices: np.ndarray, exclude: int) -> List[int]:
        indices = [int(i) for i in indices if int(i) != exclude
                   and int(pg.photo_ids[int(i)]) in rows.index]
        if not indices:
            return []
        frame = rows.loc[[int(pg.photo_ids[i]) for i in indices]].reset_index(drop=True)
        presence = [_presence(pg, i, group) for i in indices]
        by_id = {int(pg.photo_ids[i]): i for i in indices}
        return [by_id[int(pid)]
                for pid in rank_general_covers(frame, logger, presence, subjects)]

    opening = ranked(_window(pg, opening_fraction(pg.n), first=True), exclude=-1)
    if not opening:
        return None
    first = opening[0]
    closing = ranked(_window(pg, CLOSING_WINDOW, first=False), exclude=first)
    if not closing:
        return None
    return first, closing[0]


def main_people(pg) -> frozenset:
    """Every identity in at least `cover_main_share` of the identified photos.

    Empty when nobody is identified or nobody is that frequent -- a gallery of
    many people with no one at its centre -- and the caller then falls back
    to the narrator's own group.
    """
    from src.pipeline.select.narrator import settings

    identified = [s for s in pg.cast if s]
    if not identified:
        return frozenset()
    share = float(settings().get('cover_main_share', 0.25))
    counts = Counter(i for s in identified for i in s)
    return frozenset(int(i) for i, c in counts.items() if c >= share * len(identified))


def _presence(pg, i: int, group: frozenset) -> float:
    """Photo i's overlap with ``group``, and `cover_full_group_bonus` if it has all of them.

    The overlap is linear in the people a frame has -- for five main people,
    two, three and four of them are equal steps -- but the whole group is not
    one more step: it is the photo of the family, and a cover should be it
    when there is a good one. On 54117560 the overlap alone left every frame
    of all three a third of a step ahead of a pair, which a pair's sharper
    photograph made up. A group of one is any frame of that person; there is
    no whole to prefer.
    """
    from src.narrator.env import hero
    from src.pipeline.select.narrator import settings

    score = hero._group_score(pg, i, group)
    if len(group) > 1 and group <= (pg.cast[i] or frozenset()):
        score += float(settings().get('cover_full_group_bonus', 0.0))
    return score


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

    The wedding path's `drop_cover_duplicates`, on the narrator's block: the
    burst neighbour of a cover is a different photo and survives in the body.
    A spread left empty is dropped.
    """
    from src.covers.choosing import drop_cover_duplicates

    covers = [pid for field in ('firstPagePhotoIds', 'lastPagePhotoIds')
              for pid in block.get(field, [])]
    if not covers:
        return block
    body = [pid for spread in block.get('spreads', []) for pid in spread.get('photoIds', [])]
    frame = photos[photos[Col.IMAGE_ID].isin(body)]
    cover_rows = photos[photos[Col.IMAGE_ID].isin(covers)]
    kept = set(drop_cover_duplicates(frame, cover_rows, logger)[Col.IMAGE_ID])
    if len(kept) == len(set(body)):
        return block
    spreads = []
    for spread in block.get('spreads', []):
        ids = [pid for pid in spread.get('photoIds', []) if pid in kept]
        if ids:
            spreads.append({**spread, 'photoIds': ids})
    return {**block, 'spreads': spreads}
