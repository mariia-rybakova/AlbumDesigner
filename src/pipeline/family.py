"""Seeding a parents album, by standing in for the user's own selection.

`enrich.variants` can plan a second album with `focus='parents'`, which changes
the per-category spread budget. That alone moves the *shape* of the album and
not its *content*: the focus profile asks for more family spreads, and the
picker then fills them with whatever ranked best, which on a gallery the couple
dominates is the couple again.

The signal that actually moves content is the one the user has: `aiMetadata`.
`select.preselect` honours it unconditionally -- `photoIds` are committed before
any ranking, and each entry of `personIds` is guaranteed its photos -- and
`person_score` ranks every remaining photo by the share of its people who are
named. So a parents album is composed by handing selection a **pseudo
selection**: the photos of the parents, and the people the parents are
photographed with.

Nothing here is a new selection mechanism. It is the existing one, driven from a
fact the pipeline derived (`enrich.parents`) rather than from the request.

Two properties are worth stating, because they are the reason this is not just
"take the top N photos containing a parent":

**Both sides, equally.** A wedding has two families and an album that is 80%
one of them is a worse album than one that is even, however the ranking falls.
Where both sides were resolved the seed takes the same number from each.

**Matched moments where possible.** Given the choice, the pair taken from the
two sides comes from the *same* `cluster_context` -- a portrait for each, an
aisle walk for each, a parent dance for each. The album then reads as two
families at one wedding rather than two unrelated runs of photos. Contexts that
only one side has are used to fill what the pairing leaves.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from src.pipeline import subject
from src.pipeline.contracts import Col
from utils.configs import CONFIGS


def settings() -> dict:
    return CONFIGS.get("family_album", {}) or {}


def enabled() -> bool:
    return bool(settings().get("enabled", False))


def _people(value) -> set:
    return set(value) if isinstance(value, (list, tuple, set)) else set()


def _clean(ids: Optional[Iterable]) -> Tuple[int, ...]:
    """Identity ids as a tuple of ints, dropping nulls and duplicates."""
    seen, out = set(), []
    for value in ids or ():
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        number = int(value)
        if number not in seen:
            seen.add(number)
            out.append(number)
    return tuple(out)


# -- who the parents are photographed with -----------------------------------


def close_contacts(photos: pd.DataFrame, parent_ids: Iterable,
                   exclude: Iterable = (), max_contacts: Optional[int] = None,
                   min_together: Optional[int] = None) -> Tuple[int, ...]:
    """Identities that keep appearing in frame with a parent, most first.

    Co-appearance in the gallery's own photos, not the `socialCircles` proto.
    The proto is a clustering someone else made and its grouping is not the
    question being asked here -- "who is this parent photographed with" is
    answerable directly from `persons_ids`, and is answerable on galleries
    that ship no circles at all.

    ``exclude`` is normally the couple. They appear beside their parents in
    most of the frames that matter, so they would head this list on every
    gallery -- and because `person_score` is the *share* of a photo's people
    who are named, naming them would score the couple's own photos highly and
    pull the album back to the subject the first album already has.

    ``min_together`` is the noise floor: one shared frame is a guest who walked
    past, not a relation.
    """
    parents = set(_clean(parent_ids))
    if photos is None or photos.empty or not parents:
        return ()
    if Col.PERSONS_IDS not in photos.columns:
        return ()

    config = settings()
    if max_contacts is None:
        max_contacts = int(config.get("contacts_max", 6))
    if min_together is None:
        min_together = int(config.get("contacts_min_together", 3))
    blocked = set(_clean(exclude)) | parents

    together: Counter = Counter()
    for value in photos[Col.PERSONS_IDS]:
        present = _people(value)
        if not present & parents:
            continue
        for identity in present - blocked:
            together[identity] += 1

    ranked = [identity for identity, count in together.most_common()
              if count >= min_together]
    return _clean(ranked[:max_contacts])


# -- the photos that stand in for a user's picks -----------------------------


def _side_frame(photos: pd.DataFrame, ids: Iterable) -> pd.DataFrame:
    """Photos holding at least one identity from this side."""
    wanted = set(_clean(ids))
    if not wanted:
        return photos.iloc[0:0]
    holds = photos[Col.PERSONS_IDS].apply(lambda v: bool(_people(v) & wanted))
    return photos[holds]


def _by_context(frame: pd.DataFrame) -> Dict[Any, List[Any]]:
    """Image ids per `cluster_context`, best-ranked first within each.

    ``image_order`` is the content model's `selectionOrder`, where **0 is
    best** -- the same direction `select.preselect` breaks its own ties on.
    """
    if frame.empty:
        return {}
    ordered = frame
    if Col.IMAGE_ORDER in frame.columns:
        ordered = frame.sort_values(Col.IMAGE_ORDER, ascending=True, kind="stable")
    grouped: Dict[Any, List[Any]] = {}
    for context_name, rows in ordered.groupby(Col.CLUSTER_CONTEXT, sort=False):
        grouped[context_name] = list(rows[Col.IMAGE_ID])
    return grouped


def balanced_parent_photos(photos: pd.DataFrame, bride_parents: Iterable,
                           groom_parents: Iterable,
                           total: Optional[int] = None) -> Tuple[int, ...]:
    """Photos of the parents: the same number from each side that has any.

    Pairs first -- one photo from each side out of the *same* `cluster_context`
    -- so the album carries the two families through the same moments. Then,
    for whatever the pairing could not fill, each side's next best, still in
    equal numbers.

    With one side resolved there is nothing to balance and the whole allowance
    goes to that side.
    """
    if photos is None or photos.empty or Col.PERSONS_IDS not in photos.columns:
        return ()
    if Col.CLUSTER_CONTEXT not in photos.columns or Col.IMAGE_ID not in photos.columns:
        return ()

    if total is None:
        total = int(settings().get("max_photos", 8))
    if total <= 0:
        return ()

    sides = [ids for ids in (_clean(bride_parents), _clean(groom_parents)) if ids]
    if not sides:
        return ()

    frames = [_side_frame(photos, ids) for ids in sides]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return ()

    # `other` and `None` are budgeted at 0% in the focus CSV because they carry
    # nothing worth a spread, and they are the *largest* buckets, so pairing on
    # raw counts lands there first: on 49996919 four of the eight seeded photos
    # came from `other` and `None`. `select.preselect` already learned this
    # covering named identities -- see `subject.identity_tiers` -- and a pseudo
    # selection is committed the same unconditional way, so it has to be at
    # least as careful. Real classes first; the unknown ones only fill what is
    # left, and only for a side that has nothing better.
    preferred = [frame[~frame[Col.CLUSTER_CONTEXT].apply(subject.is_unknown_category)]
                 for frame in frames]
    spare = [frame[frame[Col.CLUSTER_CONTEXT].apply(subject.is_unknown_category)]
             for frame in frames]
    preferred = [good if not good.empty else frames[i]
                 for i, good in enumerate(preferred)]

    # Equal means equal. The allowance per side is capped by what the
    # *thinnest* side can actually supply, so a family photographed twice as
    # often cannot take twice the album -- taking each side's share
    # independently is what produced 4 against 1 on a gallery holding twelve
    # bride-side frames and one groom-side. The cost is a smaller seed where
    # one side is thin, which is the right way round: the seed is a floor under
    # the album's family content, and a lopsided floor is worse than a low one.
    per_side = max(1, total // len(frames))
    per_side = min(per_side, min(len(frame) for frame in preferred))
    grouped = [_by_context(frame) for frame in preferred]
    reserves = [_by_context(frame) for frame in spare]
    taken: List[List[Any]] = [[] for _ in frames]
    used = set()

    def take(index: int, image_id) -> bool:
        if image_id in used or len(taken[index]) >= per_side:
            return False
        used.add(image_id)
        taken[index].append(image_id)
        return True

    if len(frames) > 1:
        # Contexts both sides have, the best-supplied first -- `min` because a
        # context only pairs as far as the thinner side can supply it.
        shared = set(grouped[0])
        for other in grouped[1:]:
            shared &= set(other)
        order = sorted(shared, key=lambda name: (
            -min(len(group[name]) for group in grouped), str(name)))
        for context_name in order:
            if all(len(rows) >= per_side for rows in taken):
                break
            # One each, so a context can never be represented for one family
            # and not the other.
            picks = []
            for index, group in enumerate(grouped):
                nxt = next((i for i in group[context_name] if i not in used), None)
                if nxt is None:
                    picks = []
                    break
                picks.append((index, nxt))
            for index, image_id in picks:
                take(index, image_id)

    # Whatever the pairing left, from each side's own best -- real classes
    # exhausted before an `other` or a `None` is considered at all.
    for index in range(len(grouped)):
        if len(taken[index]) >= per_side:
            continue
        for group in (grouped[index], reserves[index]):
            for image_id in [i for rows in group.values() for i in rows]:
                if len(taken[index]) >= per_side:
                    break
                take(index, image_id)

    # Interleaved so a truncation downstream still leaves both sides present.
    out: List[Any] = []
    for position in range(per_side):
        for rows in taken:
            if position < len(rows):
                out.append(rows[position])
    return _clean(out)


# -- the whole seed ----------------------------------------------------------


def parents_seed(photos: pd.DataFrame, bride_parents: Iterable,
                 groom_parents: Iterable, bride_id=None, groom_id=None
                 ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """``(photo_ids, person_ids)`` standing in for a user's aiMetadata.

    Empty photo ids mean the gallery resolved parents but holds no photo of
    them the seed could use, in which case the caller should leave the variant
    as it was: a focus change with nothing behind it is still a valid album,
    and an empty pseudo selection is not worth a special case downstream.
    """
    parents = _clean(tuple(_clean(bride_parents)) + tuple(_clean(groom_parents)))
    if not parents:
        return (), ()

    picks = balanced_parent_photos(photos, bride_parents, groom_parents)
    if not picks:
        # Parents named but nothing to show them in. Returning the people
        # anyway would be a seed the caller is going to discard, since it
        # stands the variant down on empty picks -- say nothing instead of
        # half a thing.
        return (), ()

    couple = _clean((bride_id, groom_id))
    people = list(parents)
    if settings().get("include_couple", False):
        people.extend(couple)
    people.extend(close_contacts(photos, parents, exclude=couple))
    return picks, _clean(people)
