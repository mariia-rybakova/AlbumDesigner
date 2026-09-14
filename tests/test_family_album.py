"""Seeding the parents album from the identities `enrich.parents` resolved.

A focus changes an album's shape; these tests are about its content. The lever
is a pseudo `aiMetadata` -- `select.preselect` commits `photoIds` before any
ranking and guarantees each `personIds` entry its photos -- so what is checked
here is the seed that gets built, and the gate that decides whether to build
one at all.

    python -m pytest tests/test_family_album.py -v
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import family  # noqa: E402
from src.pipeline.contracts import Col  # noqa: E402
from utils.configs import CONFIGS  # noqa: E402

BRIDE, GROOM = 1, 2
#: Bride's mother and father, groom's mother and father.
BM, BF, GM, GF = 10, 11, 20, 21
#: A sibling who is in most of the bride-side frames, and a passing guest.
SIBLING, GUEST = 30, 31


def gallery(rows):
    """rows: (image_id, persons_ids, cluster_context, image_order)."""
    return pd.DataFrame([
        {Col.IMAGE_ID: i, Col.PERSONS_IDS: list(p),
         Col.CLUSTER_CONTEXT: c, Col.IMAGE_ORDER: o}
        for i, p, c, o in rows])


def symmetric_gallery():
    """Both families, photographed in the same three moments."""
    rows = []
    image_id = 100
    for context in ("parents portrait", "first dance", "walking the aisle"):
        for side in ((BM, BF), (GM, GF)):
            for n in range(3):
                rows.append((image_id, [BRIDE, *side], context, image_id - 100))
                image_id += 1
    return gallery(rows)


# -- the gate ---------------------------------------------------------------


def test_no_parents_means_no_seed():
    """The normal outcome on a gallery whose candidates cannot be separated.
    It must stay a working two-album plan, not a failure."""
    photos = symmetric_gallery()

    picks, people = family.parents_seed(photos, (), (), BRIDE, GROOM)

    assert picks == ()
    assert people == ()


def test_one_side_is_enough_to_seed():
    photos = symmetric_gallery()

    picks, people = family.parents_seed(photos, (BM, BF), (), BRIDE, GROOM)

    assert picks, "parents on one side still build a parents album"
    assert BM in people and BF in people


def test_a_gallery_without_identities_does_not_raise():
    photos = symmetric_gallery().drop(columns=[Col.PERSONS_IDS])

    assert family.parents_seed(photos, (BM,), (GM,), BRIDE, GROOM) == ((), ())


# -- both sides, equally ----------------------------------------------------


def _side_counts(picks, photos):
    by_id = {int(r[Col.IMAGE_ID]): set(r[Col.PERSONS_IDS])
             for _, r in photos.iterrows()}
    bride_side = sum(1 for i in picks if by_id[i] & {BM, BF})
    groom_side = sum(1 for i in picks if by_id[i] & {GM, GF})
    return bride_side, groom_side


def test_both_sides_get_the_same_number_of_photos():
    """An album that is 80% one family is a worse album than an even one,
    however the ranking falls."""
    photos = symmetric_gallery()

    picks, _ = family.parents_seed(photos, (BM, BF), (GM, GF), BRIDE, GROOM)

    assert _side_counts(picks, photos) == (len(picks) // 2, len(picks) // 2)


def test_the_sides_are_matched_on_context_where_both_have_one():
    """Two families at one wedding, not two unrelated runs of photos."""
    photos = symmetric_gallery()

    picks, _ = family.parents_seed(photos, (BM, BF), (GM, GF), BRIDE, GROOM)

    by_id = {int(r[Col.IMAGE_ID]): r for _, r in photos.iterrows()}
    contexts = {"bride": set(), "groom": set()}
    for image_id in picks:
        row = by_id[image_id]
        side = "bride" if set(row[Col.PERSONS_IDS]) & {BM, BF} else "groom"
        contexts[side].add(row[Col.CLUSTER_CONTEXT])

    # Every moment the seed takes for one family it also takes for the other.
    # Not one photo each per context -- once the pairs are placed the rest of
    # the allowance is filled from each side's own best, which can double up
    # inside a context -- but no context belonging to one family alone.
    assert contexts["bride"] == contexts["groom"], (
        "a moment was taken for one family and not the other: "
        f"bride={sorted(contexts['bride'])} groom={sorted(contexts['groom'])}")


def test_a_lopsided_gallery_still_balances():
    """The thinner side is what the pairing can supply, so neither side runs
    away with the album just because it was photographed more."""
    rows = [(200 + n, [BRIDE, BM, BF], "parents portrait", n) for n in range(12)]
    rows += [(300, [BRIDE, GM, GF], "parents portrait", 0)]
    photos = gallery(rows)

    picks, _ = family.parents_seed(photos, (BM, BF), (GM, GF), BRIDE, GROOM)
    bride_side, groom_side = _side_counts(picks, photos)

    assert groom_side >= 1
    assert bride_side <= groom_side + 1, \
        f"one family took the album: {bride_side} vs {groom_side}"


# -- the parents' own circle -------------------------------------------------


def test_close_contacts_come_from_co_appearance():
    rows = [(400 + n, [BM, BF, SIBLING], "parents portrait", n) for n in range(5)]
    rows += [(500, [BM, GUEST], "party", 0)]
    photos = gallery(rows)

    contacts = family.close_contacts(photos, (BM, BF), exclude=(BRIDE, GROOM))

    assert SIBLING in contacts, "five shared frames is a relation"
    assert GUEST not in contacts, "one shared frame is someone walking past"


def test_the_couple_are_left_out_of_the_person_ids():
    """`person_score` is the share of a photo's people who are named, so
    naming the couple scores the couple's own photos and pulls this album
    back to the subject the first album already has."""
    photos = symmetric_gallery()

    _picks, people = family.parents_seed(photos, (BM, BF), (GM, GF), BRIDE, GROOM)

    assert BRIDE not in people and GROOM not in people
    assert {BM, BF, GM, GF} <= set(people)


def test_the_couple_can_be_put_back():
    photos = symmetric_gallery()
    original = CONFIGS['family_album']['include_couple']
    CONFIGS['family_album']['include_couple'] = True
    try:
        _picks, people = family.parents_seed(photos, (BM, BF), (GM, GF), BRIDE, GROOM)
    finally:
        CONFIGS['family_album']['include_couple'] = original

    assert BRIDE in people and GROOM in people


def test_parents_are_never_their_own_contacts():
    photos = symmetric_gallery()

    contacts = family.close_contacts(photos, (BM, BF), exclude=(BRIDE, GROOM))

    assert not ({BM, BF} & set(contacts))


# -- the plan ----------------------------------------------------------------


def planned(photos, facts_kwargs, auto=True):
    from src.pipeline.contracts import AlbumContext, GalleryFacts
    from src.pipeline.enrich.variants import VariantsSubStage

    context = AlbumContext(photos=photos,
                           request={'autoAlbums': auto} if auto else {},
                           facts=GalleryFacts(**facts_kwargs))
    VariantsSubStage()(context)
    return context.variants


def test_without_parents_the_plan_is_what_it_always_was():
    """`keep as it is now` -- two focus-only variants, nothing seeded."""
    variants = planned(symmetric_gallery(),
                       dict(bride_id=BRIDE, groom_id=GROOM))

    assert [v.name for v in variants] == ["brideAndGroom", "parents"]
    assert all(v.photo_ids is None and v.person_ids is None for v in variants)


def test_with_parents_only_the_second_album_is_seeded():
    variants = planned(symmetric_gallery(),
                       dict(bride_id=BRIDE, groom_id=GROOM,
                            bride_parents=(BM, BF), groom_parents=(GM, GF)))

    couple, parents = variants
    assert couple.photo_ids is None, "the couple's album is left alone"
    assert couple.person_ids is None
    assert parents.photo_ids, "the parents album carries a pseudo selection"
    assert {BM, BF, GM, GF} <= set(parents.person_ids)


def test_one_side_resolved_still_seeds():
    variants = planned(symmetric_gallery(),
                       dict(bride_id=BRIDE, groom_id=GROOM,
                            bride_parents=(BM, BF)))

    assert variants[1].photo_ids


def test_the_single_album_default_is_never_seeded():
    variants = planned(symmetric_gallery(),
                       dict(bride_id=BRIDE, groom_id=GROOM,
                            bride_parents=(BM, BF), groom_parents=(GM, GF)),
                       auto=False)

    assert len(variants) == 1
    assert variants[0].photo_ids is None and variants[0].focus is None


def test_the_seed_can_be_switched_off():
    original = CONFIGS['family_album']['enabled']
    CONFIGS['family_album']['enabled'] = False
    try:
        variants = planned(symmetric_gallery(),
                           dict(bride_id=BRIDE, groom_id=GROOM,
                                bride_parents=(BM, BF), groom_parents=(GM, GF)))
    finally:
        CONFIGS['family_album']['enabled'] = original

    assert all(v.photo_ids is None for v in variants)


# -- the overlay -------------------------------------------------------------


def test_applying_the_variant_overrides_the_hints():
    from src.pipeline.albums import AlbumVariant
    from src.pipeline.contracts import AiHints, AlbumContext

    context = AlbumContext(hints=AiHints(photo_ids=[9, 9, 9], person_ids=[7],
                                         focus=['everyoneElse'], present=True))
    AlbumVariant('parents', focus=('parents',),
                 photo_ids=(1, 2), person_ids=(BM, BF)).apply(context)

    assert context.hints.photo_ids == [1, 2]
    assert context.hints.person_ids == [BM, BF]
    assert context.hints.focus == ['parents']


def test_applying_a_variant_never_changes_the_routing():
    """`present` is what `select.route` splits manual from AI on. A variant
    that flipped it would not steer an album, it would send the request down a
    different path."""
    from src.pipeline.albums import AlbumVariant
    from src.pipeline.contracts import AiHints, AlbumContext

    for present in (True, False):
        context = AlbumContext(hints=AiHints(present=present))
        AlbumVariant('parents', photo_ids=(1,), person_ids=(BM,)).apply(context)
        assert context.hints.present is present


def test_a_variant_with_no_overrides_leaves_the_hints_alone():
    from src.pipeline.albums import AlbumVariant
    from src.pipeline.contracts import AiHints, AlbumContext

    hints = AiHints(photo_ids=[5], person_ids=[6], focus=['everyoneElse'])
    context = AlbumContext(hints=hints)
    AlbumVariant('requested').apply(context)

    assert context.hints is hints, "the default must not even replace the object"


def test_the_seed_avoids_the_classes_budgeted_at_nothing():
    """`other` and `None` carry nothing worth a spread and are the *largest*
    buckets, so pairing on raw counts lands there first -- on 49996919 four of
    the eight seeded photos came from them before this rule. A pseudo selection
    is committed as unconditionally as a real one, so it has to be at least as
    careful as `subject.identity_tiers` already is.
    """
    rows = []
    image_id = 600
    for context in ("other", "None"):
        for side in ((BM, BF), (GM, GF)):
            for n in range(6):
                rows.append((image_id, [BRIDE, *side], context, n))
                image_id += 1
    for side in ((BM, BF), (GM, GF)):
        for n in range(2):
            rows.append((image_id, [BRIDE, *side], "parents portrait", 50 + n))
            image_id += 1
    photos = gallery(rows)

    picks, _ = family.parents_seed(photos, (BM, BF), (GM, GF), BRIDE, GROOM)

    by_id = {int(r[Col.IMAGE_ID]): r for _, r in photos.iterrows()}
    chosen = [by_id[i][Col.CLUSTER_CONTEXT] for i in picks]
    assert "parents portrait" in chosen
    assert chosen.count("parents portrait") == 4, \
        f"a real class was available and went unused: {chosen}"


def test_a_side_with_only_unknown_classes_is_not_dropped():
    """Preferring real classes must not cost a family its place in the album
    entirely -- that would be a worse failure than an `other` photo."""
    rows = [(700 + n, [BRIDE, BM, BF], "parents portrait", n) for n in range(4)]
    rows += [(800 + n, [BRIDE, GM, GF], "other", n) for n in range(4)]
    photos = gallery(rows)

    picks, _ = family.parents_seed(photos, (BM, BF), (GM, GF), BRIDE, GROOM)
    bride_side, groom_side = _side_counts(picks, photos)

    assert groom_side >= 1, "the groom's family disappeared from its own album"
    assert bride_side == groom_side
