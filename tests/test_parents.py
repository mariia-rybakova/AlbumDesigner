"""Tests for `enrich.parents` -- naming the parents, or declining to.

The old rule's two discriminating tests were both broken, and both failures
are pinned here so a regression cannot bring them back: its age window
accepted +3..+27 years (the sibling band) and rejected +28 and up (the actual
parent band), and its "must be a couple in the social circles" test collapsed
to "either of them appears in any circle at all".

    python -m pytest tests/test_parents.py -v
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import ENRICH, AlbumContext, Col  # noqa: E402
from src.pipeline.contracts import GalleryFacts  # noqa: E402
from src.pipeline.enrich import parents  # noqa: E402
from src.pipeline.registry import get  # noqa: E402
from utils.configs import CONFIGS  # noqa: E402

BRIDE, GROOM = 1, 2
MOTHER, FATHER = 10, 11          # the bride's, old and on her side
GROOM_MOTHER, GROOM_FATHER = 20, 21
BRIDESMAID, GUEST = 30, 31
#: A wedding is mostly young guests. The real galleries carry 27-68
#: identities, and the age *rank* floor is relative to that population -- a
#: fixture with four parents among eight identities puts half of them below
#: the floor, which says nothing about the detector.
CROWD = tuple(range(40, 52))

CLASS = {"portrait": 21, "bride getting dressed": 3, "getting hair-makeup": 14,
         "walking the aisle": 28, "ceremony": 6, "dancing": 8, "first dance": 11,
         "bride party": 4, "groom party": 16, "full party": 13, "suit": 25,
         "bride and groom": 2}


def quiet():
    logger = logging.getLogger("parents-test")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    return logger


# -- fixtures ---------------------------------------------------------------


def gallery(rows):
    """``rows`` of ``(context, people)``, in time order."""
    frame = pd.DataFrame([
        {Col.IMAGE_ID: 1000 + i,
         Col.CLUSTER_CONTEXT: context,
         Col.IMAGE_CLASS: CLASS[context],
         Col.PERSONS_IDS: list(people),
         Col.GENERAL_TIME: float(i),
         Col.MODEL_VERSION: 2,
         Col.EMBEDDING: np.full(768, 1.0 / 768 ** 0.5, dtype=np.float32),
         Col.BRIDE_ID: BRIDE, Col.GROOM_ID: GROOM}
        for i, (context, people) in enumerate(rows)
    ])
    return frame


def details(ages):
    """``ages`` maps identity -> (age, gender)."""
    return pd.DataFrame([{"identity_id": i, "age": float(a), "gender": g}
                         for i, (a, g) in ages.items()])


def circles(groups):
    return pd.DataFrame([{"circle_index": i, "identity_ids": list(ids),
                          "num_ids": len(ids)}
                         for i, ids in enumerate(groups)])


AGES = {BRIDE: (30, 1), GROOM: (32, 0),
        MOTHER: (58, 1), FATHER: (60, 0),
        GROOM_MOTHER: (57, 1), GROOM_FATHER: (61, 0),
        BRIDESMAID: (27, 1), GUEST: (35, 0),
        **{i: (24 + (i - 40) * 1.5, i % 2) for i in CROWD}}


def a_normal_wedding():
    """A gallery where both sides are well covered.

    The bride's parents are in her prep and walk her up the aisle; the groom's
    are with him; a bridesmaid is in the prep too but dominates `bride party`,
    which is the confusion the detector has to resolve.
    """
    rows = []
    rows += [("bride getting dressed", [BRIDE, MOTHER]) for _ in range(4)]
    rows += [("getting hair-makeup", [BRIDE, MOTHER, BRIDESMAID]) for _ in range(3)]
    rows += [("walking the aisle", [BRIDE, FATHER]) for _ in range(3)]
    rows += [("portrait", [BRIDE, MOTHER, FATHER]) for _ in range(6)]
    rows += [("portrait", [BRIDE, GROOM, MOTHER, FATHER]) for _ in range(4)]
    rows += [("suit", [GROOM, GROOM_FATHER]) for _ in range(4)]
    rows += [("portrait", [GROOM, GROOM_MOTHER, GROOM_FATHER]) for _ in range(6)]
    rows += [("portrait", [GROOM, GROOM_MOTHER]) for _ in range(5)]
    rows += [("bride party", [BRIDE, BRIDESMAID]) for _ in range(30)]
    rows += [("full party", [BRIDE, GROOM, BRIDESMAID, GUEST]) for _ in range(6)]
    rows += [("ceremony", [BRIDE, GROOM, GUEST]) for _ in range(8)]
    rows += [("dancing", [BRIDE, GROOM]) for _ in range(5)]
    # The crowd: enough young guests that the age rank means what it means on
    # a real gallery. They are also candidates, and must not be named.
    for guest in CROWD:
        rows += [("ceremony", [BRIDE, GROOM, guest]) for _ in range(3)]
        rows += [("dancing", [guest, BRIDE]) for _ in range(3)]
    return gallery(rows)


def resolve(frame, ages=None, groups=None, quiet_queries=True, monkeypatch=None):
    """Measure and resolve, with the CLIP term neutralised by default.

    The query term is a second opinion on age worth 0.15; the tests are about
    the structural indicators, so it is zeroed unless a test is about it. That
    also keeps them hermetic -- they do not need the concept bins on disk.
    """
    if quiet_queries:
        original = parents.concept_scores
        parents.concept_scores = lambda photos, concept: np.zeros(len(photos))
    try:
        candidates = parents.measure(
            frame, details(ages or AGES), circles(groups or []), BRIDE, GROOM)
        return candidates, parents.resolve(candidates)
    finally:
        if quiet_queries:
            parents.concept_scores = original


# -- the outcome is three-valued -------------------------------------------


def test_the_parents_are_named_as_identities():
    """The point of the rewrite: an answer you can act on, not a photo label."""
    _, outcome = resolve(a_normal_wedding())

    assert MOTHER in outcome.bride_parents
    assert FATHER in outcome.bride_parents
    assert GROOM_MOTHER in outcome.groom_parents
    assert GROOM_FATHER in outcome.groom_parents


def test_the_wedding_party_is_not_mistaken_for_a_parent():
    """The bridesmaid is in the prep and photographed constantly with the
    bride -- she shares every indicator except age and the party classes."""
    _, outcome = resolve(a_normal_wedding())

    assert BRIDESMAID not in outcome.all_parents()
    assert GUEST not in outcome.all_parents()


def test_indistinguishable_candidates_are_left_unresolved():
    """Better no mark than a false mark.

    Three candidates alike in every indicator and only two slots: any pair we
    take is a guess about the third, so none is named. Two alike candidates
    with two slots is *not* this case -- there is nothing to choose between
    them and both are taken.
    """
    rows = []
    for triplet in (MOTHER, FATHER, GUEST):
        # Twelve each, so all three clear the score floor comfortably and it
        # is the separation test that decides, not the floor.
        rows += [("portrait", [BRIDE, triplet]) for _ in range(12)]
    alike = {**AGES, MOTHER: (58, 1), FATHER: (58, 0), GUEST: (58, 0)}

    _, outcome = resolve(gallery(rows), ages=alike)

    assert not outcome.bride_parents
    assert "cannot separate" in outcome.inconclusive["bride"]


def test_two_alike_candidates_filling_both_slots_are_both_taken():
    """The other side of the rule above: no third candidate, no ambiguity."""
    rows = [("portrait", [BRIDE, MOTHER]) for _ in range(8)]
    rows += [("portrait", [BRIDE, FATHER]) for _ in range(8)]
    twins = {**AGES, MOTHER: (58, 1), FATHER: (58, 0)}

    _, outcome = resolve(gallery(rows), ages=twins)

    assert set(outcome.bride_parents) == {MOTHER, FATHER}


def test_a_gallery_with_no_evidence_resolves_nothing():
    """52894932 in miniature: no prep, no aisle, no party, no dances. Every
    candidate is just an age, and age alone is not enough."""
    rows = [("bride and groom", [BRIDE, GROOM, MOTHER, GROOM_FATHER])
            for _ in range(20)]

    _, outcome = resolve(gallery(rows))

    assert not outcome.resolved()
    assert outcome.inconclusive


# -- the two old bugs ------------------------------------------------------


def test_a_real_parent_is_not_rejected_for_being_too_old():
    """The old rule's age window was `couple_age + 15 +/- 10`, so it accepted
    +3..+27 and rejected +28 up -- the actual parent-child gap. A 58-year-old
    mother of a 30-year-old bride was rejected outright."""
    _, outcome = resolve(a_normal_wedding())

    assert MOTHER in outcome.bride_parents, "58 to the bride's 30 is a +28 gap"
    assert FATHER in outcome.bride_parents, "60 to the bride's 30 is a +30 gap"


def test_age_is_read_as_a_rank_not_as_an_offset():
    """Estimators regress toward the mean, so the same wedding read young or
    read old must give the same answer. Only the ordering is trusted."""
    compressed = {i: (30.0 + (age - 30.0) * 0.6, gender)
                  for i, (age, gender) in AGES.items()}

    _, wide = resolve(a_normal_wedding())
    _, narrow = resolve(a_normal_wedding(), ages=compressed)

    assert set(wide.bride_parents) == set(narrow.bride_parents)
    assert set(wide.groom_parents) == set(narrow.groom_parents)


def test_a_young_pair_in_a_circle_is_not_promoted_to_parents():
    """The old circle test was `pair & set_of_all_ids_in_any_circle`, which any
    two guests pass. Here the two candidates share a circle *and* are close in
    age to the couple; only their age rank should keep them out."""
    rows = [("portrait", [BRIDE, BRIDESMAID]) for _ in range(10)]
    rows += [("portrait", [BRIDE, GUEST]) for _ in range(10)]

    _, outcome = resolve(gallery(rows), groups=[[BRIDESMAID, GUEST]])

    assert not outcome.resolved()


# -- the indicators --------------------------------------------------------


def test_a_side_is_needed_before_a_parent_can_be_named():
    """An old person photographed evenly with both partners may well be a
    parent, but we cannot say whose -- and the wrong side is a false mark."""
    rows = [("portrait", [BRIDE, MOTHER]) for _ in range(6)]
    rows += [("portrait", [GROOM, MOTHER]) for _ in range(6)]

    candidates, outcome = resolve(gallery(rows))

    mother = next(c for c in candidates if c.identity == MOTHER)
    assert mother.side_skew() is None
    assert MOTHER not in outcome.all_parents()


def test_the_parent_dance_is_a_strong_signal():
    """A two-person dance frame that is not the couple: on 53459898 this is
    what identified the groom's mother, and it is nearly unambiguous."""
    rows = [("portrait", [GROOM, GROOM_MOTHER]) for _ in range(5)]
    rows += [("dancing", [GROOM, GROOM_MOTHER]) for _ in range(2)]

    candidates, _ = resolve(gallery(rows))
    mother = next(c for c in candidates if c.identity == GROOM_MOTHER)

    assert mother.duo_dance_groom == 2
    assert mother.duo_dance_bride == 0
    assert parents.score(mother, "groom") > parents.score(mother, "bride")


def test_the_party_penalty_is_relative_to_this_gallery():
    """The groom's father on 53459898 has 10 `groom party` frames -- in a suit
    he is not separable from the groomsmen -- while the groomsmen have 45-53.
    An absolute threshold sank him; a relative one does not."""
    rows = [("suit", [GROOM, GROOM_FATHER]) for _ in range(4)]
    rows += [("portrait", [GROOM, GROOM_FATHER]) for _ in range(8)]
    rows += [("groom party", [GROOM, GROOM_FATHER]) for _ in range(10)]
    rows += [("groom party", [GROOM, GUEST]) for _ in range(50)]

    candidates, outcome = resolve(gallery(rows))
    father = next(c for c in candidates if c.identity == GROOM_FATHER)

    assert father.party == 10
    assert father.party_share < 0.3, "ten against fifty is not party membership"
    assert GROOM_FATHER in outcome.groom_parents


def test_the_officiant_is_not_a_parent():
    """Old, at the ceremony, on neither side, and confined to a narrow band of
    the day -- id 26 on 53459898, aged 61 with 28 ceremony frames inside 8%
    of the gallery."""
    rows = [("bride getting dressed", [BRIDE, MOTHER]) for _ in range(45)]
    rows += [("ceremony", [BRIDE, GUEST]) for _ in range(12)]
    rows += [("dancing", [BRIDE, GROOM]) for _ in range(45)]
    celebrant = {**AGES, GUEST: (61, 0)}

    candidates, _ = resolve(gallery(rows), ages=celebrant)
    officiant = next(c for c in candidates if c.identity == GUEST)

    assert officiant.span[1] - officiant.span[0] <= CONFIGS["parents"]["officiant_span"]
    assert parents.score(officiant, "bride") < CONFIGS["parents"]["min_score"]


def test_a_small_circle_with_another_old_candidate_corroborates():
    """How the two parents of one side support each other. A circle shared
    with a *young* candidate says nothing -- that is a couple of guests."""
    rows = [("portrait", [BRIDE, MOTHER]) for _ in range(8)]
    rows += [("portrait", [BRIDE, FATHER]) for _ in range(8)]

    with_family, _ = resolve(gallery(rows), groups=[[MOTHER, FATHER]])
    alone, _ = resolve(gallery(rows), groups=[[MOTHER, BRIDESMAID]])

    mother_in = next(c for c in with_family if c.identity == MOTHER)
    mother_out = next(c for c in alone if c.identity == MOTHER)

    assert mother_in.circle_partners == (FATHER,)
    assert mother_out.circle_partners == ()
    assert parents.score(mother_in, "bride") > parents.score(mother_out, "bride")


# -- the query term --------------------------------------------------------


def test_the_query_term_is_shrunk_toward_zero_on_few_photos():
    """Unshrunk, the delta ranked a candidate with three attributable photos
    above the known bride's mother with thirty-one."""
    prior = CONFIGS["parents"]["query_prior"]
    assert parents._shrink(0.20, 3, prior) < parents._shrink(0.10, 31, prior)


def test_the_query_term_cannot_name_a_parent_on_its_own():
    """It is a second opinion on age, not a detector: a candidate with a
    perfect query score and no structural evidence stays unnamed."""
    rows = [("portrait", [BRIDE, GUEST]) for _ in range(20)]
    original = parents.concept_scores
    parents.concept_scores = lambda photos, concept: np.ones(len(photos))
    try:
        _, outcome = resolve(gallery(rows), quiet_queries=False)
    finally:
        parents.concept_scores = original

    assert GUEST not in outcome.all_parents()


def test_the_concept_bins_exist_in_both_clip_spaces():
    """A gallery's embeddings can be in either space, so a concept missing
    from one of them silently drops the term for half of production."""
    for concept in (parents.PARENT_CONCEPT, parents.PARTY_CONCEPT,
                    parents.DANCE_CONCEPT, parents.CELEBRANT_CONCEPT):
        for version in ("v1", "v2"):
            path = os.path.join("files", "pre_queries", version, f"{concept}.bin")
            assert os.path.exists(path), f"{path} is missing"


def test_a_missing_bin_does_not_fail_the_gallery():
    original = parents.concept_scores

    def explode(photos, concept):
        raise FileNotFoundError(concept)

    parents.concept_scores = explode
    try:
        candidates, outcome = resolve(a_normal_wedding(), quiet_queries=False)
    finally:
        parents.concept_scores = original

    assert candidates, "the structural indicators stand on their own"
    assert all(c.parent_query == 0.0 for c in candidates)


# -- labelling -------------------------------------------------------------


def test_the_portraits_are_labelled_from_the_named_identities():
    frame = a_normal_wedding()
    _, outcome = resolve(frame)

    labelled, count = parents.label(frame, outcome, BRIDE, GROOM)

    assert count > 0
    categories = set(labelled[Col.PARENT_CATEGORY].dropna())
    assert parents.BRIDE_PARENTS in categories
    assert parents.GROOM_PARENTS in categories
    assert parents.BOTH_PARENTS in categories
    relabelled = labelled[Col.CLUSTER_CONTEXT] == parents.PARENTS_PORTRAIT
    assert int(relabelled.sum()) == count


def test_one_parent_is_enough():
    """The old rule needed *exactly* two others, so a widowed, divorced or
    separately-photographed parent was never labelled at all."""
    frame = gallery([("portrait", [BRIDE, MOTHER])])
    outcome = parents.Resolution(bride_parents=(MOTHER,))

    _, count = parents.label(frame, outcome, BRIDE, GROOM)

    assert count == 1


def test_two_parents_of_the_same_gender_are_labelled():
    """The old rule rejected them outright on `gender_1 == gender_2`."""
    frame = gallery([("portrait", [BRIDE, MOTHER, GROOM_MOTHER])])
    outcome = parents.Resolution(bride_parents=(MOTHER, GROOM_MOTHER))

    _, count = parents.label(frame, outcome, BRIDE, GROOM)

    assert count == 1


def test_a_portrait_without_a_partner_is_not_a_parents_portrait():
    frame = gallery([("portrait", [MOTHER, FATHER])])
    outcome = parents.Resolution(bride_parents=(MOTHER, FATHER))

    _, count = parents.label(frame, outcome, BRIDE, GROOM)

    assert count == 0


def test_the_column_exists_even_when_nothing_is_resolved():
    """On three of eight validation galleries the old rule returned early and
    never created `parent_category` at all."""
    frame = gallery([("bride and groom", [BRIDE, GROOM])])

    labelled, count = parents.label(frame, parents.Resolution(), BRIDE, GROOM)

    assert count == 0
    assert Col.PARENT_CATEGORY in labelled.columns


# -- the guards the old rule lacked ----------------------------------------


def test_a_gallery_with_no_couple_is_not_an_error():
    """`main_persons[1]` raised IndexError on a gallery with one identity."""
    frame = gallery([("portrait", [BRIDE, MOTHER])])

    assert parents.measure(frame, details(AGES), None, BRIDE, None) == []
    assert parents.measure(frame, details(AGES), None, None, None) == []


def test_missing_person_details_is_not_an_error():
    """`persons_details_df.set_index` raised AttributeError on None."""
    frame = a_normal_wedding()

    assert parents.measure(frame, None, None, BRIDE, GROOM) == []
    assert parents.measure(frame, pd.DataFrame(), None, BRIDE, GROOM) == []


def test_a_candidate_with_no_age_row_is_not_an_error():
    """`abs(None - float)` raised TypeError, masked only by `or`
    short-circuiting when the first comparison happened to pass."""
    partial = {i: v for i, v in AGES.items() if i != FATHER}

    candidates, outcome = resolve(a_normal_wedding(), ages=partial)

    father = next(c for c in candidates if c.identity == FATHER)
    assert father.age is None and father.age_rank == 0.0
    assert FATHER not in outcome.all_parents(), "unknown age cannot clear the rank floor"


# -- the substage ----------------------------------------------------------


def test_it_declares_what_it_reads_and_writes():
    """Empty `requires`/`provides` meant no precondition check and no static
    ordering validation -- `Pipeline.unsatisfied` could not see this stage."""
    substage = get("enrich.parents")()

    assert Col.PARENT_CATEGORY in {t.split(":", 1)[1] for t in substage.provides}
    assert substage.requires, "it reads the couple, the classes and the timeline"
    assert substage.optional, "a gallery it cannot read must not fail the request"


def test_it_runs_only_on_weddings():
    """It had no `applies_to`, so it ran on every gallery type."""
    substage = get("enrich.parents")()
    context = AlbumContext(logger=quiet(), photos=a_normal_wedding(),
                           facts=GalleryFacts(is_wedding=False))

    assert not substage.applies_to(context)


def test_it_runs_after_the_couple_is_resolved():
    order = list(ENRICH)
    assert order.index("enrich.identities") < order.index("enrich.parents")
    assert order.index("enrich.temporal") < order.index("enrich.parents")


def test_the_named_parents_reach_the_gallery_facts():
    original = parents.concept_scores
    parents.concept_scores = lambda photos, concept: np.zeros(len(photos))
    try:
        context = AlbumContext(
            logger=quiet(), photos=a_normal_wedding(),
            facts=GalleryFacts(is_wedding=True, bride_id=BRIDE, groom_id=GROOM),
            person_details=details(AGES), social_circles=circles([]))
        context = get("enrich.parents")()(context)
    finally:
        parents.concept_scores = original

    assert not context.failed
    assert MOTHER in context.facts.bride_parents
    assert GROOM_MOTHER in context.facts.groom_parents


def test_the_old_rule_is_still_reachable():
    """The equivalence tests need a way back to the pre-refactor behaviour."""
    original = CONFIGS["parents"]["by_identity"]
    CONFIGS["parents"] = {**CONFIGS["parents"], "by_identity": False}
    try:
        context = AlbumContext(
            logger=quiet(), photos=a_normal_wedding(),
            facts=GalleryFacts(is_wedding=True, bride_id=BRIDE, groom_id=GROOM),
            person_details=details(AGES), social_circles=circles([[MOTHER, FATHER]]))
        context = get("enrich.parents")()(context)
    finally:
        CONFIGS["parents"] = {**CONFIGS["parents"], "by_identity": original}

    assert not context.failed
    assert context.facts.bride_parents == (), "the old rule names nobody"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
