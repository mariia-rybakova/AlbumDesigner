"""Tests for the four defects found reviewing gallery 49995684's album.

Each one is pinned to what the album actually did, because every one of them
looked like a plausible ranking outcome from the outside and only the data said
otherwise:

* a faceless frame outranking every photo of the couple, on the heaviest
  weight there is (`person_score`);
* `may kiss bride` and `cake cutting` going to the one candidate without the
  couple in it;
* a six-person "guests watching ceremony" frame tagged `groom walking the
  aisle`;
* the bride's mother dropped for an ambiguous side, which then invalidated
  every family portrait she stood in.

    python -m pytest tests/test_subject_preference.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.contracts import Col  # noqa: E402
from src.pipeline import subject  # noqa: E402
from src.selection.ai_wedding_selection import calculate_scores, get_scores  # noqa: E402
from utils.configs import CONFIGS  # noqa: E402

BRIDE, GROOM = 1, 7
#: The identities the 49995684 request named -- deliberately not the couple.
NAMED = [58, 100, 71]


def rows(specs, category):
    """A category's frame: ``(image_id, persons_ids, n_faces)`` each."""
    frame = pd.DataFrame([
        {Col.IMAGE_ID: image_id, Col.PERSONS_IDS: list(people),
         Col.N_FACES: faces, Col.CLUSTER_CONTEXT: category,
         Col.IMAGE_CLASS: 12, Col.IMAGE_ORDER: index,
         Col.RANKING: 0.5, Col.IMAGE_SUBQUERY_CONTENT: 'x'}
        for index, (image_id, people, faces) in enumerate(specs)
    ])
    return frame


# -- person_score must not reward the absence of people ---------------------


def test_a_faceless_photo_does_not_outrank_a_photo_of_people():
    """The bug in one line. `people_ids` names identities the photo does not
    have; a photo with nobody in it must not therefore be the best answer."""
    people = pd.Series({Col.PERSONS_IDS: [BRIDE, GROOM]})
    nobody = pd.Series({Col.PERSONS_IDS: []})

    _, _, with_people, _ = calculate_scores(people, pd.DataFrame(), NAMED, [])
    _, _, without, _ = calculate_scores(nobody, pd.DataFrame(), NAMED, [])

    assert without <= with_people, (
        "a frame with no identified people scored above one holding the couple")


def test_the_named_person_still_wins():
    """The fix must not flatten the signal it is there to carry."""
    named = pd.Series({Col.PERSONS_IDS: [58]})
    stranger = pd.Series({Col.PERSONS_IDS: [30]})

    _, _, hit, _ = calculate_scores(named, pd.DataFrame(), NAMED, [])
    _, _, miss, _ = calculate_scores(stranger, pd.DataFrame(), NAMED, [])

    assert hit > miss


def test_the_faceless_frame_no_longer_normalises_to_the_top():
    """End to end through `get_scores`, which is where the damage happened:
    `normalize()` turned a 1e-7 fallback into a full 1.0 because it is 100x the
    degenerate-range guard. Measured on the real `may kiss bride` group, where
    the faceless frame scored 0.539 against 0.317."""
    frame = rows([(1, [], 0), (2, [BRIDE, GROOM], 2), (3, [BRIDE], 1)],
                 'may kiss bride')

    _ranked, scored = get_scores(frame, pd.DataFrame(), NAMED, [], None, {})

    faceless = scored.loc[scored[Col.IMAGE_ID] == 1, 'total_score'].iloc[0]
    couple = scored.loc[scored[Col.IMAGE_ID] == 2, 'total_score'].iloc[0]
    assert faceless <= couple


# -- the category's subject -------------------------------------------------


def test_the_cake_photo_without_the_couple_is_dropped():
    """12 of 13 cake photos on 49995684 hold the couple; the album took the
    thirteenth."""
    frame = rows([(1, [], 0), (2, [BRIDE, GROOM], 2), (3, [BRIDE, GROOM], 3)],
                 'cake cutting')

    kept = subject.prefer_subject(frame, 'cake cutting', BRIDE, GROOM)

    assert set(kept[Col.IMAGE_ID]) == {2, 3}


def test_a_category_is_never_emptied():
    """The whole reason this is a preference. On 53459898 the couple is
    detected in none of `walking the aisle`'s frames, and excluding them there
    lost a scripted moment of the wedding to a detection gap."""
    frame = rows([(1, [30], 1), (2, [31], 1)], 'cake cutting')

    kept = subject.prefer_subject(frame, 'cake cutting', BRIDE, GROOM)

    assert len(kept) == 2


def test_a_category_about_nobody_is_untouched():
    frame = rows([(1, [], 0), (2, [BRIDE], 1)], 'detail')

    kept = subject.prefer_subject(frame, 'detail', BRIDE, GROOM)

    assert len(kept) == 2


def test_the_processionals_want_their_own_half_of_the_couple():
    assert subject.required_roles('bride walking the aisle') == ('bride',)
    assert subject.required_roles('groom walking the aisle') == ('groom',)

    frame = rows([(1, [BRIDE], 1), (2, [GROOM], 1)], 'groom walking the aisle')
    kept = subject.prefer_subject(frame, 'groom walking the aisle', BRIDE, GROOM)

    assert set(kept[Col.IMAGE_ID]) == {2}


def test_an_unresolved_couple_changes_nothing():
    """Galleries where the identity model found no couple must behave exactly
    as they did before."""
    frame = rows([(1, [], 0), (2, [30], 1)], 'cake cutting')

    kept = subject.prefer_subject(frame, 'cake cutting', None, None)

    assert len(kept) == 2


def test_one_table_drives_both_pickers():
    """`select.cpsat` used to own this table privately, so the loop picker and
    the `yes` categories had no notion of a class's subject at all. A second
    table that could disagree with it is the thing to avoid."""
    from src.pipeline.select import cpsat

    assert cpsat.IDENTITY_RULES is subject.IDENTITY_RULES
    for category in ('cake cutting', 'may kiss bride', 'kiss', 'first dance',
                     'send off', 'couple'):
        assert category in subject.IDENTITY_RULES, category


def test_the_couple_moments_are_not_exclusive():
    """Guests crowd the cake and line the send-off, so a frame naming someone
    else there is not the wrong photo -- only the less preferred one. Marking
    them exclusive would have CP-SAT drop them outright."""
    for category in ('cake cutting', 'send off', 'kiss', 'may kiss bride',
                     'first dance', 'couple'):
        _rule, _subjects, exclusive = subject.IDENTITY_RULES[category]
        assert exclusive is False, category


# -- covering a named identity ----------------------------------------------


def test_a_named_identity_is_covered_from_a_real_class_first():
    """`other` and `None` are budgeted 0% precisely because they carry nothing
    worth a spread. Two of 49995684's `other` photos are in the album only to
    cover identities 58 and 71."""
    frame = pd.DataFrame([
        {Col.IMAGE_ID: 1, Col.CLUSTER_CONTEXT: 'other'},
        {Col.IMAGE_ID: 2, Col.CLUSTER_CONTEXT: 'dancing'},
        {Col.IMAGE_ID: 3, Col.CLUSTER_CONTEXT: 'bride and groom'},
    ])

    tiers = subject.identity_tiers(frame)

    assert list(tiers[0][Col.IMAGE_ID]) == [2], "a real class the couple does not own"
    assert list(tiers[1][Col.IMAGE_ID]) == [3], "a real class about the couple"
    assert list(tiers[2][Col.IMAGE_ID]) == [1], "other/None, the last resort"


def test_a_missing_class_counts_as_unknown():
    """`cluster_context` carries both the string 'None' and genuine nulls."""
    assert subject.is_unknown_category('other')
    assert subject.is_unknown_category('None')
    assert subject.is_unknown_category(None)
    assert subject.is_unknown_category(np.nan)
    assert not subject.is_unknown_category('dancing')


def test_the_preference_can_be_switched_off():
    frame = rows([(1, [], 0), (2, [BRIDE, GROOM], 2)], 'cake cutting')
    original = CONFIGS['subject']
    CONFIGS['subject'] = {**original, 'enabled': False}
    try:
        kept = subject.prefer_subject(frame, 'cake cutting', BRIDE, GROOM)
    finally:
        CONFIGS['subject'] = original

    assert len(kept) == 2


# -- a class meaning "the two of them" needs both of them -------------------


def _couple_frame(specs):
    return pd.DataFrame([
        {Col.IMAGE_ID: image_id, Col.CLUSTER_CONTEXT: category,
         Col.PERSONS_IDS: list(people)}
        for image_id, category, people in specs
    ])


def test_one_of_them_with_a_stranger_is_not_a_couple_photo():
    """Page 17 on 49995684: a `bride and groom` spread carrying the bride and a
    groomsman. The other half is not merely unrecognised -- two people were
    identified and neither is the groom."""
    assert subject.one_half_with_an_outsider([BRIDE, 29], BRIDE, GROOM)
    assert subject.one_half_with_an_outsider([GROOM, 19], BRIDE, GROOM)


def test_one_of_them_alone_is_left_alone():
    """The distinction the whole rule turns on. `[bride]` is almost always the
    couple with one face unrecognised -- 29 of 120 on 49995684 -- and demoting
    those would cost a quarter of the class to a detection gap."""
    assert not subject.one_half_with_an_outsider([BRIDE], BRIDE, GROOM)
    assert not subject.one_half_with_an_outsider([], BRIDE, GROOM)
    assert not subject.one_half_with_an_outsider([BRIDE, GROOM], BRIDE, GROOM)
    assert not subject.one_half_with_an_outsider([BRIDE, GROOM, 29], BRIDE, GROOM)


def test_the_misfiled_photo_is_re_filed_not_dropped():
    """It has to be a relabel: the offending photo on 49995684 is one of the
    request's own `aiMetadata.photoIds`, and `select.preselect` commits a hand
    pick whatever it scores, so no preference can displace it. The user picked a
    photo, not a category."""
    frame = _couple_frame([
        (1, 'bride and groom', [BRIDE, 29]),
        (2, 'bride and groom', [BRIDE, GROOM]),
        (3, 'bride and groom', [BRIDE]),
    ])

    out, moved = subject.refile_misfiled_couple(frame, BRIDE, GROOM)

    assert moved == 1
    assert len(out) == 3, "nothing is dropped"
    by_id = out.set_index(Col.IMAGE_ID)[Col.CLUSTER_CONTEXT]
    assert by_id[1] == 'portrait'
    assert by_id[2] == 'bride and groom'
    assert by_id[3] == 'bride and groom'


def test_re_filing_needs_both_halves_named():
    frame = _couple_frame([(1, 'bride and groom', [BRIDE, 29])])

    _out, moved = subject.refile_misfiled_couple(frame, BRIDE, None)

    assert moved == 0


def test_only_the_couple_alone_classes_are_re_filed():
    """`walking the aisle` and the party classes legitimately hold other
    people, so the rule must not reach them."""
    frame = _couple_frame([
        (1, 'walking the aisle', [BRIDE, 29]),
        (2, 'bride party', [BRIDE, 29]),
        (3, 'cake cutting', [BRIDE, 29]),
    ])

    out, moved = subject.refile_misfiled_couple(frame, BRIDE, GROOM)

    assert moved == 0
    assert list(out[Col.CLUSTER_CONTEXT]) == ['walking the aisle', 'bride party',
                                              'cake cutting']


# -- whole scenes filed as the couple ---------------------------------------
#
# 49996919: the album opened on the bride being kissed by her father, in a
# first look filed `bride and groom`. The father is never recognised, so no
# frame-level test can see it; across the scene the bride is named four times
# and the groom not once.

FATHER_SCENE = [
    # (image_id, general_time, persons_ids) -- the real run, as read.
    (11548148002, 12591, []),
    (11548148006, 12593, [BRIDE]),
    (11548148022, 12598, [BRIDE]),
    (11548148023, 12605, [BRIDE]),
]


def scene_frame(scene, category="bride and groom", extra=()):
    """A gallery of one scene plus whatever context the test needs."""
    rows_ = [{Col.IMAGE_ID: i, Col.GENERAL_TIME: t, Col.PERSONS_IDS: list(p),
              Col.CLUSTER_CONTEXT: category} for i, t, p in scene]
    rows_.extend({Col.IMAGE_ID: i, Col.GENERAL_TIME: t, Col.PERSONS_IDS: list(p),
                  Col.CLUSTER_CONTEXT: c} for i, t, p, c in extra)
    return pd.DataFrame(rows_)


def _couple_context(n=6, start=90000):
    """Enough frames naming the groom that his absence elsewhere means something."""
    return [(9000 + i, start + i * 10, [BRIDE, GROOM], "bride and groom")
            for i in range(n)]


def test_a_one_sided_scene_is_refiled():
    frame = scene_frame(FATHER_SCENE, extra=_couple_context())

    out, moved = subject.refile_lopsided_couple_scenes(frame, BRIDE, GROOM)

    assert moved == len(FATHER_SCENE)
    misfiled = out[out[Col.IMAGE_ID].isin(i for i, _, _ in FATHER_SCENE)]
    assert set(misfiled[Col.CLUSTER_CONTEXT]) == {
        CONFIGS['subject']['misfiled_couple_class']}


def test_the_couple_keep_their_own_scenes():
    """The frames naming both are the control: whatever the rule does to the
    first look, it must not touch these."""
    frame = scene_frame(FATHER_SCENE, extra=_couple_context())

    out, _ = subject.refile_lopsided_couple_scenes(frame, BRIDE, GROOM)

    context_ids = {i for i, _t, _p, _c in _couple_context()}
    kept = out[out[Col.IMAGE_ID].isin(context_ids)]
    assert set(kept[Col.CLUSTER_CONTEXT]) == {"bride and groom"}


def test_a_single_frame_naming_one_of_them_is_not_a_scene():
    """The ordinary case the rule must stay cheap on: the couple with one face
    unrecognised. 29 of 120 couple frames on 49995684 look like this, and one
    detection is not a pattern."""
    lone = [(11548148023, 12605, [BRIDE])]
    frame = scene_frame(lone, extra=_couple_context())

    _out, moved = subject.refile_lopsided_couple_scenes(frame, BRIDE, GROOM)

    assert moved == 0


def test_a_scene_the_groom_appears_in_is_left_alone():
    """One frame naming him is enough -- it is his scene, shot from behind."""
    scene = [(1, 100, [BRIDE]), (2, 110, [BRIDE]), (3, 120, [GROOM])]
    frame = scene_frame(scene, extra=_couple_context())

    _out, moved = subject.refile_lopsided_couple_scenes(frame, BRIDE, GROOM)

    assert moved == 0


def test_a_gallery_that_never_recognises_the_groom_is_left_alone():
    """Absence is only evidence if the model can find him at all. Without this
    guard every couple scene on such a gallery looks lopsided and the rule
    empties the class."""
    frame = scene_frame(FATHER_SCENE)

    _out, moved = subject.refile_lopsided_couple_scenes(frame, BRIDE, GROOM)

    assert moved == 0


def test_a_far_apart_run_is_two_scenes():
    """The gap is what makes a scene. Two frames an hour apart, each naming
    only the bride, are not one lopsided moment."""
    scene = [(1, 100, [BRIDE]), (2, 100 + 3600, [BRIDE])]
    frame = scene_frame(scene, extra=_couple_context())

    _out, moved = subject.refile_lopsided_couple_scenes(frame, BRIDE, GROOM)

    assert moved == 0


def test_an_unresolved_couple_is_not_an_error():
    frame = scene_frame(FATHER_SCENE, extra=_couple_context())

    assert subject.lopsided_couple_scenes(frame, BRIDE, None) is None
    assert subject.lopsided_couple_scenes(frame, None, None) is None
