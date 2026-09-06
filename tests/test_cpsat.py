"""Tests for the CP-SAT picker's hard rules.

The model is off by default and falls back to the loop on any exception, which
is good for safety and terrible for noticing that a constraint is wrong: a
mis-modelled rule looks exactly like a working one from outside. These pin the
two that can silently ruin an album.

`_add_distinct_shots` is the dangerous one. It reproduces the loop's
`_take_all_distinct`, and it is safe only because of its condition -- applied to
every class it would cap `dancing` at a single photo, since every frame there
holds the same couple and carries the same subquery.

    python -m pytest tests/test_cpsat.py -v
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.contracts import AlbumContext, Col, GalleryFacts  # noqa: E402
from src.pipeline.select.contracts import SelectionInputs, SelectionPlan  # noqa: E402
from src.pipeline.select.cpsat import CpSatPicker  # noqa: E402
from utils.configs import CONFIGS  # noqa: E402

BRIDE, GROOM = 1, 2
CLASS = {'dancing': 8, 'entertainment': 10, 'kiss': 19, 'portrait': 21}


def quiet():
    logger = logging.getLogger('cpsat-test')
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    return logger


def gallery(rows):
    """``rows`` of ``(context, people, subquery)``, in time order.

    Embeddings are drawn from a fixed seed rather than laid out on a line.
    Consecutive `linspace` vectors all point in nearly the same direction, so
    `_add_exclusions` -- the near-duplicate rule at cosine 0.97 -- capped a
    twelve-photo class at two before any rule under test could apply. Random
    8-d vectors sit near cosine 0, which leaves the field clear.
    """
    rng = np.random.default_rng(11)
    return pd.DataFrame([
        {Col.IMAGE_ID: 1000 + i,
         Col.CLUSTER_CONTEXT: context,
         Col.IMAGE_CLASS: CLASS[context],
         Col.PERSONS_IDS: list(people),
         Col.IMAGE_SUBQUERY_CONTENT: subquery,
         Col.GENERAL_TIME: float(i),
         Col.IMAGE_ORDER: float(i),
         Col.IMAGE_COLOR: 1,
         Col.MODEL_VERSION: 2,
         Col.EMBEDDING: rng.normal(size=16).astype(np.float32),
         Col.BRIDE_ID: BRIDE, Col.GROOM_ID: GROOM}
        for i, (context, people, subquery) in enumerate(rows)
    ])


def pick(photos, images, committed=None, **overrides):
    """Run the solver alone, with a hand-made plan."""
    context = AlbumContext(
        logger=quiet(), photos=photos,
        facts=GalleryFacts(is_wedding=True, model_version=2),
        selection_inputs=SelectionInputs(),
        selection_plan=SelectionPlan(images=dict(images),
                                     committed=dict(committed or {})),
    )
    original = CONFIGS['pick_cpsat']
    CONFIGS['pick_cpsat'] = {**original, 'enabled': True, **overrides}
    try:
        result = CpSatPicker(context).run()
    finally:
        CONFIGS['pick_cpsat'] = original
    assert result is not None, "the model should have produced a selection"
    chosen, _per_category = result
    return set(chosen)


def in_class(photos, chosen, category):
    ids = set(photos.loc[photos[Col.CLUSTER_CONTEXT] == category, Col.IMAGE_ID])
    return chosen & ids


# -- the distinct-shot rule -------------------------------------------------


def test_the_same_people_doing_the_same_thing_counts_once():
    """`entertainment` on 53459898: three frames, two of them the same band
    shot with the same person in it. The loop keeps two, and no coverage
    weighting reproduced that -- in a three-photo pool every photo is its own
    bucket in every dimension, so coverage rewards taking all three."""
    photos = gallery([
        ('entertainment', [18], 'live band or musician performing'),
        ('entertainment', [18], 'live band or musician performing'),
        ('entertainment', [], 'live band or musician performing'),
    ])

    chosen = pick(photos, {'entertainment': 3})

    assert len(in_class(photos, chosen, 'entertainment')) == 2, (
        "two distinct keys, so two photos -- not the whole pool")


def test_it_does_not_fire_where_there_is_slack():
    """The condition is the whole safety of the rule. Every frame in `dancing`
    holds the same couple and carries the same subquery, so applied without the
    supply <= demand test it would cap the class at one photo."""
    photos = gallery([
        ('dancing', [BRIDE, GROOM], 'bride and groom dancing') for _ in range(12)
    ])

    chosen = pick(photos, {'dancing': 4})

    assert len(in_class(photos, chosen, 'dancing')) == 4, (
        "a class with slack is ranked, not deduplicated")


def test_a_class_of_all_distinct_shots_keeps_them_all():
    photos = gallery([
        ('portrait', [10], 'a formal portrait'),
        ('portrait', [11], 'a formal portrait'),
        ('portrait', [12], 'a formal portrait'),
    ])

    chosen = pick(photos, {'portrait': 3})

    assert len(in_class(photos, chosen, 'portrait')) == 3


def test_committed_duplicates_do_not_make_the_model_infeasible():
    """Two committed photos sharing a key cannot both be given up, so a
    constraint over them has no solution. The loop never sees this because it
    deduplicates a frame `select.preselect` has already emptied of them."""
    photos = gallery([
        ('kiss', [BRIDE], 'bride and groom kissing'),
        ('kiss', [BRIDE], 'bride and groom kissing'),
        ('kiss', [], 'bride and groom kissing'),
    ])
    committed = {1000: 'user:test', 1001: 'user:test'}

    chosen = pick(photos, {'kiss': 1}, committed=committed)

    assert {1000, 1001} <= chosen, "committed photos are fixed, not re-decided"


def test_the_rule_can_be_switched_off():
    photos = gallery([
        ('entertainment', [18], 'live band'),
        ('entertainment', [18], 'live band'),
        ('entertainment', [], 'live band'),
    ])

    off = pick(photos, {'entertainment': 3},
               distinct_shots={'enabled': False})

    assert len(in_class(photos, off, 'entertainment')) == 3


# -- the admission cost is waived where nothing is ranked -------------------


def test_a_class_with_no_slack_is_not_charged_for_its_pages():
    """The loop reaches `_take_all_distinct` down the supply <= demand branch
    and takes everything bar the repeats *without consulting a score*. Charging
    a page bar there dropped a distinct shot from `entertainment` and `kiss`
    alike -- two keys apiece, the loop keeping both, the model one."""
    photos = gallery([
        ('entertainment', [18], 'live band'),
        ('entertainment', [], 'a solo singer'),
    ])

    # A bar so high that nothing could clear it on quality.
    chosen = pick(photos, {'entertainment': 2},
                  quota_ceiling={'enabled': True, 'admission_quantile': 0.99,
                                 'admission_cost': 0})

    assert len(in_class(photos, chosen, 'entertainment')) == 2


def test_a_class_with_slack_still_pays_for_its_pages():
    """The waiver is only for the branch that does no ranking; everywhere else
    the bar is what stops the model filling every allowance to the brim."""
    photos = gallery([
        ('portrait', [10 + i], f'portrait number {i}') for i in range(12)
    ])

    charged = pick(photos, {'portrait': 8},
                   quota_ceiling={'enabled': True, 'admission_quantile': 0.9,
                                  'admission_cost': 0})
    free = pick(photos, {'portrait': 8},
                quota_ceiling={'enabled': True, 'admission_quantile': 0.0,
                               'admission_cost': 0})

    assert len(in_class(photos, charged, 'portrait')) < len(
        in_class(photos, free, 'portrait'))


# -- who a class is about ---------------------------------------------------


def test_the_bride_class_prefers_the_bride_alone():
    """`persons_ids == [bride_id]`, as `CoupleTimelineStrategy` filters. The
    better-ranked frame here is a group shot, and it should still lose."""
    photos = gallery([
        ('portrait', [BRIDE, 30, 31], 'the bride with friends'),   # rank 0, best
        ('portrait', [BRIDE], 'the bride alone'),
    ])
    photos[Col.CLUSTER_CONTEXT] = 'bride'
    photos[Col.IMAGE_CLASS] = 1

    chosen = pick(photos, {'bride': 1})

    assert 1001 in chosen and 1000 not in chosen


def test_the_couple_class_wants_the_two_of_them_and_nobody_else():
    photos = gallery([
        ('portrait', [BRIDE, GROOM, 30], 'the couple with a guest'),
        ('portrait', [BRIDE, GROOM], 'the couple alone'),
    ])
    photos[Col.CLUSTER_CONTEXT] = 'bride and groom'
    photos[Col.IMAGE_CLASS] = 2

    chosen = pick(photos, {'bride and groom': 1})

    assert 1001 in chosen and 1000 not in chosen


def test_the_getting_ready_class_wants_the_bride_in_it():
    """The failure that prompted the rule: a hair-and-makeup spread of someone
    unrelated to the couple, chosen on rank alone."""
    photos = gallery([
        ('portrait', [30], 'someone getting their hair done'),
        ('portrait', [BRIDE], 'the bride getting her hair done'),
    ])
    photos[Col.CLUSTER_CONTEXT] = 'getting hair-makeup'
    photos[Col.IMAGE_CLASS] = 14

    chosen = pick(photos, {'getting hair-makeup': 1})

    assert 1001 in chosen


def test_an_undetected_subject_still_fills_the_class():
    """The reason this is a score and not a filter. A hard rule empties the
    class on a gallery where face detection missed the couple, which is why the
    loop needs `_recover_over_filtering` behind its filters. An empty
    `persons_ids` is a detection that did not happen, so it stays neutral and
    rank decides."""
    photos = gallery([
        ('portrait', [], 'someone, unidentified'),
        ('portrait', [], 'someone else, unidentified'),
    ])
    photos[Col.CLUSTER_CONTEXT] = 'bride'
    photos[Col.IMAGE_CLASS] = 1

    chosen = pick(photos, {'bride': 2})

    assert len(chosen) == 2, "nobody was detected, so nothing contradicts"


def test_an_exclusive_class_of_wrong_people_is_left_empty():
    """`bride` is *only* about the bride, so a frame naming someone else is the
    wrong photo rather than an unconfirmed one. Better an unfilled page than a
    stranger on the bride's spread -- the same judgement as `enrich.parents`."""
    photos = gallery([
        ('portrait', [30], 'a guest'),
        ('portrait', [31], 'another guest'),
    ])
    photos[Col.CLUSTER_CONTEXT] = 'bride'
    photos[Col.IMAGE_CLASS] = 1

    assert not pick(photos, {'bride': 2})


def test_a_group_class_is_not_exclusive():
    """Parents and flower girls walk the aisle, and the party classes are about
    a group, so another face there is not a wrong photo. Penalising it emptied
    `walking the aisle` outright on 53459898 -- where the couple is detected in
    none of its frames -- losing a scripted moment to a detection gap.
    """
    photos = gallery([
        ('portrait', [30], 'the flower girl walking in'),
        ('portrait', [31], 'the parents walking in'),
    ])
    photos[Col.CLUSTER_CONTEXT] = 'walking the aisle'
    photos[Col.IMAGE_CLASS] = 28

    chosen = pick(photos, {'walking the aisle': 2})

    assert len(chosen) == 2, "others belong in the processional"


def test_a_wrong_identity_loses_to_an_unknown_one():
    """The three-valued part. Both fail the rule, and they are not equal: the
    better-ranked frame here names the wrong person, and should still lose to
    the one that names nobody."""
    photos = gallery([
        ('portrait', [30], 'definitely a guest'),     # rank 0, best
        ('portrait', [], 'unidentified'),
    ])
    photos[Col.CLUSTER_CONTEXT] = 'bride'
    photos[Col.IMAGE_CLASS] = 1

    chosen = pick(photos, {'bride': 1})

    assert 1001 in chosen and 1000 not in chosen


def test_the_couple_is_read_from_the_photo_table():
    """`resolve_bride_groom` stamps both on every row, and a SELECT-only driver
    never puts them on `facts`. Reading `facts` first made the whole rule a
    silent no-op -- the measured numbers did not move at all."""
    photos = gallery([
        ('portrait', [30], 'a guest'),
        ('portrait', [BRIDE], 'the bride alone'),
    ])
    photos[Col.CLUSTER_CONTEXT] = 'bride'
    photos[Col.IMAGE_CLASS] = 1
    # `pick` builds GalleryFacts without a couple, exactly as the harness does.

    chosen = pick(photos, {'bride': 1})

    assert 1001 in chosen


def test_the_preference_can_be_switched_off():
    photos = gallery([
        ('portrait', [30], 'a guest'),
        ('portrait', [BRIDE], 'the bride alone'),
    ])
    photos[Col.CLUSTER_CONTEXT] = 'bride'
    photos[Col.IMAGE_CLASS] = 1

    off = pick(photos, {'bride': 1},
               identity_preference={'enabled': False, 'weight': 1200,
                                    'per_class': {}})

    assert 1000 in off, "rank alone, so the best-ranked frame wins"


# -- the quota is a ceiling -------------------------------------------------


def test_the_allowance_is_never_exceeded():
    photos = gallery([
        ('portrait', [10 + i], f'portrait number {i}') for i in range(20)
    ])

    chosen = pick(photos, {'portrait': 5})

    assert len(in_class(photos, chosen, 'portrait')) <= 5


def test_a_class_with_no_allowance_takes_nothing():
    """Another class carries the allowance, because a model with nothing to
    decide anywhere returns None on purpose and hands back to the loop."""
    photos = gallery(
        [('portrait', [10], 'a portrait') for _ in range(4)]
        + [('dancing', [BRIDE, GROOM], f'dancing {i}') for i in range(6)])

    chosen = pick(photos, {'portrait': 0, 'dancing': 2})

    assert not in_class(photos, chosen, 'portrait')
    assert in_class(photos, chosen, 'dancing'), "the solve still happened"


def test_a_model_with_nothing_to_decide_hands_back():
    """Every class settled or unbudgeted: there is no decision to make, so the
    picker declines rather than returning an empty album."""
    photos = gallery([('portrait', [10], 'a portrait') for _ in range(4)])
    context = AlbumContext(
        logger=quiet(), photos=photos,
        facts=GalleryFacts(is_wedding=True, model_version=2),
        selection_inputs=SelectionInputs(),
        selection_plan=SelectionPlan(images={'portrait': 0}))
    original = CONFIGS['pick_cpsat']
    CONFIGS['pick_cpsat'] = {**original, 'enabled': True}
    try:
        assert CpSatPicker(context).run() is None
    finally:
        CONFIGS['pick_cpsat'] = original


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
