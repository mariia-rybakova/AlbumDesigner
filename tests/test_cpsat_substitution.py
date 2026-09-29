"""The retry's slot substitution: a repeat's slot may move to another class.

With the walls down the retry fills every class to its own allowance, so a
class with nothing left but copies of one shot takes the copies. Substitution
keeps the album full but lets that slot go to a class with a distinct photo
for it, preferring the class the profile weights most, scaled by how much of
the day it covers.

    python -m pytest tests/test_cpsat_substitution.py -v
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.contracts import AlbumContext, Col, GalleryFacts  # noqa: E402
from src.pipeline.select import cpsat as cpsat_module  # noqa: E402
from src.pipeline.select.contracts import SelectionInputs, SelectionPlan  # noqa: E402
from src.pipeline.select.cpsat import CpSatPicker, Relaxation  # noqa: E402
from utils.configs import CONFIGS  # noqa: E402

RNG = np.random.default_rng(5)
SHOT = RNG.normal(size=16)


def _copy_of_shot():
    """A frame at cosine ~0.99 to every other copy: over the 0.97 wall."""
    return (SHOT + RNG.normal(scale=0.08, size=16)).astype(np.float32)


def _distinct():
    return RNG.normal(size=16).astype(np.float32)


def photos(spec):
    """``spec`` is ``[(category, how many, 'burst' | 'distinct', minutes apart)]``.

    Each class is laid out in its own stretch of the day, one photo per
    ``minutes apart``, so a class's span is set by the test.
    """
    rows, image_id, start = [], 1000, 0.0
    for category, count, kind, step in spec:
        for i in range(count):
            rows.append({
                Col.IMAGE_ID: image_id,
                Col.CLUSTER_CONTEXT: category,
                Col.IMAGE_CLASS: 8,
                Col.PERSONS_IDS: [],
                Col.IMAGE_SUBQUERY_CONTENT: f"{category} {i}",
                Col.GENERAL_TIME: start + i * step,
                Col.IMAGE_ORDER: float(i),
                Col.IMAGE_COLOR: 1,
                Col.MODEL_VERSION: 2,
                Col.EMBEDDING: _copy_of_shot() if kind == 'burst' else _distinct(),
            })
            image_id += 1
        start += count * step + 1000.0
    return pd.DataFrame(rows)


def retry(substitution=True):
    return {'relaxed_retry': {
        'enabled': True, 'shortfall_trigger': 1,
        'drop_distinct_shots': True, 'drop_exclusions': True,
        'drop_contradictions': True, 'allowance_is_target': True,
        'substitution': substitution,
    }}


def solve(frame, images, shares=None, yes=(), **overrides):
    logger = logging.getLogger('cpsat-substitution-test')
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    context = AlbumContext(
        logger=logger, photos=frame,
        facts=GalleryFacts(is_wedding=True, model_version=2),
        selection_inputs=SelectionInputs(),
        selection_plan=SelectionPlan(images=dict(images), shares=dict(shares or {}),
                                     yes_categories=tuple(yes)),
    )
    original = CONFIGS['pick_cpsat']
    CONFIGS['pick_cpsat'] = {**original, 'enabled': True, **overrides}
    try:
        picker = CpSatPicker(context)
        result = picker.run()
    finally:
        CONFIGS['pick_cpsat'] = original
    assert result is not None
    chosen, per_category = result
    counts = frame[frame[Col.IMAGE_ID].isin(chosen)][Col.CLUSTER_CONTEXT].value_counts()
    return counts.to_dict(), per_category


#: `dancing` needs six and has two distinct frames plus six copies of one shot.
DANCING = ('dancing', 2, 'distinct', 1.0), ('dancing', 6, 'burst', 1.0)


def test_without_substitution_the_retry_fills_the_class_with_copies():
    """The behaviour being replaced, pinned so the change is visibly it."""
    frame = photos([*DANCING, ('bride', 10, 'distinct', 5.0)])

    counts, _ = solve(frame, {'dancing': 6, 'bride': 2}, **retry(substitution=False))

    assert counts['dancing'] == 6
    assert counts['bride'] == 2


def test_a_repeat_slot_moves_to_a_class_with_a_distinct_photo():
    frame = photos([*DANCING, ('bride', 10, 'distinct', 5.0)])

    counts, per_category = solve(frame, {'dancing': 6, 'bride': 2}, **retry())

    assert counts['dancing'] < 6, "fewer copies of the one shot"
    assert counts['bride'] > 2, "the slots went to the bride"
    assert counts['dancing'] + counts['bride'] == 8, "the album is still full"
    assert per_category['bride']['moved_in'] == counts['bride'] - 2
    assert per_category['dancing']['moved_out'] == 6 - counts['dancing']


def test_the_album_grows_by_at_most_half_a_spread():
    """A dancing photo is a twenty-fourth of a spread and a bride photo a
    quarter, so each move lengthens the album; `max_growth_spreads` caps it."""
    frame = photos([*DANCING, ('bride', 10, 'distinct', 5.0)])

    counts, _ = solve(frame, {'dancing': 6, 'bride': 2}, **retry())

    growth = (counts['bride'] - 2) / 4 - (6 - counts['dancing']) / 24
    assert growth <= 0.5 + 1e-9


def test_a_single_moment_class_never_receives():
    frame = photos([*DANCING, ('cake cutting', 10, 'distinct', 5.0)])

    counts, _ = solve(frame, {'dancing': 6, 'cake cutting': 2}, **retry())

    assert counts['cake cutting'] == 2


def test_a_yes_class_never_receives():
    frame = photos([*DANCING, ('rings', 10, 'distinct', 5.0)])

    counts, _ = solve(frame, {'dancing': 6, 'rings': 1}, yes=('rings',), **retry())

    assert counts['rings'] == 1


def test_the_slot_goes_to_the_class_the_profile_weights_most():
    """`bride` and `groom` are alike in photos, span and spread size; only the
    profile tells them apart."""
    frame = photos([*DANCING, ('bride', 10, 'distinct', 5.0), ('groom', 10, 'distinct', 5.0)])
    images = {'dancing': 6, 'bride': 2, 'groom': 2}

    counts, _ = solve(frame, images, shares={'dancing': 0.1, 'bride': 0.5, 'groom': 0.1},
                      **retry())

    assert counts['bride'] > counts['groom'] == 2


def test_abundance_decides_between_equal_shares():
    """Equal shares: the class with far more photos across a longer stretch of
    the day is the one the wedding put its time into."""
    frame = photos([*DANCING, ('bride', 30, 'distinct', 5.0), ('groom', 6, 'distinct', 1.0)])
    images = {'dancing': 6, 'bride': 2, 'groom': 2}

    counts, _ = solve(frame, images, shares={'dancing': 0.2, 'bride': 0.4, 'groom': 0.4},
                      **retry())

    assert counts['bride'] > counts['groom'] == 2


def test_importance_is_share_times_one_plus_abundance():
    frame = photos([('bride', 20, 'distinct', 5.0), ('groom', 10, 'distinct', 5.0)])
    context = AlbumContext(
        logger=logging.getLogger('x'), photos=frame,
        facts=GalleryFacts(is_wedding=True, model_version=2),
        selection_inputs=SelectionInputs(),
        selection_plan=SelectionPlan(images={'bride': 2, 'groom': 2},
                                     shares={'bride': 0.5, 'groom': 0.5}),
    )
    picker = CpSatPicker(context)
    receivers = picker._receivers(cpsat_module.tl.ordered(frame))

    # bride: most photos and longest span -> abundance 1; groom: half of both.
    assert receivers['bride'].importance == 0.5 * 2.0
    assert abs(receivers['groom'].importance - 0.5 * 1.5) < 0.05
    assert receivers['bride'].preference == CONFIGS['pick_cpsat']['substitution'][
        'preference_weight']
    assert receivers['bride'].cap == 4, "one spread of `bride`"


def test_a_full_first_solve_is_untouched():
    """Substitution lives in the retry. When the first solve is full there is
    no retry, and the reserve candidates never enter a model."""
    frame = photos([('dancing', 8, 'distinct', 1.0), ('bride', 10, 'distinct', 5.0)])

    on, _ = solve(frame, {'dancing': 6, 'bride': 2}, **retry())
    off, _ = solve(frame, {'dancing': 6, 'bride': 2}, **retry(substitution=False))

    assert on == off == {'dancing': 6, 'bride': 2}


def test_the_wall_is_priced_when_it_is_down():
    """The ramp stops at the wall because the wall used to take over there; in
    a retry that drops the wall, a pair above it must not be the one pair
    charged nothing."""
    frame = photos([('dancing', 3, 'burst', 1.0)])
    added = []
    original = CpSatPicker._add_similarity_penalty

    def spy(self, model, rows, x, penalties):
        before = len(penalties)
        original(self, model, rows, x, penalties)
        added.append((bool(self.relaxation.exclusions), len(penalties) - before))

    CpSatPicker._add_similarity_penalty = spy
    try:
        solve(frame, {'dancing': 3}, **retry(substitution=False))
    finally:
        CpSatPicker._add_similarity_penalty = original

    assert (False, 0) in added, "strict: the wall forbids, nothing is charged"
    assert any(down and n > 0 for down, n in added), "relaxed: the pairs are charged"


def test_describe_names_the_substitution():
    assert 'fixed_class_quotas' in Relaxation(substitution=True).describe()
    assert bool(Relaxation(substitution=True))


def test_a_class_of_distinct_photos_gives_nothing():
    """The donor limit: `dancing` has six different shots for six slots, so no
    slot is repeat-bound and none moves -- however much better the bride's
    photos score. Before the limit a slot moved on rank alone."""
    frame = photos([('dancing', 6, 'distinct', 1.0), ('bride', 10, 'distinct', 5.0)])
    frame.loc[frame[Col.CLUSTER_CONTEXT] == 'dancing', Col.IMAGE_ORDER] = 50.0

    counts, per_category = solve(frame, {'dancing': 6, 'bride': 2},
                                 shares={'dancing': 0.1, 'bride': 0.9}, **retry())

    assert counts == {'dancing': 6, 'bride': 2}
    assert 'moved_out' not in per_category['dancing']


def test_a_class_gives_at_most_its_repeat_bound_slots():
    """Two distinct frames and one burst are three shots against six slots:
    three are repeat-bound and at most three can move."""
    frame = photos([*DANCING, ('bride', 40, 'distinct', 5.0)])

    counts, _ = solve(frame, {'dancing': 6, 'bride': 2},
                      **{**retry(), 'substitution': {
                          **CONFIGS['pick_cpsat']['substitution'], 'max_growth_spreads': 5.0}})

    assert counts['dancing'] >= 3


def test_the_groom_gives_to_his_party_first():
    """Affinity adds the groom's party as a receiver and prefers it a little;
    the couple, the default receiver by profile share, keep receiving too."""
    frame = photos([('groom', 2, 'distinct', 1.0), ('groom', 6, 'burst', 1.0),
                    ('bride and groom', 10, 'distinct', 5.0),
                    ('groom party', 10, 'distinct', 5.0)])
    images = {'groom': 6, 'bride and groom': 2, 'groom party': 2}
    shares = {'groom': 0.12, 'bride and groom': 0.12, 'groom party': 0.03}

    counts, per_category = solve(frame, images, shares=shares, **retry())

    assert counts['groom party'] > 2, "the groom's slots reach his party"
    assert counts['groom party'] - 2 >= counts['bride and groom'] - 2, "and it is preferred"


def test_without_affinity_the_couple_takes_the_groom_slots():
    frame = photos([('groom', 2, 'distinct', 1.0), ('groom', 6, 'burst', 1.0),
                    ('bride and groom', 10, 'distinct', 5.0),
                    ('groom party', 10, 'distinct', 5.0)])
    images = {'groom': 6, 'bride and groom': 2, 'groom party': 2}
    shares = {'groom': 0.12, 'bride and groom': 0.12, 'groom party': 0.03}
    substitution = {**CONFIGS['pick_cpsat']['substitution'], 'affinity': {}}

    counts, _ = solve(frame, images, shares=shares, substitution=substitution, **retry())

    assert counts['bride and groom'] > 2
    assert counts['groom party'] == 2
