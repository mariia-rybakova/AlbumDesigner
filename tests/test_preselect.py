"""Tests for ``select.preselect`` -- the constraints, and what they cost.

`tests/test_selection_equivalence.py` covers the other side of this: with every
constraint switched off the pipeline still reproduces the monolith exactly, so
each departure measured here is the constraint and not drift underneath it.

    python -m pytest tests/test_preselect.py -v
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import SELECT, AlbumContext, Col  # noqa: E402
from src.pipeline.contracts import GalleryFacts, KeyPages, SelectionOutcome  # noqa: E402
from src.pipeline.registry import get  # noqa: E402
from src.pipeline.select.contracts import SelectionInputs, SelectionPlan  # noqa: E402
from utils.configs import CONFIGS  # noqa: E402

BRIDE, GROOM, GUEST = 101, 202, 303


@contextmanager
def only(**flags):
    """Run with just the named constraints on."""
    original = CONFIGS['preselect']
    off = {'user_picks': False, 'identities': False, 'key_pages': False,
           'yes_categories': False}
    CONFIGS['preselect'] = {**original, **off, **flags}
    try:
        yield
    finally:
        CONFIGS['preselect'] = original


def gallery():
    """Three categories: one percentage-budgeted, two `yes`.

    ``image_order`` ascends with the index, and 0 is the best rank, so photo
    1000 is the strongest of every category it appears in.
    """
    rows = []
    image_id = 1000
    for category in ('bride and groom', 'rings', 'invite'):
        for i in range(6):
            rows.append({
                Col.IMAGE_ID: image_id,
                Col.CLUSTER_CONTEXT: category,
                Col.IMAGE_ORDER: float(i),
                Col.PERSONS_IDS: [BRIDE, GROOM] if i % 2 else [GUEST],
                Col.N_FACES: 2,
                Col.IMAGE_ORIENTATION: 'landscape',
                Col.EMBEDDING: np.linspace(i, i + 1, 8).astype(np.float32),
            })
            image_id += 1
    return pd.DataFrame(rows)


def run(photos=None, *, user_picks=(), person_ids=(), key_pages=None,
        images=None, yes_categories=('rings', 'invite')):
    photos = gallery() if photos is None else photos
    logger = logging.getLogger("preselect-test")
    logger.addHandler(logging.NullHandler())

    context = AlbumContext(
        logger=logger,
        photos=photos,
        facts=GalleryFacts(is_wedding=True),
        key_pages=key_pages,
        selection=SelectionOutcome(manual=False),
        selection_inputs=SelectionInputs(
            user_selected_ids=list(user_picks),
            person_ids=list(person_ids),
        ),
        selection_plan=SelectionPlan(
            images=dict(images or {'bride and groom': 3, 'rings': 1, 'invite': 1}),
            yes_categories=tuple(yes_categories),
        ),
    )
    return get("select.preselect")()(context)


def reasons(context, kind):
    return {i for i, why in context.selection_plan.committed.items()
            if why.split(':', 1)[0] == kind}


# -- composition -----------------------------------------------------------


def test_runs_between_budget_and_pick():
    order = list(SELECT)
    assert order.index("select.budget") < order.index("select.preselect")
    assert order.index("select.preselect") < order.index("select.pick")


def test_skipped_for_a_manual_request():
    """The user already chose everything; there is no allowance to constrain."""
    context = run()
    context.selection.manual = True
    context = get("select.preselect")()(context)
    assert context.diagnostics[-1].note == "skipped"


# -- user picks ------------------------------------------------------------


def test_a_hand_picked_photo_is_committed_whatever_it_scores():
    with only(user_picks=True):
        context = run(user_picks=[1004, 1013])

    assert reasons(context, 'user') == {1004, 1013}


def test_hand_picks_are_not_charged_to_their_category():
    """Asking for a photo should not cost you another one -- the rule the
    monolith already applied, adding the user's picks on top of the budget."""
    with only(user_picks=True):
        context = run(user_picks=[1004])

    assert context.selection_plan.images['bride and groom'] == 3


def test_a_pick_outside_the_pool_is_ignored():
    with only(user_picks=True):
        context = run(user_picks=[999999])

    assert context.selection_plan.committed == {}


# -- identity coverage -----------------------------------------------------


def test_a_named_identity_is_guaranteed_a_photo():
    """Until now `personIds` only fed `person_score`, so a requested person
    could be ranked up everywhere and still appear in nothing."""
    with only(identities=True):
        context = run(person_ids=[GUEST])

    committed = reasons(context, 'identity')
    assert len(committed) == 1
    row = gallery().set_index(Col.IMAGE_ID).loc[committed.pop()]
    assert GUEST in row[Col.PERSONS_IDS]


def test_each_named_identity_gets_its_own():
    with only(identities=True):
        context = run(person_ids=[GUEST, BRIDE])

    assert len(reasons(context, 'identity')) == 2


def test_one_photo_covers_two_identities_at_once():
    """The couple share frames, so covering both costs one photo, not two."""
    with only(identities=True):
        context = run(person_ids=[BRIDE, GROOM])

    assert len(reasons(context, 'identity')) == 1


def test_coverage_depth_is_configurable():
    with only(identities=True):
        original = CONFIGS['preselect']['photos_per_identity']
        CONFIGS['preselect'] = {**CONFIGS['preselect'], 'photos_per_identity': 3}
        try:
            context = run(person_ids=[GUEST])
        finally:
            CONFIGS['preselect'] = {**CONFIGS['preselect'],
                                    'photos_per_identity': original}

    assert len(reasons(context, 'identity')) == 3


def test_an_identity_nobody_photographed_commits_nothing():
    with only(identities=True):
        context = run(person_ids=[999])

    assert context.selection_plan.committed == {}


# -- key pages -------------------------------------------------------------


def test_the_covers_are_committed():
    """They were chosen over the whole gallery, so nothing else guaranteed
    selection would keep them -- and ProcessStage takes its covers from the
    selected pool."""
    with only(key_pages=True):
        context = run(key_pages=KeyPages(opening=[1001], closing=[1005]))

    assert reasons(context, 'key_page') == {1001, 1005}


def test_a_cover_falls_through_to_the_next_candidate():
    """What the ranked lists are for: the best opening photo may not have
    survived into the pool selection is working from."""
    with only(key_pages=True):
        context = run(key_pages=KeyPages(opening=[999999, 1002], closing=[]))

    assert reasons(context, 'key_page') == {1002}


def test_only_one_photo_is_taken_per_end():
    with only(key_pages=True):
        context = run(key_pages=KeyPages(opening=[1001, 1002, 1003], closing=[1004]))

    assert reasons(context, 'key_page') == {1001, 1004}


def test_no_covers_is_not_an_error():
    with only(key_pages=True):
        context = run(key_pages=None)

    assert not context.failed
    assert context.selection_plan.committed == {}


# -- yes categories --------------------------------------------------------


def test_a_yes_category_is_resolved_outright():
    """Promised one photo if the thing happened: there is no allowance to
    divide and nothing for the ranked picker to weigh."""
    with only(yes_categories=True):
        context = run()

    assert reasons(context, 'yes') == {1006, 1012}, "the best-ranked ring and invite"


def test_a_yes_category_is_charged_its_whole_allowance():
    """Unlike the other three: here the commitment *is* the allowance, so the
    need has to reach zero or the picker would take another on top."""
    with only(yes_categories=True):
        context = run()

    assert context.selection_plan.images['rings'] == 0
    assert context.selection_plan.images['invite'] == 0
    assert context.selection_plan.images['bride and groom'] == 3, "untouched"


def test_a_percentage_category_is_left_to_the_picker():
    with only(yes_categories=True):
        context = run()

    committed = set(context.selection_plan.committed)
    body = set(gallery().loc[gallery()[Col.CLUSTER_CONTEXT] == 'bride and groom',
                             Col.IMAGE_ID])
    assert not (committed & body)


def test_a_yes_category_the_gallery_lacks_commits_nothing():
    """`yes_categories` on the plan is already filtered to what the gallery
    has, but a category with a zero allowance must be skipped too."""
    with only(yes_categories=True):
        context = run(images={'bride and groom': 3, 'rings': 0, 'invite': 1})

    assert reasons(context, 'yes') == {1012}


def test_a_hand_pick_satisfies_its_yes_category():
    """Order matters: the user's own choice is honoured first, so the `yes`
    rule finds the slot filled instead of adding a second ring shot."""
    with only(user_picks=True, yes_categories=True):
        context = run(user_picks=[1009])

    assert 1009 in reasons(context, 'user')
    assert 1009 not in reasons(context, 'yes')
    assert context.selection_plan.images['rings'] == 0, "charged by the yes rule"
    assert len([i for i in context.selection_plan.committed
                if i in range(1006, 1012)]) == 1, "one ring photo, not two"


# -- the switches ----------------------------------------------------------


def test_everything_off_commits_nothing():
    with only():
        context = run(user_picks=[1004], person_ids=[GUEST],
                      key_pages=KeyPages(opening=[1001], closing=[1005]))

    assert context.selection_plan.committed == {}
    assert context.selection_plan.images == {'bride and groom': 3, 'rings': 1, 'invite': 1}


def test_reasons_are_recorded_for_every_commitment():
    """The reason is what makes a surprising album explainable afterwards."""
    context = run(user_picks=[1004], person_ids=[GUEST],
                  key_pages=KeyPages(opening=[1001], closing=[1005]))

    committed = context.selection_plan.committed
    assert committed, "the default configuration should commit something"
    kinds = {why.split(':', 1)[0] for why in committed.values()}
    assert kinds <= {'user', 'identity', 'key_page', 'yes'}
    assert 'user' in kinds and 'yes' in kinds


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
