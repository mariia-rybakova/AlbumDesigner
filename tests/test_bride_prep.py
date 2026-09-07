"""Tests for the getting-ready categories: keep the bride, and the same bride.

Two separate faults put a stranger on a real album's hair-and-makeup spread,
and both are pinned here.

    python -m pytest tests/test_bride_prep.py -v
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import AlbumContext, Col  # noqa: E402
from src.pipeline.contracts import GalleryFacts, SelectionOutcome  # noqa: E402
from src.pipeline.registry import get  # noqa: E402
from src.pipeline.select.contracts import (  # noqa: E402
    CategoryRequest,
    SelectionInputs,
    SelectionPlan,
)
from src.pipeline.select.strategies import default_registry  # noqa: E402
from src.pipeline.select.strategies.bride_prep import BridePrepStrategy  # noqa: E402
from utils.configs import CONFIGS  # noqa: E402

BRIDE, STRANGER, MOTHER = 2, 6, 8
CATEGORY = 'getting hair-makeup'


def quiet():
    logger = logging.getLogger("bride-prep-test")
    logger.addHandler(logging.NullHandler())
    return logger


def frames(rows):
    """``rows`` of ``(image_id, persons_ids, subquery, rank)``."""
    return pd.DataFrame([
        {Col.IMAGE_ID: image_id, Col.PERSONS_IDS: list(people),
         Col.IMAGE_SUBQUERY_CONTENT: subquery, Col.IMAGE_ORDER: float(rank),
         Col.BRIDE_ID: BRIDE, Col.GROOM_ID: 10, Col.CLUSTER_CONTEXT: CATEGORY,
         Col.IMAGE_COLOR: 1, Col.EMBEDDING: np.linspace(rank, rank + 1, 8)}
        for image_id, people, subquery, rank in rows
    ])


def request(color, need=2):
    return CategoryRequest(category=CATEGORY, need=need, color=color,
                           grayscale=color.iloc[0:0], pool=color,
                           scored=False,
                           order_index=dict(zip(color[Col.IMAGE_ID],
                                                color[Col.IMAGE_ORDER])),
                           logger=quiet())


def chosen(color, need=2):
    picks = BridePrepStrategy().pick(request(color, need))
    return list(picks.preferred or [])


# -- the subject -----------------------------------------------------------


def test_the_bride_is_found_by_her_identity():
    """`bride_id` is what `enrich.identities` resolved. The strategy used to
    ignore it and match the substring 'bride' in the subquery text instead,
    which admits bridesmaid, bridal suite and the bride's mother."""
    color = frames([
        (1, [STRANGER], 'unknown_getting_hair_makeup', 0),   # best rank, wrong person
        (2, [STRANGER], 'unknown_getting_hair_makeup', 1),
        (3, [BRIDE], 'unknown_getting_hair_makeup', 9),      # worst rank, right person
        (4, [BRIDE], 'unknown_getting_hair_makeup', 8),
    ])

    picks = chosen(color)

    assert set(picks) == {3, 4}, f"only the bride's frames should be kept, got {picks}"


def test_a_stranger_is_never_preferred_on_rank_alone():
    """What went wrong on gallery 53459898: seven of ten frames carried
    `unknown_getting_hair_makeup`, so there was no text to match, and the photo
    that reached the album contained `persons=[6]`."""
    color = frames([
        (1, [STRANGER], 'unknown_getting_hair_makeup', 0),
        (2, [BRIDE], 'bride getting makeup applied', 7),
    ])

    assert chosen(color, need=1) == [2]


def test_the_subquery_is_the_fallback_when_she_is_in_no_frame():
    """A gallery where the identity model missed her in her own prep shots."""
    color = frames([
        (1, [MOTHER], 'brides mother helping her', 3),
        (2, [MOTHER], 'brides mother helping her', 4),
        (3, [STRANGER], 'unknown_getting_hair_makeup', 0),
    ])

    picks = chosen(color)

    assert 3 not in picks, "the frame with no bride text should not win on rank"
    assert set(picks) <= {1, 2}


def test_the_whole_pool_is_the_last_resort():
    """Neither identity nor text says anything, so rank is all there is."""
    color = frames([
        (1, [STRANGER], 'unknown_getting_hair_makeup', 5),
        (2, [MOTHER], 'unknown_getting_hair_makeup', 0),
    ])

    assert chosen(color, need=1) == [2], "best rank when nothing else is known"


def test_identity_first_can_be_switched_off():
    color = frames([
        (1, [STRANGER], 'bride getting makeup applied', 0),
        (2, [BRIDE], 'unknown_getting_hair_makeup', 9),
    ])

    original = CONFIGS['bride_prep_by_identity']
    CONFIGS['bride_prep_by_identity'] = False
    try:
        by_text = chosen(color, need=1)
    finally:
        CONFIGS['bride_prep_by_identity'] = original

    assert by_text == [1], "the old rule follows the text"
    assert chosen(color, need=1) == [2], "the new rule follows the identity"


# -- and it has to be allowed to run at all --------------------------------


def test_a_yes_category_with_a_strategy_is_left_to_it():
    """`getting hair-makeup` carries `yes` in the profile *and* has a strategy.

    Resolving it in `select.preselect` zeroed the allowance, so the strategy
    never ran and the photo was chosen on rank alone -- which is how a stranger
    reached the album. The premise for settling a `yes` category early is that
    the picker adds nothing, and that is false wherever a strategy exists.
    """
    assert CATEGORY in set(default_registry().categories()), (
        "the premise of this test: the category has a strategy")

    photos = frames([
        (1, [STRANGER], 'unknown_getting_hair_makeup', 0),
        (2, [BRIDE], 'bride getting makeup applied', 7),
    ])
    context = AlbumContext(
        logger=quiet(), photos=photos, facts=GalleryFacts(is_wedding=True),
        selection=SelectionOutcome(manual=False),
        selection_inputs=SelectionInputs(),
        selection_plan=SelectionPlan(images={CATEGORY: 1},
                                     yes_categories=(CATEGORY,)),
    )

    context = get("select.preselect")()(context)

    assert context.selection_plan.committed == {}, "nothing should be committed"
    assert context.selection_plan.images[CATEGORY] == 1, (
        "and the allowance must survive for the strategy to spend")


def test_a_yes_category_without_a_strategy_is_still_settled():
    photos = frames([(1, [BRIDE], 'the invitation', 0)])
    photos[Col.CLUSTER_CONTEXT] = 'invite'
    context = AlbumContext(
        logger=quiet(), photos=photos, facts=GalleryFacts(is_wedding=True),
        selection=SelectionOutcome(manual=False),
        selection_inputs=SelectionInputs(),
        selection_plan=SelectionPlan(images={'invite': 1},
                                     yes_categories=('invite',)),
    )

    context = get("select.preselect")()(context)

    assert list(context.selection_plan.committed) == [1]
    assert context.selection_plan.images['invite'] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
