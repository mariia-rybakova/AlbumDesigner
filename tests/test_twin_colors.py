"""Treatment twins in the layout: colour-neutral while scoring, settled after.

A shot the gallery holds in colour and in black and white can be placed in
either, so the layout does not count it when it asks whether a page mixes
colours, and once the pages are fixed each page's twins take its colour.

    python -m pytest tests/test_twin_colors.py -v
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.models import GroupProcessingResult, Spread  # noqa: E402
from src.core.photos import Photo, get_photos_from_df  # noqa: E402
from src.spreads_layout.spreads.spread import SingleSpreadLayout  # noqa: E402
from src.spreads_layout.twins import apply_twin_colors, settle_page  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def photo(i, colour, twin=None):
    return Photo(id=i, ar=0.67, color=colour, rank=i, photo_class='ceremony', cluster_label=1,
                 general_time=i, original_context='ceremony',
                 twin_id=twin, twin_color=(not colour) if twin is not None else None)


def same_colour(page):
    return SingleSpreadLayout.check_page_properties(set(range(len(page))), page).is_same_color


def test_a_twin_does_not_make_a_page_mixed():
    page = [photo(1, True), photo(2, True), photo(3, False, twin=30)]
    assert same_colour(page)


def test_a_black_and_white_photo_without_a_twin_still_does():
    assert not same_colour([photo(1, True), photo(2, True), photo(3, False)])


def test_the_twin_takes_the_colour_of_its_page():
    page, swapped = settle_page([photo(1, True), photo(2, True), photo(3, False, twin=30)])

    assert swapped == 1
    assert [p.color for p in page] == [True, True, True]
    assert page[2].output_id == 30 and page[2].id == 3


def test_twins_on_one_page_all_take_one_colour():
    """Every photo a twin: the majority of their own colours, for all of them."""
    page, swapped = settle_page([photo(1, False, twin=10), photo(2, False, twin=20),
                                 photo(3, True, twin=30)])

    assert swapped == 1
    assert {p.color for p in page} == {False}


def test_a_tie_among_twins_goes_to_colour():
    page, _ = settle_page([photo(1, False, twin=10), photo(2, True, twin=20)])
    assert {p.color for p in page} == {True}


def test_a_page_without_twins_is_left_alone():
    page = [photo(1, True), photo(2, False)]
    assert settle_page(page) == (page, 0)


def test_apply_settles_every_page_of_the_album():
    spread = Spread(layout_id=0, left_photos=[photo(1, True), photo(2, False, twin=20)],
                    right_photos=[photo(3, False), photo(4, True, twin=40)])
    result = [{'g': GroupProcessingResult(group_name='g', spreads=[spread])}]

    assert apply_twin_colors(result) == 2
    assert [p.color for p in spread.left_photos] == [True, True]
    assert [p.color for p in spread.right_photos] == [False, False]
    assert [p.output_id for p in spread.left_photos + spread.right_photos] == [None, 20, None, 40]


def test_photos_read_their_twin_from_the_frame():
    frame = pd.DataFrame([
        {'image_id': 1, 'cluster_context': 'c', 'cluster_label': 1, 'image_color': 1, 'image_as': 1.5,
         'image_order': 0, 'general_time': 0.0, 'treatment_twin': 11, 'twin_color': 0},
        {'image_id': 2, 'cluster_context': 'c', 'cluster_label': 1, 'image_color': 1, 'image_as': 1.5,
         'image_order': 1, 'general_time': 1.0, 'treatment_twin': None, 'twin_color': None},
    ])
    first, second = get_photos_from_df(frame, is_wedding=True)

    assert (first.twin_id, first.twin_color) == (11, False)
    assert (second.twin_id, second.twin_color) == (None, None)


def test_cpsat_prefers_a_shot_it_holds_in_both_treatments():
    """The bonus is what moves the pick. With no hints the two photos are
    ranked 1.0 and 0.5 by order, a gap of 500: a bonus above it flips the
    choice to the twin, none leaves it."""
    import numpy as np
    from test_cpsat import pick
    from src.pipeline.contracts import Col
    frame = pd.DataFrame([
        {Col.IMAGE_ID: 1000 + i, Col.CLUSTER_CONTEXT: 'portrait', Col.IMAGE_CLASS: 21,
         Col.PERSONS_IDS: [], Col.IMAGE_SUBQUERY_CONTENT: f'shot {i}', Col.GENERAL_TIME: float(i),
         Col.IMAGE_ORDER: 5.0, Col.IMAGE_COLOR: 1, Col.MODEL_VERSION: 2,
         Col.EMBEDDING: np.eye(8, dtype=np.float32)[i], Col.BRIDE_ID: 1, Col.GROOM_ID: 2,
         Col.TREATMENT_TWIN: 5000 if i == 1 else None}
        for i in range(2)])

    assert pick(frame, {'portrait': 1}, twin_bonus=0) == {1000}
    assert pick(frame, {'portrait': 1}, twin_bonus=600) == {1001}


def test_the_shipped_twin_bonus_is_minor():
    """Enough to break a near-tie, never to take a clearly weaker photo: well
    under a tenth of the rank term's 1000."""
    from utils.configs import CONFIGS
    assert 0 < CONFIGS['pick_cpsat']['twin_bonus'] < 100


def test_a_twin_id_that_became_a_float_is_placed_as_an_integer():
    """A merge upstream can turn the id column into floats; the reply must not
    place 11547936405.0."""
    frame = pd.DataFrame([
        {'image_id': 1, 'cluster_context': 'c', 'cluster_label': 1, 'image_color': 1, 'image_as': 1.5,
         'image_order': 0, 'general_time': 0.0, 'treatment_twin': 11547936405.0, 'twin_color': 0.0},
    ])
    (only,) = get_photos_from_df(frame, is_wedding=True)

    assert only.twin_id == 11547936405 and isinstance(only.twin_id, int)
    page, _ = settle_page([photo(2, False), only])
    assert isinstance(page[1].output_id, int)
