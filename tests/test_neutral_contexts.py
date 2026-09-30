"""`yes` and 0% contexts do not make a page mixed.

On 49994361 a single `accessories`, `settings` or `food` photo, merged into a
moment's group because it was too small to stand alone, counted as a context of
its own and split four groups into two spreads apiece.

    python -m pytest tests/test_neutral_contexts.py -v
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.photos import Photo  # noqa: E402
from src.pipeline.select.budget import neutral_contexts  # noqa: E402
from src.spreads_layout.spreads.spread import SingleSpreadLayout, set_neutral_contexts  # noqa: E402


def _photo(i, context):
    return Photo(id=i, ar=0.67, color=True, rank=i, photo_class='bride', cluster_label=1,
                 general_time=i, original_context=context)


PAGE = [_photo(0, 'bride getting dressed'), _photo(1, 'bride getting dressed'),
        _photo(2, 'accessories')]


def contexts_on(page):
    return SingleSpreadLayout.check_page_properties(set(range(len(page))), page).number_of_unique_contexts


def test_a_yes_photo_counts_as_a_context_without_the_set():
    set_neutral_contexts(None)
    assert contexts_on(PAGE) == 2


def test_a_yes_photo_rides_along_with_its_moment():
    set_neutral_contexts({'accessories'})
    try:
        assert contexts_on(PAGE) == 1
    finally:
        set_neutral_contexts(None)


def test_two_real_moments_still_mix():
    set_neutral_contexts({'accessories'})
    try:
        assert contexts_on([_photo(0, 'kiss'), _photo(1, 'ceremony'), _photo(2, 'accessories')]) == 2
    finally:
        set_neutral_contexts(None)


def test_a_page_of_only_neutral_photos_is_one_context():
    set_neutral_contexts({'accessories', 'settings'})
    try:
        assert contexts_on([_photo(0, 'accessories'), _photo(1, 'settings')]) == 1
    finally:
        set_neutral_contexts(None)


def test_the_set_is_the_profiles_yes_and_zero_categories():
    neutral = neutral_contexts(['brideAndGroom'])

    assert {'accessories', 'settings', 'food', 'other', 'None', 'send off'} <= neutral
    assert not {'bride', 'ceremony', 'kiss', 'dancing', 'bride and groom'} & neutral
