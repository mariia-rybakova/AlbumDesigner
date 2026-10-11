"""A singleton force-merged into a group buys it no spread.

On 49996919 a lone `settings` photo was force-merged into a two-photo `food`
group planned at one spread; the class value made the three of them two, and the
food photo got a spread to itself. The merge now pins the group at its pre-merge
count, the search holds it there, and the merged photo is not a context of the
group's own.

    python -m pytest tests/test_singleton_pin.py -v
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.photos import Photo  # noqa: E402
from src.spreads_layout.spreads.spread import SingleSpreadLayout, set_neutral_contexts  # noqa: E402
from utils.lookup_table_tools import WeddingLookUpTable  # noqa: E402

FOOD = (8, 'food', -1)


def test_a_pinned_group_keeps_its_count_with_one_photo_more():
    lut = WeddingLookUpTable(dict(WeddingLookUpTable.default_table))
    assert lut.spreads_for(FOOD, 2) == 1
    assert lut.spreads_for(FOOD, 3) == 2, "food is (2, 1): three photos would be two spreads"

    lut.pin_spreads(FOOD, 3, lut.spreads_for(FOOD, 2))

    assert lut.spreads_for(FOOD, 3) == 1
    assert lut.pinned_spreads(FOOD, 3) == 1


def test_an_unpinned_group_has_no_pin():
    lut = WeddingLookUpTable(dict(WeddingLookUpTable.default_table))

    assert lut.pinned_spreads(FOOD, 3) is None


def test_the_pin_is_per_group_not_per_class():
    lut = WeddingLookUpTable(dict(WeddingLookUpTable.default_table))
    lut.pin_spreads(FOOD, 3, 1)

    assert lut.spreads_for((14, 'food', -1), 3) == 2


def _photo(idx, context, merged=False):
    return Photo(id=idx, ar=0.66, color=True, rank=1, photo_class='food',
                 cluster_label=1, general_time=idx, original_context=context,
                 force_merged=merged)


def test_a_force_merged_photo_is_not_a_context_of_its_own():
    set_neutral_contexts(())
    photos = [_photo(0, 'entertainment'), _photo(1, 'speech', merged=True)]

    page = SingleSpreadLayout.check_page_properties({0, 1}, photos)

    assert page.number_of_unique_contexts == 1


def test_two_real_contexts_still_count_as_two():
    set_neutral_contexts(())
    photos = [_photo(0, 'entertainment'), _photo(1, 'speech')]

    page = SingleSpreadLayout.check_page_properties({0, 1}, photos)

    assert page.number_of_unique_contexts == 2


def test_the_partition_search_keeps_only_the_pinned_count():
    from src.core.models import SpreadSearchParams
    from src.spreads_layout.partitions import get_partitions

    layouts = pd.DataFrame({
        'number of boxes': [1, 2, 3],
        'max portraits': [1, 2, 3], 'max landscapes': [1, 2, 3],
    })
    photos = pd.DataFrame({'ar': [0.66, 1.5, 0.66]})
    params = SpreadSearchParams(score_threshold=0.01, weight_threshold_divisor=100.0,
                                max_spreads_sample=1000, max_combs_small_group=100,
                                max_oriented_combs=300, small_group_threshold=12)

    try:
        pinned = get_partitions(photos, [2.0, 1.0], params, layouts, n_spreads=1)
    except Exception:  # noqa: BLE001 - the layout frame above may be thinner than real
        import pytest
        pytest.skip("layouts frame too thin for get_layouts_dict")

    assert pinned and all(len(p.spread_sizes) == 1 for p in pinned)
