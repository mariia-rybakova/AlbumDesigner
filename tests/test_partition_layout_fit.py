"""Does a partition of a group into spreads fit the design's layouts?

Every number below is from the 2026-09-23T09:16 dev run of project 53819935
(condition AC.187f011a-...0.1). Its design has one 1-box spread layout, for a
landscape only, and six 2-box ones, listed here in the order the layouts frame
holds them -- the order the greedy check walks. The LUT asked for 14 spreads;
`filter_by_layout` kept only single-spread partitions and the album came back
with 6.
"""

import pandas as pd
import pytest

from src.core.models import SpreadSearchParams
from src.spreads_layout.layouts_tools import get_layouts_dict
from src.spreads_layout.partitions import Partition, get_partitions

# (number of boxes, max portraits, max landscapes)
LAYOUT_ROWS = [
    (1, 0, 1),
    (2, 0, 2), (2, 2, 2), (2, 2, 0), (2, 1, 1), (2, 2, 1), (2, 1, 2),
]

PORTRAIT, LANDSCAPE = 0.67, 1.5


@pytest.fixture
def layouts_df():
    return pd.DataFrame(LAYOUT_ROWS, columns=['number of boxes', 'max portraits', 'max landscapes'])


@pytest.fixture
def layouts_dict(layouts_df):
    return get_layouts_dict(layouts_df, set(layouts_df['number of boxes']))


def _kept(sizes, layouts_dict, n_portraits, n_landscapes, params=None):
    part = Partition(sizes, weight=1.0)
    return Partition.filter_by_layout([part], layouts_dict, n_portraits, n_landscapes,
                                      params or SpreadSearchParams()) == [part]


def test_three_portraits_and_a_landscape_fit_two_spreads_of_two(layouts_dict):
    """None|9: the (2, 2) row used up three photos for the first spread, so
    the greedy check had one photo left for the second and rejected `[2, 2]`.
    (2, 0) and then (1, 1) holds it."""
    assert not Partition._fits_greedily([2, 2], layouts_dict, 3, 1), "premise: the greedy check rejects it"
    assert _kept([2, 2], layouts_dict, 3, 1)


def test_a_portrait_and_two_landscapes_fit_two_then_one(layouts_dict):
    """None|8: the portrait and a landscape on (1, 1), the other landscape alone."""
    assert not Partition._fits_greedily([2, 1], layouts_dict, 1, 2), "premise: the greedy check rejects it"
    assert _kept([2, 1], layouts_dict, 1, 2)


@pytest.mark.parametrize('n_portraits, n_landscapes', [(1, 1), (2, 0)])
def test_a_portrait_cannot_be_alone_on_a_spread_this_design_has_no_layout_for(
        layouts_dict, n_portraits, n_landscapes):
    """The four 2-photo groups: `[1, 1]` is out because of the design, not the check."""
    assert not _kept([1, 1], layouts_dict, n_portraits, n_landscapes)
    assert not _kept([1, 1, 1], layouts_dict, 1, 2)


def test_two_landscapes_may_each_take_a_spread(layouts_dict):
    assert _kept([1, 1], layouts_dict, 0, 2)


def test_more_portraits_than_the_layouts_hold_do_not_fit(layouts_dict):
    assert not _kept([2, 1], layouts_dict, 3, 0)


def test_groups_above_the_small_group_threshold_keep_the_greedy_check(layouts_dict):
    """The exact check lets more partitions through to the combination search,
    so large groups, where that search is expensive, keep the greedy one."""
    params = SpreadSearchParams(small_group_threshold=3)
    assert not _kept([2, 2], layouts_dict, 3, 1, params)


def test_the_four_photo_group_is_offered_two_spreads(layouts_df):
    """End to end on None|9 as the layout stage sees it: LUT mean 2, std 0.5."""
    photos_df = pd.DataFrame({'ar': [PORTRAIT, PORTRAIT, LANDSCAPE, PORTRAIT]})
    parts = get_partitions(photos_df, [2, 0.5], SpreadSearchParams(), layouts_df)
    assert parts and parts[0].spread_sizes == [2, 2]
