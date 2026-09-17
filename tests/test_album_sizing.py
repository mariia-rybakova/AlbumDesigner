"""The content-spread budget `album_processing` derives from a design.

The numbers below are compositions as the caller counts them, because that is
what a photographer sees: `size_album` returns content spreads, and the reply
adds `FIXED_COMPOSITIONS` (cover, first page, last page) on top.
"""

import pytest

from src.album_processing import FIXED_COMPOSITIONS, size_album
from utils.configs import CONFIGS


def design(min_pages, max_pages):
    return {'minPages': min_pages, 'maxPages': max_pages}


def compositions(min_total, max_total):
    """Read a spread budget back as the composition count it produces."""
    return min_total + FIXED_COMPOSITIONS, max_total + FIXED_COMPOSITIONS


# --------------------------------------------------------------------------
# minPages == maxPages: an exact album
# --------------------------------------------------------------------------

def test_exact_design_yields_exactly_that_many_compositions():
    """20 pages in, 20 compositions out -- no lower and no higher."""
    low, high = size_album(design(20, 20), selection_max_total_spreads=22)
    assert low == high == 17
    assert compositions(low, high) == (20, 20)


def test_exact_design_ignores_a_tighter_selection_ceiling():
    """Selection's gallery-aware proposal may not shrink a fixed-size design.

    This is the case the clamp used to lose: a small gallery lands in the
    `(15, 18)` branch of `define_min_max_spreads`, and before the special case
    `min(18, 17)` dragged a 20-composition design down to 18.
    """
    low, high = size_album(design(20, 20), selection_max_total_spreads=15)
    assert compositions(low, high) == (20, 20)


def test_exact_design_below_the_config_default_is_not_widened():
    """`max(CONFIGS[...], maxPages)` takes the larger, so an 11-page design was
    silently sized as if it were `CONFIGS['max_total_spreads']` pages."""
    assert CONFIGS['max_total_spreads'] > 11, "premise: the default is the larger of the two"

    low, high = size_album(design(11, 11), selection_max_total_spreads=22)
    assert compositions(low, high) == (11, 11)


def test_exact_design_never_asks_for_fewer_than_one_spread():
    """A design smaller than the fixed compositions still needs a spread."""
    low, high = size_album(design(2, 2), selection_max_total_spreads=22)
    assert low == high == 1


@pytest.mark.parametrize('pages', [11, 14, 20, 26])
def test_exact_design_round_trips_for_any_page_count(pages):
    low, high = size_album(design(pages, pages), selection_max_total_spreads=22)
    assert compositions(low, high) == (pages, pages)


# --------------------------------------------------------------------------
# minPages != maxPages: the range behaviour, unchanged
# --------------------------------------------------------------------------

def test_a_range_still_takes_the_margins_and_the_selection_clamp():
    """Guards the untouched path: `-3` on the ceiling, `+6` on the floor, and
    selection clamped to the hard ceiling."""
    low, high = size_album(design(10, 30), selection_max_total_spreads=22)
    assert high == 22                      # min(22, max(20, 30) - 3) == 22
    assert low == 16                       # min(22, 10 + 6)


def test_a_range_without_a_selection_target_keeps_the_design_limit():
    """Non-wedding has no selection proposal and keeps the hard ceiling."""
    low, high = size_album(design(10, 30), selection_max_total_spreads=None)
    assert high == max(CONFIGS['max_total_spreads'], 30) - FIXED_COMPOSITIONS


def test_a_range_floor_is_clamped_to_its_ceiling():
    """`minPages + 6` must never exceed the ceiling it is clamped against."""
    low, high = size_album(design(40, 41), selection_max_total_spreads=22)
    assert low == high == 22
