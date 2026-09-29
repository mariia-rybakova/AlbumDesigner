"""The layout plan aims for the album selection sized, not for its ceiling.

On 49994361 (2026-09-28 dev run) selection picked 125 photos for 18.8 spreads
of a 19-22 range, and the layout plan reduced its first estimate of 42 spreads
only as far as the ceiling, 22. With the album planned at its limit, the one
extra spread the layout search gave an 8-photo group took it to 23 -- and the
reduction had squeezed the dancing groups to 19 and 24 photos a spread to get
down to 22 at all.

    python -m pytest tests/test_layout_target.py -v
"""

import copy
import math

import pytest

from src.album_processing import layout_target
from utils.lookup_table_tools import WeddingLookUpTable, wedding_lookup_table

# The final groups of that run, as the second LUT pass saw them.
GROUPS_49994361 = {
    (1, 'bride', -1): 11, (1, 'bride getting dressed', -1): 8, (2, 'bride party', -1): 6,
    (4, 'ceremony', 0): 9, (4, 'groom', -1): 5, (4, 'groom party', 0): 3,
    (4, 'kiss', -1): 5, (4, 'send off', 2): 2, (5, 'bride and groom', -1): 4,
    (6, 'entertainment', -1): 4, (6, 'full party', -1): 4, (8, 'bride and groom', -1): 9,
    (9, 'speech', -1): 5, (10, 'dancing', 0): 19, (10, 'dancing', 1): 24,
    (10, 'first dance', -1): 5,
}


@pytest.fixture
def table():
    """A wedding LUT on a private copy of the class priors (`get_table` mutates them)."""
    t = WeddingLookUpTable(copy.deepcopy(wedding_lookup_table))
    t.get_table(GROUPS_49994361, logger=None, density=3)
    return t


def planned(table, groups):
    """The spread count the table now asks for, as the layout reads it."""
    return sum(max(1, math.ceil(n / table.get_current_spread_parameters(key, n)[0]))
               for key, n in groups.items())


# -- layout_target ---------------------------------------------------------


def test_the_target_is_selections_total_rounded():
    assert layout_target(18.78, 18, 22) == 19


def test_the_target_stays_inside_the_albums_range():
    assert layout_target(30.0, 18, 22) == 22, "the ceiling is still the limit"
    assert layout_target(12.0, 18, 22) == 18, "a design's minimum still stands"


@pytest.mark.parametrize("missing", [None, 0, 0.0])
def test_no_plan_means_no_target(missing):
    assert layout_target(missing, 18, 22) is None


# -- update_with_limit -----------------------------------------------------


def test_a_reduction_stops_at_the_target_not_the_ceiling(table):
    before = planned(table, GROUPS_49994361)
    assert before > 22, "the fixture no longer needs reducing"

    table.update_with_limit(GROUPS_49994361, max_total_spreads=22, min_total_spreads=18,
                            per_group=True, target_total_spreads=19)

    assert planned(table, GROUPS_49994361) <= 19


def test_without_a_target_the_ceiling_decides_as_before(table):
    table.update_with_limit(GROUPS_49994361, max_total_spreads=22, min_total_spreads=18,
                            per_group=True)

    assert planned(table, GROUPS_49994361) <= 22


def test_a_target_never_expands_a_small_plan(table):
    """The target only says where a reduction stops. An album already below it
    is left alone -- selection's plan is not a floor."""
    small = {(1, 'bride', -1): 6, (4, 'ceremony', 0): 6}
    before = planned(table, small)

    table.update_with_limit(small, max_total_spreads=22, min_total_spreads=1,
                            per_group=True, target_total_spreads=19)

    assert planned(table, small) == before
