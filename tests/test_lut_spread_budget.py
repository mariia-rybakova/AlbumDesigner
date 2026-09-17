"""The LUT's spread budget: does an expansion plan survive into the table?

`update_with_limit` plans a spread count per group and then encodes it as
photos-per-spread. Every number below is from the 2026-09-17T11:32 production
run of project 53739038 (condition AC.6755bda1-...0.1), where a plan of 17
spreads was encoded as 15 and the album came back 4 compositions short.
"""

import copy
import math

import pytest

from utils.configs import CONFIGS
from utils.lookup_table_tools import WeddingLookUpTable, wedding_lookup_table

# Detected groups: (time_cluster, cluster_context) -> photos
RUN1_INITIAL = {
    (0, 'None'): 2, (0, 'bride'): 1, (0, 'bride and groom'): 15,
    (0, 'bride walking the aisle'): 1, (0, 'cake cutting'): 1,
    (0, 'ceremony'): 11, (0, 'couple'): 11, (0, 'first dance'): 2,
    (0, 'groom'): 2, (0, 'groom walking the aisle'): 1, (0, 'kiss'): 4,
    (0, 'may kiss bride'): 1, (0, 'other'): 2, (0, 'vehicle'): 1,
    (0, 'walking the aisle'): 1,
}

# After illegal-group handling: (time_cluster, cluster_context, sub_index)
RUN1_FINAL = {
    (0, 'bride and groom', 0): 7, (0, 'bride and groom', 1): 11,
    (0, 'ceremony', -1): 12, (0, 'couple', -1): 11, (0, 'groom', -1): 3,
    (0, 'kiss', -1): 6, (0, 'vehicle', -1): 3, (0, 'walking the aisle', -1): 3,
}

RUN1_TARGET = 17


@pytest.fixture
def table():
    """A wedding LUT on a private copy of the class priors.

    `get_table` mutates the module-level `wedding_lookup_table` in place, so a
    test that used it would leak its density scaling into every later test.
    """
    return WeddingLookUpTable(copy.deepcopy(wedding_lookup_table))


def implied_spreads(table, groups):
    """The spread count the table now asks for, as layout will read it."""
    return sum(
        math.ceil(n / table.get_current_spread_parameters(key, n)[0])
        for key, n in groups.items()
    )


# --------------------------------------------------------------------------
# The regression: a plan that did not survive encoding
# --------------------------------------------------------------------------

def test_run1_plan_survives_into_the_table(table):
    """17 planned spreads must read back as 17, not 15.

    Both halves of the original miss are covered: `ceremony` and `couple` were
    handed 5 and 4 spreads against a 3-spread cap, and the two `bride and
    groom` groups -- 18 of the 56 photos -- were skipped as dense by design
    because the preceding reduction had raised their prior to 15.
    """
    table.update_with_limit(RUN1_INITIAL, max_total_spreads=RUN1_TARGET,
                            min_total_spreads=RUN1_TARGET)
    table.update_with_limit(RUN1_FINAL, max_total_spreads=RUN1_TARGET,
                            min_total_spreads=RUN1_TARGET, per_group=True)

    assert implied_spreads(table, RUN1_FINAL) == RUN1_TARGET


def test_run1_no_group_is_planned_past_the_cap(table):
    """Nothing may be asked for more spreads than `max_group_spread`."""
    table.update_with_limit(RUN1_INITIAL, max_total_spreads=RUN1_TARGET,
                            min_total_spreads=RUN1_TARGET)
    table.update_with_limit(RUN1_FINAL, max_total_spreads=RUN1_TARGET,
                            min_total_spreads=RUN1_TARGET, per_group=True)

    for key, n in RUN1_FINAL.items():
        spreads = math.ceil(n / table.get_current_spread_parameters(key, n)[0])
        assert spreads <= CONFIGS['max_group_spread'], key


def test_run1_bride_and_groom_is_diluted_not_protected(table):
    """The album's largest class must take part in the expansion."""
    table.update_with_limit(RUN1_INITIAL, max_total_spreads=RUN1_TARGET,
                            min_total_spreads=RUN1_TARGET)
    inflated, _ = table.get_spread_params((0, 'bride and groom'))
    assert inflated >= CONFIGS['expansion_dense_threshold'], (
        'premise: the reduction raises this prior past the dense threshold')

    table.update_with_limit(RUN1_FINAL, max_total_spreads=RUN1_TARGET,
                            min_total_spreads=RUN1_TARGET, per_group=True)

    for key in ((0, 'bride and groom', 0), (0, 'bride and groom', 1)):
        assert table.get_spread_params(key)[0] < inflated, key


# --------------------------------------------------------------------------
# The pieces, on their own
# --------------------------------------------------------------------------

def test_realisable_spreads_respects_the_per_group_cap():
    cap = CONFIGS['max_group_spread']
    assert WeddingLookUpTable._realisable_spreads(12, 2) == cap
    assert WeddingLookUpTable._realisable_spreads(100, 2) > cap, (
        'capacity outranks the cap when a group cannot fit inside it')


def test_realisable_spreads_yields_to_per_spread_capacity():
    """A group too big for `max_group_spread` spreads keeps what capacity forces."""
    per_spread = CONFIGS['max_imges_per_spread']
    n = per_spread * CONFIGS['max_group_spread'] * 2
    assert WeddingLookUpTable._realisable_spreads(n, 1) == math.ceil(n / per_spread)


def test_realisable_spreads_never_returns_zero():
    assert WeddingLookUpTable._realisable_spreads(1, 2) == 1


def test_a_genuinely_dense_class_is_still_protected(table):
    """`dancing` sits at 24/spread by design and must not be diluted first."""
    dense = CONFIGS['expansion_dense_threshold']
    assert table.get_spread_params((0, 'dancing'))[0] >= dense

    assert table._packed_by_design((0, 'dancing'), dense) is True
    assert table._packed_by_design((0, 'bride and groom'), dense) is False


def test_expansion_encoding_is_the_inverse_of_how_it_is_read(table):
    """`ceil(n / value)` is how the table is read, so writing must use `ceil`."""
    groups = {(0, 'couple', -1): 11}
    table.update_with_limit(groups, max_total_spreads=3, min_total_spreads=3)
    table.update_with_limit(groups, max_total_spreads=3, min_total_spreads=3,
                            per_group=True)
    assert implied_spreads(table, groups) == 3


def test_reduction_still_shortens_an_overlong_album(table):
    """The untouched direction: run 1's 20 spreads come back as 17."""
    groups = dict(RUN1_INITIAL)
    assert implied_spreads(table, groups) == 20, 'premise: the run started at 20'

    table.update_with_limit(groups, max_total_spreads=RUN1_TARGET,
                            min_total_spreads=None)
    assert implied_spreads(table, groups) == RUN1_TARGET


def test_reduction_cannot_go_below_one_spread_per_group(table):
    """Every group needs a spread, so the ceiling is not always reachable."""
    groups = dict(RUN1_INITIAL)
    table.update_with_limit(groups, max_total_spreads=8, min_total_spreads=None)
    assert implied_spreads(table, groups) == len(groups)
