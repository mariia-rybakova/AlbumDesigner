"""A broken spread ratio must be reported where it breaks.

On 2026-09-18 two `AC.` albums failed with `'NoneType' object has no attribute
'items'` in `_compute_initial_spreads` -- a function that had done nothing
wrong. The real failure was eight lines upstream: `_update_group_spreads` could
not set its column, `process_wedding_illegal_groups` caught that and returned
`None, None, None`, and `album_processing` fed the middle one straight into
`update_with_limit`.

The pandas message it started from -- "Cannot set a DataFrame with multiple
columns to the single column group_spreads" -- only appears when the per-row
function answers with something that is not a number, which is what a
duplicated column makes of `row['group_size']`.
"""
import pandas as pd
import pytest

from src.groups_operations.groups_management import _update_group_spreads
from utils.lookup_table_tools import WeddingLookUpTable


class _Table:
    """Just the one method `_update_group_spreads` calls."""

    def compute_spreads_number(self, cluster_context, group_size):
        return group_size / 2


def photos(columns=('image_id', 'cluster_context', 'group_size')):
    rows = [[1, 'bride', 4], [2, 'ceremony', 6]]
    return pd.DataFrame(rows, columns=list(columns))


def test_spread_ratios_are_one_number_per_row():
    df = photos()
    _update_group_spreads(df, _Table())
    assert list(df['group_spreads']) == [2.0, 3.0]
    assert df['group_spreads'].dtype == 'float64'


def test_a_duplicate_column_is_named_where_it_breaks():
    df = photos()
    df.insert(len(df.columns), 'group_size', [4, 6], allow_duplicates=True)

    with pytest.raises(ValueError) as caught:
        _update_group_spreads(df, _Table())

    # The column, not pandas' account of the assignment it could not make.
    assert 'group_size' in str(caught.value)
    assert 'duplicate' in str(caught.value).lower()


def test_no_look_up_table_still_gives_every_row_a_ratio():
    df = photos()
    _update_group_spreads(df, None)
    assert list(df['group_spreads']) == [1.0, 1.0]


def test_sizing_an_album_from_none_says_so():
    """The symptom that used to be reported instead of the cause."""
    with pytest.raises(ValueError) as caught:
        WeddingLookUpTable()._compute_initial_spreads(None)

    assert 'group2images is None' in str(caught.value)


def test_update_with_limit_refuses_none_too():
    with pytest.raises(ValueError):
        WeddingLookUpTable().update_with_limit(None, max_total_spreads=17)
