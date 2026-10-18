"""Groups that each fill a spread are paired only when the album is over its maximum.

54167191 (2026-10-17) left illegal-groups handling with 32 groups of 2-8
photos against a maximum of 20 spreads. None was a merge candidate (each
already filled a spread) and the reduction takes no group below one spread,
so the album was laid out at 37. `_merge_over_maximum` pairs such groups until
one spread per group fits, and does nothing to an album that already fits.

    python -m pytest tests/test_merge_over_maximum.py -v
"""

from __future__ import annotations

import logging
import os
import sys
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.groups_operations.groups_management import _merge_over_maximum  # noqa: E402
from utils.lookup_table_tools import WeddingLookUpTable  # noqa: E402

LOGGER = logging.getLogger('test_merge_over_maximum')
KEYS = ['time_cluster', 'cluster_context', 'group_sub_index']


def _photos(groups):
    """``groups``: [(context, size)], laid out one after another in time."""
    rows, t = [], 0
    for i, (context, size) in enumerate(groups):
        for _ in range(size):
            rows.append({'image_id': len(rows), 'time_cluster': 0,
                         'cluster_context': context, 'group_sub_index': -1,
                         'original_context': context.split('|')[0],
                         'general_time': float(t), 'group_size': size,
                         'groups_merged': 1, 'merge_allowed': True})
            t += 1
        t += 10
    return pd.DataFrame(rows)


def _resources(box_counts=(1, 2, 3, 4, 5, 6)):
    lut = WeddingLookUpTable(dict(WeddingLookUpTable.default_table))
    layouts = pd.DataFrame({'number of boxes': list(box_counts)})
    return SimpleNamespace(look_up_table=lut, printlab_data=SimpleNamespace(layouts_df=layouts))


SIX_PAIRS = [(f'other|{i}', 2) for i in range(6)]


def test_an_album_within_its_maximum_is_left_alone():
    df = _photos(SIX_PAIRS)
    before = df.copy()

    out = _merge_over_maximum(df, _resources(), max_total_spreads=6, logger=LOGGER)

    pd.testing.assert_frame_equal(out, before)


def test_an_album_over_its_maximum_is_paired_down_to_it():
    resources = _resources()
    df = _merge_over_maximum(_photos(SIX_PAIRS), resources, max_total_spreads=4, logger=LOGGER)

    groups = df.groupby(KEYS)
    assert groups.ngroups == 4, "two merges, and no more than the maximum needs"
    sizes = sorted(len(g) for _, g in groups)
    assert sizes == [2, 2, 4, 4]
    lut = resources.look_up_table
    assert sum(lut.spreads_for(k, len(g)) for k, g in groups) == 4, \
        "a merged pair is pinned at one spread, not two by its class value"


def test_a_merged_pair_keeps_a_class_it_already_had():
    df = _photos([('bride', 3), ('other|0', 2), ('groom', 3)])

    df = _merge_over_maximum(df, _resources(), max_total_spreads=2, logger=LOGGER)

    assert set(df['cluster_context']) <= {'bride', 'other|0', 'groom'}
    # The photos keep their own content for the layout's scoring.
    assert sorted(df['original_context'].unique()) == ['bride', 'groom', 'other']


def test_no_pair_without_a_layout_for_its_size():
    # Only 2-box layouts: two 2-photo groups would need a 4-box one.
    df = _photos(SIX_PAIRS)

    out = _merge_over_maximum(df, _resources(box_counts=(2,)), max_total_spreads=4, logger=LOGGER)

    assert out.groupby(KEYS).ngroups == 6


def test_groups_in_different_time_clusters_are_not_paired():
    df = _photos([('other|0', 2), ('other|1', 2)])
    df.loc[df['cluster_context'] == 'other|1', 'time_cluster'] = 1

    out = _merge_over_maximum(df, _resources(), max_total_spreads=1, logger=LOGGER)

    assert out.groupby(KEYS).ngroups == 2
