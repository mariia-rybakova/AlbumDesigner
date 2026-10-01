"""The final spread pass: move a spread only when the clock cannot be argued with.

`sort_groups_by_time` orders whole groups by median time and spreads inside a
group by their first photo, so a group is an atomic block and no spread can pass
one from another group. A group that merely ends late is therefore placed late in
full, and its opening spread sits behind spreads shot after it.

`reorder_disjoint_spreads` relaxes exactly that, and only where the times leave
nothing to decide: every photo of the later spread taken before every photo of
the earlier one. Overlapping spreads keep the group sort's order, which is what
holds a group together.
"""

import logging

import pytest

from src.core.models import GroupProcessingResult, Spread
from src.core.photos import Photo
from utils.time_processing import reorder_disjoint_spreads


LOGGER = logging.getLogger('test')


def photo(general_time, photo_id=1):
    return Photo(id=photo_id, ar=1.5, color=True, rank=1.0, photo_class='c',
                 cluster_label=0, general_time=general_time, original_context='c')


def spread(*times):
    """A spread holding one photo per time, split across the two pages."""
    photos = [photo(t, photo_id=100 + i) for i, t in enumerate(times)]
    half = max(1, len(photos) // 2)
    return Spread(layout_id=0, left_photos=photos[:half], right_photos=photos[half:])


def block(group_id, *spreads):
    return {group_id: GroupProcessingResult(group_name=group_id, spreads=list(spreads))}


def order(groups_list):
    """Flat `(group_id, first photo time)` per spread, in final page order."""
    out = []
    for group_dict in groups_list:
        for group_id, result in group_dict.items():
            for sp in result.spreads:
                times = [p.general_time for p in sp.left_photos + sp.right_photos
                         if p.id != -1]
                out.append((group_id, min(times)))
    return out


def test_a_wholly_earlier_spread_moves_ahead_of_another_group():
    """The case the group sort cannot reach: B's opening spread predates A's."""
    groups = [block('A', spread(500, 600)), block('B', spread(100, 200), spread(900))]
    assert order(reorder_disjoint_spreads(groups, LOGGER)) == [
        ('B', 100), ('A', 500), ('B', 900)]


def test_overlapping_spreads_are_left_alone():
    """Touching by one second is enough to leave the group order standing."""
    groups = [block('A', spread(500, 900)), block('B', spread(100, 600))]
    assert order(reorder_disjoint_spreads(groups, LOGGER)) == [('A', 500), ('B', 100)]


def test_an_album_already_in_order_is_returned_untouched():
    groups = [block('A', spread(100, 200)), block('B', spread(300, 400))]
    assert reorder_disjoint_spreads(groups, LOGGER) is groups


def test_spreads_of_one_group_keep_their_order_when_nothing_moves():
    groups = [block('A', spread(100), spread(200), spread(300))]
    assert reorder_disjoint_spreads(groups, LOGGER) is groups


def test_a_group_the_pass_does_not_split_stays_one_result():
    """Contiguity is only given up where a spread actually crossed."""
    groups = [block('A', spread(500), spread(600)), block('B', spread(100))]
    result = reorder_disjoint_spreads(groups, LOGGER)
    assert order(result) == [('B', 100), ('A', 500), ('A', 600)]
    # B first, then A's two spreads merged back into a single result.
    assert [len(r.spreads) for d in result for r in d.values()] == [1, 2]


def test_a_fully_reversed_album_is_fully_reversed():
    """Termination, and the ordering is total when every pair is disjoint."""
    groups = [block(str(i), spread(t)) for i, t in enumerate([900, 700, 500, 300, 100])]
    assert [t for _, t in order(reorder_disjoint_spreads(groups, LOGGER))] == [
        100, 300, 500, 700, 900]


def test_the_padding_photo_is_ignored():
    """The layout search's dummy carries id -1 and a sentinel time."""
    padded = Spread(layout_id=0, left_photos=[photo(100, photo_id=1)],
                    right_photos=[photo(1000000, photo_id=-1)])
    groups = [block('A', spread(500)), block('B', padded)]
    assert order(reorder_disjoint_spreads(groups, LOGGER)) == [('B', 100), ('A', 500)]


def test_a_spread_with_no_usable_time_never_moves():
    timeless = Spread(layout_id=0, left_photos=[photo(1000000, photo_id=-1)],
                      right_photos=[])
    groups = [block('A', spread(500)), block('B', timeless)]
    assert reorder_disjoint_spreads(groups, LOGGER) is groups


def test_a_broken_structure_keeps_the_group_order():
    """Recording must never cost an album: the pass falls back, it does not raise."""
    groups = [block('A', spread(500)), {'B': 'not a result'}]
    assert reorder_disjoint_spreads(groups, LOGGER) is groups


@pytest.mark.parametrize('n_groups', [0, 1])
def test_too_few_spreads_to_compare(n_groups):
    groups = [block(str(i), spread(100 * i)) for i in range(n_groups)]
    assert reorder_disjoint_spreads(groups, LOGGER) is groups
