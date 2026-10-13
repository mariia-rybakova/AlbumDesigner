"""A non-wedding album keeps the groups that fit its design, most people first.

44090624 (2026-10-12) reached the non-wedding route with every photo's
`image_class` at -1, split into about 140 people groups of one spread each, and
was sent with 146 compositions against a product maximum of 41.

    python -m pytest tests/test_non_wedding_cap.py -v
"""
from types import SimpleNamespace

from src.album_processing import (_persons_in, cap_non_wedding_groups,
                                  trim_non_wedding_results)


def test_the_persons_count_is_read_off_the_group_key():
    assert _persons_in(('10_pple_10_1_26_30_46',)) == 10
    assert _persons_in(('1_person_10',)) == 1
    assert _persons_in(('No PEOPLE',)) == 0
    assert _persons_in(('Invalid_numbers',)) == 0


def test_an_album_that_fits_keeps_every_group():
    groups = {('2_pple_1',): 3, ('1_person_4',): 2}
    assert cap_non_wedding_groups(groups, {g: 1 for g in groups}, 38) == list(groups)


def test_the_groups_with_most_people_are_kept():
    groups = {('1_person_4',): 2, ('No PEOPLE',): 9, ('5_pple_1',): 3,
              ('2_pple_1',): 2, ('3_pple_7',): 1}
    kept = cap_non_wedding_groups(groups, {g: 1 for g in groups}, 3)
    assert set(kept) == {('5_pple_1',), ('3_pple_7',), ('2_pple_1',)}
    assert kept == [g for g in groups if g in kept], "kept in the album's own order"


def test_a_group_too_big_for_what_is_left_lets_a_smaller_one_in():
    groups = {('4_pple_1',): 30, ('3_pple_2',): 2, ('2_pple_3',): 2}
    spreads = {('4_pple_1',): 3, ('3_pple_2',): 1, ('2_pple_3',): 1}
    assert set(cap_non_wedding_groups(groups, spreads, 2)) == {('3_pple_2',), ('2_pple_3',)}


def test_the_logged_44090624_album_comes_down_to_the_design():
    """140 one-spread groups against 41 pages: 38 content spreads."""
    groups = {(f'{n % 11 + 1}_pple_{n}',): 1 for n in range(140)}
    kept = cap_non_wedding_groups(groups, {g: 1 for g in groups}, 41 - 3)
    assert len(kept) == 38
    assert min(_persons_in(g) for g in kept) >= max(
        _persons_in(g) for g in groups if g not in kept)


def _result(n_spreads):
    return {'g': SimpleNamespace(spreads=[object()] * n_spreads)}


def test_a_layout_over_the_cap_loses_its_fewest_people_groups():
    groups = {('3_pple_1',): 3, ('2_pple_1',): 2, ('1_person_1',): 1}
    named = [(g, _result(2)) for g in groups]
    kept = trim_non_wedding_results(named, groups, 4)
    assert [g for g, _ in kept] == [('3_pple_1',), ('2_pple_1',)]


def test_a_layout_within_the_cap_is_left_alone():
    groups = {('3_pple_1',): 3, ('1_person_1',): 1}
    named = [(g, _result(1)) for g in groups]
    assert trim_non_wedding_results(named, groups, 4) == named
