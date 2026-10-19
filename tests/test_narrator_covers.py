"""Tests for ``select.narrator_covers``: the narrator album's opening and closing.

Pure: a stand-in packed gallery, no policy, no checkpoint.

    python -m pytest tests/test_narrator_covers.py -v
"""

from __future__ import annotations

import logging
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.covers.choosing as kp  # noqa: E402
import src.covers.scoring as kp_scoring  # noqa: E402
from src.pipeline.contracts import Col  # noqa: E402
from src.pipeline.select import narrator_covers as nc  # noqa: E402

LOG = logging.getLogger('test_narrator_covers')
LOG.addHandler(logging.NullHandler())


def packed(n, cast=None):
    """What `pick_hero_photos` reads off a PackedGallery."""
    cast = cast or [frozenset({1, 2})] * n
    return types.SimpleNamespace(
        n=n, photo_ids=np.arange(100, 100 + n, dtype=np.int64),
        time_norm=np.linspace(0.0, 1.0, n).astype(np.float32),
        time_valid=np.ones(n, dtype=bool), cast=cast,
        n_persons=2, cast_multihot=np.ones((n, 2)), person_ids=np.array([1, 2]))


def frame(n, orientation=None, order=None):
    vectors = np.eye(n, 8, dtype=np.float32)
    return pd.DataFrame({
        Col.IMAGE_ID: list(range(100, 100 + n)),
        'image_orientation': orientation or ['portrait'] * n,
        Col.IMAGE_ORDER: order if order is not None else list(range(n)),
        Col.EMBEDDING: list(vectors),
    })


@pytest.fixture
def no_quality(monkeypatch):
    """Quality from the photos' own embeddings would make the test about them."""
    monkeypatch.setattr(kp_scoring, '_quality', lambda f, logger: np.zeros(len(f)))


def test_a_good_enough_landscape_opens_over_the_best_ranked_portrait(no_quality):
    """53952231: the narrator opened on a portrait with landscapes beside it."""
    n = 20
    orientation = ['portrait'] * n
    orientation[2] = 'landscape'
    photos = frame(n, orientation, order=list(range(n)))   # 100 is the best ranked

    first, last = nc._choose(packed(n), photos.set_index(Col.IMAGE_ID, drop=False), LOG)

    assert first == 2, "the landscape, ranked third, takes the opening"
    assert last >= 15, "the closing comes from the last quarter"


def test_the_people_the_album_is_about_still_count(no_quality):
    """The narrator's dominant-group overlap is the presence term."""
    n = 20
    cast = [frozenset({1, 2})] * n
    cast[0] = frozenset({9})                               # best ranked, but a stranger
    photos = frame(n, order=list(range(n)))

    first, _ = nc._choose(packed(n, cast), photos.set_index(Col.IMAGE_ID, drop=False), LOG)

    assert first == 1


def test_the_main_people_are_everyone_frequent_not_the_most_frequent_set():
    """54117560: mother+son in 70 frames, all three in 12 -- the father holds the camera."""
    cast = ([frozenset({1, 3})] * 70 + [frozenset({1})] * 53 + [frozenset({1, 2})] * 48
            + [frozenset({3})] * 46 + [frozenset({1, 2, 3})] * 12 + [frozenset({2})] * 10
            + [frozenset({7})] * 3 + [frozenset()] * 20)

    assert nc.main_people(packed(len(cast), cast)) == frozenset({1, 2, 3})


def test_nobody_frequent_enough_leaves_no_main_people():
    cast = [frozenset({i}) for i in range(10)]

    assert nc.main_people(packed(10, cast)) == frozenset()


def test_the_whole_family_takes_the_cover_over_a_pair_of_it(no_quality):
    n = 20
    cast = [frozenset({1, 3})] * 10 + [frozenset({1, 2})] * 6 + [frozenset({1, 2, 3})] * 4
    cast[0], cast[3] = frozenset({1, 3}), frozenset({1, 2, 3})
    photos = frame(n, order=list(range(n)))                # 100 is the best ranked

    first, _ = nc._choose(packed(n, cast), photos.set_index(Col.IMAGE_ID, drop=False), LOG)

    assert first == 3, "the frame of all three, though ranked fourth"


def test_the_whole_group_is_more_than_the_next_linear_step():
    """Five main people: two, three, four are equal steps; all five is more."""
    group = frozenset(range(1, 6))
    cast = [frozenset(range(1, k + 1)) for k in (2, 3, 4, 5)]
    pg = packed(4, cast)

    two, three, four, five = (nc._presence(pg, i, group) for i in range(4))

    assert three - two == pytest.approx(four - three)
    assert five - four > four - three


def test_a_group_of_one_gets_no_bonus():
    pg = packed(2, [frozenset({1}), frozenset({1, 2})])

    assert nc._presence(pg, 0, frozenset({1})) == pytest.approx(1.0)


def test_the_opening_window_widens_for_a_small_gallery():
    """44 photos: 40% of the timeline, not the narrator's quarter."""
    window = nc._window(packed(44), kp.opening_fraction(44), first=True)

    assert len(window) == 18


def test_a_failing_choice_keeps_the_narrators_own(monkeypatch):
    monkeypatch.setattr(nc, '_choose', lambda *a: (_ for _ in ()).throw(ValueError('no')))

    assert nc.choose_covers(packed(10), frame(10)) is None


def test_the_choice_works_without_a_logger(no_quality):
    """The cover scoring logs; a missing logger must not turn into a silent fallback."""
    assert nc.choose_covers(packed(20), frame(20)) is not None


def test_the_swap_moves_the_new_covers_out_of_the_body():
    """The policy composed around its own heroes; ours come out of its pages,
    a page left empty goes, and the narrator's heroes leave the album."""
    photos = frame(6)
    block = {'firstPagePhotoIds': [100], 'lastPagePhotoIds': [105],
             'spreads': [{'photoIds': [101, 102]}, {'photoIds': [103]}, {'photoIds': [104]}]}

    got = nc.apply_covers(block, (102, 104), photos, LOG)

    assert got['firstPagePhotoIds'] == [102] and got['lastPagePhotoIds'] == [104]
    assert got['spreads'] == [{'photoIds': [101]}, {'photoIds': [103]}]


def test_a_body_photo_repeating_a_cover_is_pruned(monkeypatch):
    monkeypatch.setitem(kp_scoring.settings(), 'cover_duplicate_similarity', 0.8)
    photos = frame(5)
    near_copy = photos.at[0, Col.EMBEDDING] + 0.05 * photos.at[1, Col.EMBEDDING]
    photos.at[3, Col.EMBEDDING] = near_copy                # the cover's burst neighbour
    block = {'firstPagePhotoIds': [100], 'lastPagePhotoIds': [104],
             'spreads': [{'photoIds': [101, 102]}, {'photoIds': [103]}]}

    got = nc.prune_cover_duplicates(block, photos, LOG)

    assert got['spreads'] == [{'photoIds': [101, 102]}], "the emptied spread goes too"
    assert got['firstPagePhotoIds'] == [100]
