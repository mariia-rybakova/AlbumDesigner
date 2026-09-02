"""Tests for ``enrich.duplicate_shots``.

The rule is a judgement about the gallery rather than about a pair of photos,
so most of what matters here is when it declines to act. See
`src/pipeline/enrich/dedupe.py` for the measurements behind that.

    python -m pytest tests/test_dedupe.py -v
"""

from __future__ import annotations

import logging
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import ENRICH, AlbumContext, Col  # noqa: E402
from src.pipeline.contracts import GalleryFacts  # noqa: E402
from src.pipeline.enrich.dedupe import shot_groups  # noqa: E402
from src.pipeline.registry import get  # noqa: E402
from utils.configs import CONFIGS  # noqa: E402

FIRST_SECOND = 1_786_188_000


def quiet():
    logger = logging.getLogger("dedupe-test")
    logger.addHandler(logging.NullHandler())
    return logger


def gallery(rows):
    """``rows`` of ``(image_id, capture_second, aspect, rank, colour)``."""
    return pd.DataFrame([
        {Col.IMAGE_ID: image_id, Col.IMAGE_TIME: second, Col.IMAGE_AS: aspect,
         Col.IMAGE_ORDER: float(rank), Col.IMAGE_COLOR: colour}
        for image_id, second, aspect, rank, colour in rows
    ])


def uploaded_twice(n=20, aspect=1.5, base=1000, offset=4000, first=FIRST_SECOND):
    """What a photographer's second, black-and-white upload looks like.

    Every shot appears twice: the colour copy, and a greyscale copy at a
    constant photo-id offset sharing its capture second. ``base`` moves the
    whole set out of the way of ids a test adds itself.
    """
    rows = []
    for i in range(n):
        second = first + i * 30
        rows.append((base + i, second, aspect, i, 1))
        rows.append((base + offset + i, second, aspect, i + 100, 0))
    return gallery(rows)


#: A duplicated backdrop for tests that add a couple of photos of their own,
#: so the gallery reads as duplicated overall. Its ids *and* its capture
#: seconds sit well clear of the test's own, or they would join its group.
def backdrop(n=10):
    return uploaded_twice(n=n, base=90_000, first=FIRST_SECOND + 100_000)


def bursts(n=20, aspect=1.5):
    """An ordinary gallery: a few genuinely different frames in one second."""
    rows = [(1000 + i, FIRST_SECOND + i * 30, aspect, i, 1) for i in range(n)]
    # three same-second pairs -- a camera at three frames a second
    rows += [(2000 + i, FIRST_SECOND + i * 30, aspect, n + i, 1) for i in range(3)]
    return gallery(rows)


def run(photos):
    context = AlbumContext(logger=quiet(), photos=photos,
                           facts=GalleryFacts(is_wedding=True))
    return get("enrich.duplicate_shots")()(context)


def kept(context):
    return sorted(context.photos[Col.IMAGE_ID])


# -- composition -----------------------------------------------------------


def test_runs_before_anything_counts_the_gallery():
    """Left until selection, the budget sizes the album against a supply twice
    as large as it really is and every category then runs out."""
    assert list(ENRICH)[0] == "enrich.duplicate_shots"


# -- a gallery uploaded twice ----------------------------------------------


def test_one_copy_of_each_shot_survives():
    context = run(uploaded_twice(n=20))

    assert len(context.photos) == 20, "twenty shots, uploaded twice"


def test_the_best_ranked_copy_is_the_one_kept():
    """`image_order` is `selectionOrder`, where 0 is best, so the content
    model chooses -- not the upload order and not the treatment."""
    photos = gallery([
        (1000, FIRST_SECOND, 1.5, 90, 1),   # colour, poorly ranked
        (5000, FIRST_SECOND, 1.5, 3, 0),    # greyscale, well ranked
    ])

    assert kept(run(photos)) == [5000]


def test_a_toned_copy_is_caught_too():
    """The point of not testing the colour flag: a blue- or brown-toned copy is
    still classified as colour, so a rule keyed on the flag would miss it."""
    photos = gallery([
        (1000, FIRST_SECOND, 1.5, 1, 1),
        (5000, FIRST_SECOND, 1.5, 2, 1),    # same flag, different treatment
    ])
    photos = pd.concat([photos, backdrop()], ignore_index=True)

    context = run(photos)

    assert 5000 not in kept(context)
    assert 1000 in kept(context)


def test_the_gallery_is_left_alone_when_nothing_is_duplicated():
    photos = gallery([(1000 + i, FIRST_SECOND + i * 30, 1.5, i, 1) for i in range(12)])

    assert len(run(photos).photos) == 12


# -- when it must decline --------------------------------------------------


def test_a_few_same_second_bursts_are_not_a_duplicated_gallery():
    """The reason this is a gallery-level judgement. Applied pair by pair, the
    same-second test wanted to drop 46, 32 and 11 real frames from the three
    ordinary validation galleries."""
    photos = bursts(n=20)

    assert len(run(photos).photos) == len(photos), "burst frames are different photos"


def test_unusable_timestamps_drop_nothing():
    """Two validation galleries carry 2 distinct `image_time` values across
    500+ photos. Everything lands in one enormous group, which is ignored."""
    photos = gallery([(1000 + i, FIRST_SECOND, 1.5, i, 1) for i in range(40)])

    assert len(run(photos).photos) == 40


def test_a_missing_capture_time_is_not_a_shot_identity():
    """`dateTaken <= 0` is the protobuf's way of saying the EXIF had nothing,
    and is treated as absent everywhere else too."""
    photos = gallery([(1000 + i, 0, 1.5, i, 1) for i in range(6)])
    photos = pd.concat([photos, backdrop()], ignore_index=True)

    context = run(photos)

    assert all(i in kept(context) for i in range(1000, 1006))


def test_different_aspect_ratios_are_different_shots():
    """Insurance against a sub-second burst: a differently framed photo in the
    same second is its own shot."""
    photos = gallery([
        (1000, FIRST_SECOND, 1.5, 1, 1),
        (5000, FIRST_SECOND, 0.66, 2, 1),   # portrait, same second
    ])
    photos = pd.concat([photos, backdrop()], ignore_index=True)

    context = run(photos)

    assert 1000 in kept(context) and 5000 in kept(context)


def test_switching_it_off_is_a_no_op():
    original = CONFIGS['near_duplicates']
    CONFIGS['near_duplicates'] = {**original, 'enabled': False}
    try:
        context = run(uploaded_twice(n=20))
    finally:
        CONFIGS['near_duplicates'] = original

    assert len(context.photos) == 40


def test_the_share_threshold_is_what_decides():
    photos = bursts(n=20)

    original = CONFIGS['near_duplicates']
    CONFIGS['near_duplicates'] = {**original, 'min_gallery_share': 0.01}
    try:
        eager = run(photos)
    finally:
        CONFIGS['near_duplicates'] = original

    assert len(eager.photos) < len(photos), "a low share makes it act"
    assert len(run(photos).photos) == len(photos), "the default does not"


# -- the grouping helper ---------------------------------------------------


def test_shot_groups_is_empty_for_an_ordinary_gallery():
    assert shot_groups(bursts(n=20), quiet()) == {}


def test_shot_groups_finds_every_pair_in_a_duplicated_gallery():
    groups = shot_groups(uploaded_twice(n=15), quiet())

    assert len(groups) == 15
    assert all(len(members) == 2 for members in groups.values())


def test_shot_groups_tolerates_a_frame_with_no_columns_to_judge():
    assert shot_groups(pd.DataFrame({Col.IMAGE_ID: [1, 2]}), quiet()) == {}
    assert shot_groups(pd.DataFrame(), quiet()) == {}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
