"""An abandoned crop must not be readable by the album that comes after it.

On 2026-09-18 two dev pods began shipping albums with blank first and last
pages. The trigger was one ordinary stage exception -- `_compute_initial_spreads`
on a `None` lookup table -- raised between starting a message's crop subprocess
and reading its result. `ProcessStage` held a single `mp.Queue` for the life of
the process, so the orphaned frame stayed on it and every later message read the
*previous* gallery's crops.

Nothing errored. The body left-joins its crops and fills the misses with a
centred window, so the spreads looked right; the covers inner-join, so their row
vanished and `_fillable_cover_boxes` left the box empty. Both pods stayed wrong
for four days, until a deployment restarted them.

These pin the ownership that makes that impossible: the queue belongs to the
crop, and `close()` ends it.
"""
import multiprocessing as mp
from types import SimpleNamespace

import pandas as pd
import pytest

from main import _CropJob


def frame(ids):
    """The columns `process_crop_images` reads, and nothing else.

    `background_centroid` has to carry `.x`/`.y`: with no faces and the square
    box the cropper is called with, that centroid is the whole crop. A `None`
    there kills the worker and the parent then waits out its full 200s.
    """
    return pd.DataFrame({
        'image_id': ids,
        'faces_info': [[] for _ in ids],
        'background_centroid': [SimpleNamespace(x=0.5, y=0.5) for _ in ids],
        'diameter': [1.0 for _ in ids],
        'image_as': [1.5 for _ in ids],
    })


@pytest.mark.skipif(mp.get_start_method() not in ('spawn', 'fork'),
                    reason='needs a working multiprocessing start method')
def test_result_is_the_frame_that_was_sent():
    job = _CropJob(frame([1, 2, 3]))
    try:
        cropped = job.result(timeout=30)
    finally:
        job.close()
    assert sorted(cropped['image_id']) == [1, 2, 3]
    assert set(cropped.columns) >= {'cropped_x', 'cropped_y', 'cropped_w', 'cropped_h'}


def test_an_abandoned_crop_cannot_be_read_by_the_next_one():
    """The regression itself: album A raises before reading, album B follows."""
    abandoned = _CropJob(frame([10, 11]))
    abandoned.close()          # what the stage's `finally` now does

    following = _CropJob(frame([20, 21]))
    try:
        cropped = following.result(timeout=30)
    finally:
        following.close()

    # Under the shared queue this came back as [10, 11] -- the previous
    # gallery's crops, whose ids match nothing in this album.
    assert sorted(cropped['image_id']) == [20, 21]


def test_each_job_owns_its_own_queue():
    first, second = _CropJob(frame([1])), _CropJob(frame([2]))
    try:
        assert first.queue is not second.queue
    finally:
        first.close()
        second.close()


def test_close_is_safe_after_a_result_was_read():
    job = _CropJob(frame([7]))
    cropped = job.result(timeout=30)
    job.close()
    assert list(cropped['image_id']) == [7]
