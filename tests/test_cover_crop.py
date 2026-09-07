"""Tests for the cover crop.

`customize_box` centres its crop window blind. On a 1.96:1 cover box a
portrait keeps 34% of its height, so the centred band is y=0.33 to 0.67 --
below the faces of a standing couple, which sit in the top third. The album
for 53507032 opened on a frame cropped to two chins.

Measured on that gallery's 48 couple candidates: the centre crop keeps the
faces in 18 of them, and a window of the *same height*, merely repositioned,
contains them in 47. The height was never the problem.

    python -m pytest tests/test_cover_crop.py -v
"""

from __future__ import annotations

import os
import sys
import types

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.request_processing import box_target_ar, cover_box, customize_box  # noqa: E402
from src.smart_cropping import face_aware_crop  # noqa: E402

#: The cover box as measured from the rendered album: a full-width single box,
#: which against `album_ar` of 2 targets 1.96:1.
COVER_BOX = {'width': 0.98, 'height': 1.0, 'orientation': 'landscape'}
SQUARE_BOX = {'width': 0.5, 'height': 1.0, 'orientation': 'square'}

#: A 433x650 portrait, the shape of this gallery's couple frames.
PORTRAIT_AR = 433 / 650


def face(x1, y1, x2, y2):
    """The shape `process_cropping` reads: `face.bbox.x1` and friends."""
    return types.SimpleNamespace(bbox=types.SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2))


def photo(faces=None, ar=PORTRAIT_AR, **extra):
    row = {
        'image_as': ar,
        'faces_info': faces if faces is not None else [],
        'background_centroid': types.SimpleNamespace(x=0.5, y=0.4),
        'diameter': 0.5,
        'cropped_x': 0.0, 'cropped_y': 0.1, 'cropped_w': 1.0, 'cropped_h': 0.66,
    }
    row.update(extra)
    return pd.Series(row)


# -- the geometry the fix is about ----------------------------------------


def test_the_cover_box_is_wide_enough_to_decapitate_a_portrait():
    """Not a test of our code -- a check that the fixture reproduces the
    condition. A portrait in this box keeps about a third of its height."""
    target = box_target_ar(COVER_BOX, album_ar=2)
    keep = PORTRAIT_AR / target

    assert 1.9 < target < 2.0
    assert 0.30 < keep < 0.36


def test_the_centred_crop_lands_below_faces_in_the_top_third():
    """What `customize_box` does, stated as a fact so the fix has a baseline."""
    x, y, w, h = customize_box(photo(), COVER_BOX, album_ar=2)

    assert (x, w) == (0.0, 1.0)
    assert y == pytest.approx(0.33, abs=0.01)
    assert y + h == pytest.approx(0.67, abs=0.01)
    # Faces at 0.10-0.35 are almost entirely above that window.
    assert 0.35 > y, "the fixture no longer reproduces the fault"


# -- face_aware_crop -------------------------------------------------------


def test_it_moves_the_window_up_to_the_faces():
    faces = [face(0.20, 0.12, 0.45, 0.30), face(0.50, 0.15, 0.75, 0.33)]

    crop = face_aware_crop(photo(faces), box_target_ar(COVER_BOX, album_ar=2))

    assert crop is not None, "the face-aware path did not run"
    _, y, _, h = crop
    centred_y = customize_box(photo(faces), COVER_BOX, album_ar=2)[1]
    assert y < centred_y, f"window not raised: {y:.3f} vs centred {centred_y:.3f}"


def test_it_keeps_the_faces_inside_the_window():
    faces = [face(0.20, 0.12, 0.45, 0.30), face(0.50, 0.15, 0.75, 0.33)]

    _, y, _, h = face_aware_crop(photo(faces), box_target_ar(COVER_BOX, album_ar=2))

    top, bottom = min(f.bbox.y1 for f in faces), max(f.bbox.y2 for f in faces)
    assert y <= top + 0.02, f"crop starts at {y:.3f}, faces at {top:.3f}"
    assert y + h >= bottom - 0.02, f"crop ends at {y + h:.3f}, faces to {bottom:.3f}"


def test_no_faces_means_no_opinion():
    """Nothing to aim at, so the caller keeps its own behaviour."""
    assert face_aware_crop(photo([]), 1.96) is None


@pytest.mark.parametrize("missing", ['faces_info', 'background_centroid', 'diameter'])
def test_a_missing_input_declines_rather_than_raises(missing):
    """The cover is worth losing a better crop over, never the placement."""
    row = photo([face(0.2, 0.1, 0.4, 0.3)]).drop(labels=[missing])

    assert face_aware_crop(row, 1.96) is None


def test_a_broken_input_declines_rather_than_raises():
    row = photo([face(0.2, 0.1, 0.4, 0.3)], diameter='not a number')

    assert face_aware_crop(row, 1.96) is None


# -- cover_box wiring ------------------------------------------------------


def test_cover_box_prefers_the_face_aware_crop():
    faces = [face(0.20, 0.12, 0.45, 0.30), face(0.50, 0.15, 0.75, 0.33)]

    mine = cover_box(photo(faces), COVER_BOX, album_ar=2)
    centred = customize_box(photo(faces), COVER_BOX, album_ar=2)

    assert mine != centred


def test_cover_box_falls_back_when_there_are_no_faces():
    row = photo([])

    assert cover_box(row, COVER_BOX, album_ar=2) == customize_box(row, COVER_BOX, album_ar=2)


def test_a_square_box_is_left_to_the_existing_crop():
    """Square boxes already use the frame's `cropped_*`, which is a face-aware
    square crop -- `process_crop_images` computes it with box_aspect_ratio=1."""
    row = photo([face(0.2, 0.1, 0.4, 0.3)])

    assert cover_box(row, SQUARE_BOX, album_ar=2) == (0.0, 0.1, 1.0, 0.66)


def test_the_rest_of_the_album_is_untouched():
    """Covers only. `customize_box` still centres, because changing it would
    move the crops on every spread of every album."""
    faces = [face(0.20, 0.12, 0.45, 0.30)]

    _, y, _, h = customize_box(photo(faces), COVER_BOX, album_ar=2)

    assert y == pytest.approx(0.33, abs=0.01)


def test_the_crop_is_plain_floats_so_the_album_doc_serialises():
    """`push_report_msg` calls `json.dumps` on the album doc with no
    `default=` handler. `np.float64` happens to subclass `float` and would
    survive, but nothing here should rely on that.
    """
    import json

    faces = [face(0.20, 0.12, 0.45, 0.30)]
    crop = face_aware_crop(photo(faces), box_target_ar(COVER_BOX, album_ar=2))

    assert all(type(v) is float for v in crop), [type(v).__name__ for v in crop]
    json.dumps(dict(zip(('cropX', 'cropY', 'cropWidth', 'cropHeight'), crop)))


def crop_aspect(crop, image_ar):
    """The crop's own width/height in pixels.

    `w` and `h` are fractions of the image, so the pixel aspect is
    ``(w / h) * (image width / image height)``.
    """
    _, _, w, h = crop
    return (w / h) * image_ar


def test_the_crop_has_the_box_aspect_ratio():
    """The property the whole fix turns on, and the one my first pass at these
    tests missed: passing `box_aspect_ratio=1` (what `process_crop_images`
    does) still moves the window onto the faces, so every other assertion
    here passed with it -- but it yields a *square* crop, which is the wrong
    shape for a 1.96:1 box.
    """
    target = box_target_ar(COVER_BOX, album_ar=2)
    faces = [face(0.20, 0.12, 0.45, 0.30), face(0.50, 0.15, 0.75, 0.33)]

    crop = face_aware_crop(photo(faces), target)

    assert crop_aspect(crop, PORTRAIT_AR) == pytest.approx(target, rel=0.05), (
        f"crop is {crop_aspect(crop, PORTRAIT_AR):.2f}:1, box wants {target:.2f}:1")


def test_the_crop_matches_what_the_centred_one_would_have_been_in_shape():
    """Same shape, different position -- the height was never the problem."""
    target = box_target_ar(COVER_BOX, album_ar=2)
    faces = [face(0.20, 0.12, 0.45, 0.30)]

    mine = face_aware_crop(photo(faces), target)
    centred = customize_box(photo(faces), COVER_BOX, album_ar=2)

    assert mine[3] == pytest.approx(centred[3], rel=0.05), "crop height changed"
    assert mine[1] != pytest.approx(centred[1], abs=0.01), "crop did not move"
