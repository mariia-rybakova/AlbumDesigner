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


# -- when the faces do not all fit ------------------------------------------
#
# The opening cover of 53840120 (2026-09-27T05:40 dev run, photo 12497678983),
# as `cover_box` received it. The search pinned the window to the top of the
# frame to keep a 0.1-wide detection in the corner, one the face service had
# itself marked not-a-face (blurLevel -1), and cut the main face through the
# mouth: 58% of its height kept.

OPENING_AR = 0.7507692575454712
OPENING_TARGET_AR = 1.9620315074640127


def rated_face(x1, y1, x2, y2, blur):
    f = face(x1, y1, x2, y2)
    f.blurLevel = blur
    return f


def opening_photo(faces):
    return photo(faces, ar=OPENING_AR,
                 background_centroid=types.SimpleNamespace(x=0.404, y=0.664), diameter=0.780)


MAIN = rated_face(0.277, 0.241, 0.490, 0.484, 626.9)
CORNER = rated_face(0.028, 0.067, 0.139, 0.192, -1.0)


def kept(crop, f):
    """Share of the face's height inside the crop."""
    _, y, _, h = crop
    b = f.bbox
    return max(0.0, min(b.y2, y + h) - max(b.y1, y)) / (b.y2 - b.y1)


def test_faces_that_do_not_fit_one_window_give_way_to_the_main_face():
    crop = face_aware_crop(opening_photo([MAIN, CORNER]), OPENING_TARGET_AR)

    assert kept(crop, MAIN) == pytest.approx(1.0), f"main face kept {kept(crop, MAIN):.0%}"
    assert crop[3] == pytest.approx(OPENING_AR / OPENING_TARGET_AR, rel=0.01), "crop height changed"


def test_the_main_face_is_centred():
    _, y, _, h = face_aware_crop(opening_photo([MAIN, CORNER]), OPENING_TARGET_AR)

    assert y + h / 2 == pytest.approx((MAIN.bbox.y1 + MAIN.bbox.y2) / 2, abs=0.01)


def test_a_large_blurred_face_does_not_lead():
    """The largest face is out of focus; the sharp one is the subject."""
    blurred = rated_face(0.05, 0.55, 0.60, 0.95, 8.0)
    sharp = rated_face(0.40, 0.10, 0.60, 0.30, 400.0)

    crop = face_aware_crop(opening_photo([blurred, sharp]), OPENING_TARGET_AR)

    assert kept(crop, sharp) == pytest.approx(1.0)


def test_with_no_sharp_face_the_search_decides():
    """Nothing worth centring on, so the old behaviour stands."""
    import src.smart_cropping as sc

    faces = [rated_face(0.277, 0.241, 0.490, 0.484, 5.0), rated_face(0.028, 0.067, 0.139, 0.192, 3.0)]
    row = opening_photo(faces)
    searched = sc.process_cropping(OPENING_AR, faces, row['background_centroid'], 0.780, OPENING_TARGET_AR)

    assert face_aware_crop(row, OPENING_TARGET_AR) == pytest.approx(tuple(searched))


def test_faces_that_fit_together_are_all_kept():
    """Two faces side by side fit one window, and both stay in it."""
    left = rated_face(0.20, 0.30, 0.40, 0.45, 300.0)
    right = rated_face(0.55, 0.35, 0.75, 0.50, 300.0)

    crop = face_aware_crop(opening_photo([left, right]), OPENING_TARGET_AR)

    assert kept(crop, left) == pytest.approx(1.0) and kept(crop, right) == pytest.approx(1.0)


def test_a_detection_marked_not_a_face_is_ignored():
    """The closing cover of the same run: a sharp face and a -1 detection in the
    corner. The crop is what the sharp face alone would get."""
    sharp = rated_face(0.460, 0.432, 0.553, 0.594, 195.6)
    corner = rated_face(0.876, 0.033, 0.950, 0.179, -1.0)
    row = lambda faces: photo(faces, ar=1.3319672346115112,
                              background_centroid=types.SimpleNamespace(x=0.490, y=0.733), diameter=0.624)

    assert face_aware_crop(row([sharp, corner]), OPENING_TARGET_AR) == \
        face_aware_crop(row([sharp]), OPENING_TARGET_AR)


def test_only_non_faces_means_no_opinion():
    assert face_aware_crop(opening_photo([CORNER]), OPENING_TARGET_AR) is None
