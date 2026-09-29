"""One person is framed by the head and body, not centred on the face.

49995684's seated bride (rows 1-2 of the crop review, photos 11547935829 and
11547935839): the face-centred crop left a band of wall above her head nearly
twice her face's height and cut her at the chest; the human crop left about one
face-height above her head and ran down over her body to her hands. The numbers
below are those photos', rounded.

    python -m pytest tests/test_single_subject_crop.py -v
"""

import types

import pandas as pd
import pytest

from src.request_processing import customize_box
from src.smart_cropping import face_aware_crop, process_cropping, single_subject_window
from utils.configs import CONFIGS

PORTRAIT = 0.666
HEADROOM = CONFIGS['single_subject_crop']['headroom_faces']


def rect(x1, y1, x2, y2, **extra):
    return types.SimpleNamespace(bbox=types.SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2), **extra)


FACE = rect(0.443, 0.369, 0.596, 0.509, blurLevel=724.0)
BODY = rect(0.247, 0.335, 0.808, 0.998)
CENTROID = types.SimpleNamespace(x=0.531, y=0.75)


def top_of(window):
    return window[1]


def test_the_head_not_the_face_places_the_window():
    """In the review's 1.26:1 box, as the human crop: about 0.233 to 0.761."""
    x, y, w, h = single_subject_window([FACE], [BODY], PORTRAIT, 1.261)

    assert y == pytest.approx(FACE.bbox.y1 - HEADROOM * (FACE.bbox.y2 - FACE.bbox.y1), abs=1e-6)
    assert y == pytest.approx(0.229, abs=0.01)
    assert y + h > 0.75, "the window runs down over her body"


def test_the_pre_computed_square_crop_uses_it():
    """Route 1: `process_crop_images` asks for 1:1."""
    crop = process_cropping(PORTRAIT, [FACE], CENTROID, 0.658, 1, bodies=[BODY])

    assert top_of(crop) == pytest.approx(0.229, abs=0.01)
    assert crop[3] == pytest.approx(PORTRAIT, abs=1e-3)


def test_a_spread_box_that_is_not_square_uses_it():
    """Route 2: `customize_box` used to centre these blind."""
    row = pd.Series({'image_as': PORTRAIT, 'faces_info': [FACE], 'bodies_info': [BODY]})
    box = {'width': 0.113, 'height': 0.175, 'orientation': 'landscape'}

    x, y, w, h = customize_box(row, box, album_ar=1.962)

    assert y == pytest.approx(0.229, abs=0.01)


def test_the_covers_use_it():
    """Route 4: the face-aware crop of the opening and closing pages."""
    row = pd.Series({'image_as': PORTRAIT, 'faces_info': [FACE], 'bodies_info': [BODY],
                     'background_centroid': CENTROID, 'diameter': 0.658})

    assert top_of(face_aware_crop(row, 1.261)) == pytest.approx(0.229, abs=0.01)


def test_a_face_at_the_top_edge_keeps_the_top_edge():
    """Row 3 of the review: the groom's face at the top of the frame."""
    face = rect(0.3, 0.02, 0.6, 0.25, blurLevel=300.0)

    assert top_of(single_subject_window([face], [], PORTRAIT, 1.261)) == 0.0


def test_across_it_centres_on_the_body_and_keeps_the_face():
    """A landscape photo in a narrow box: centred on the body, face whole."""
    face = rect(0.62, 0.2, 0.70, 0.45, blurLevel=300.0)
    body = rect(0.40, 0.15, 0.80, 1.0)

    x, y, w, h = single_subject_window([face], [body], 1.5, 0.8)

    assert x <= face.bbox.x1 and face.bbox.x2 <= x + w
    assert x + w / 2 == pytest.approx(0.60, abs=0.02)


def test_two_faces_are_not_one_person():
    other = rect(0.1, 0.3, 0.2, 0.45, blurLevel=300.0)

    assert single_subject_window([FACE, other], [BODY], PORTRAIT, 1.0) is None


def test_a_not_a_face_detection_does_not_count():
    ghost = rect(0.1, 0.3, 0.2, 0.45, blurLevel=-1.0)

    assert single_subject_window([FACE, ghost], [BODY], PORTRAIT, 1.0) is not None


def test_a_second_large_body_makes_it_a_group():
    beside = rect(0.0, 0.3, 0.3, 0.99)

    assert single_subject_window([FACE], [BODY, beside], PORTRAIT, 1.0) is None


def test_a_small_figure_behind_does_not():
    behind = rect(0.02, 0.4, 0.08, 0.55)

    assert single_subject_window([FACE], [BODY, behind], PORTRAIT, 1.0) is not None


def test_without_bodies_the_face_alone_places_it():
    assert top_of(single_subject_window([FACE], None, PORTRAIT, 1.0)) == pytest.approx(0.229, abs=0.01)


def test_it_can_be_switched_off(monkeypatch):
    monkeypatch.setitem(CONFIGS['single_subject_crop'], 'enabled', False)

    assert single_subject_window([FACE], [BODY], PORTRAIT, 1.0) is None
