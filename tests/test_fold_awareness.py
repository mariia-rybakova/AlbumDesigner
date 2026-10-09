"""The album's fold does not run down a person's face.

A box whose x-range straddles 0.5 prints across the fold. `files/designs.json`
carries 13826 of them over 10813 designs; 8986 are a full-spread landscape with
the fold dead centre, which is where two people standing together are.

Only the opening and closing pages need this: a box crosses the fold only in a
design holding one photo, and a content spread with one photo is a bug of its
own. See `src/spreads_layout/fold.py`.

    python -m pytest tests/test_fold_awareness.py -v
"""

import types

import pandas as pd
import pytest

from src.request_processing import customize_box
from src.smart_cropping import placement_window
from src.spreads_layout import fold
from utils.configs import CONFIGS
from utils.layouts_tools import generate_layouts_df, get_layouts_data


def face(x1, x2, blur=300.0, y1=0.2, y2=0.4):
    return types.SimpleNamespace(bbox=types.SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2),
                                 blurLevel=blur)


def box(x, width, height=1.0, orientation='landscape', y=0.0):
    return {'x': x, 'y': y, 'width': width, 'height': height, 'orientation': orientation}


# ---------------------------------------------------------------- geometry

def test_a_full_spread_box_puts_the_fold_in_its_middle():
    assert fold.fold_fraction(box(0.0, 1.0)) == pytest.approx(0.5)


def test_a_box_that_ends_at_the_fold_does_not_cross_it():
    """A full page is flush with the fold, not cut by it."""
    assert fold.fold_fraction(box(0.0, 0.5)) is None
    assert fold.fold_fraction(box(0.5, 0.5)) is None


def test_design_data_a_hair_over_the_fold_still_does_not_cross():
    assert fold.fold_fraction(box(0.0, 0.502)) is None


def test_a_page_and_a_lip_puts_the_fold_near_one_edge():
    """The f~0.9 family: the left page plus a strip of the right."""
    assert fold.fold_fraction(box(0.2, 0.35)) == pytest.approx(0.857, abs=0.001)


def test_mirroring_moves_the_fold_to_the_other_side_of_the_photo():
    """x -> 1 - x - w, which is what `is_mirrored` layouts carry."""
    original, mirrored = box(0.2, 0.35), box(1 - 0.2 - 0.35, 0.35)

    assert fold.fold_fraction(original) == pytest.approx(1 - fold.fold_fraction(mirrored))


# ------------------------------------------------------------------- faces

def test_a_not_a_face_detection_is_not_a_face():
    """Face-Recognition writes blurLevel -1 for what it judged not to be one."""
    assert fold.face_spans([face(0.4, 0.6, blur=-1.0)]) == ()


def test_the_widest_face_counts_most():
    spans = fold.face_spans([face(0.1, 0.3), face(0.6, 0.65)])

    assert [round(weight, 3) for _, _, weight in spans] == [1.0, 0.25]


def test_a_fold_down_the_middle_of_a_face_costs_the_most():
    assert fold.fold_cost(fold.face_spans([face(0.4, 0.6)]), 0.5) == pytest.approx(1.0)


def test_a_fold_clear_of_every_face_costs_nothing():
    assert fold.fold_cost(fold.face_spans([face(0.4, 0.6)]), 0.8) == 0.0


def test_a_fold_grazing_a_face_costs_less_than_one_through_it():
    spans = fold.face_spans([face(0.4, 0.6)])

    assert 0 < fold.fold_cost(spans, 0.57) < fold.fold_cost(spans, 0.5)


def test_a_fold_through_two_faces_is_worse_than_through_one():
    one = fold.face_spans([face(0.45, 0.55)])
    two = fold.face_spans([face(0.42, 0.52), face(0.48, 0.58)])

    assert fold.fold_cost(two, 0.5) > fold.fold_cost(one, 0.5)


# ------------------------------------------- the gutter is a width of paper

def test_the_gutter_does_not_shrink_with_the_face():
    """53316807 comp 10: a face ending 0.009 of the image from the fold scored
    exactly zero, because the only margin was a share of its own half-width."""
    small = fold.face_spans([face(0.449, 0.491)])
    window = placement_window(None, None, 1.5, 2.0)
    gutter = fold.gutter_in_image(box(0.0, 1.0), window)

    assert fold.fold_cost(small, 0.5) == 0.0, "what it used to score"
    assert fold.fold_cost(small, 0.5, gutter=gutter) > 0


def test_a_strip_of_paper_is_wider_inside_a_narrow_box():
    """The same gutter covers more of the photo when the box shows less page."""
    window = placement_window(None, None, 1.5, 1.4)

    assert fold.gutter_in_image(box(0.3, 0.4), window) > fold.gutter_in_image(box(0.0, 1.0), window)


def test_a_face_well_clear_of_the_fold_still_costs_nothing():
    far = fold.face_spans([face(0.10, 0.20)])
    window = placement_window(None, None, 1.5, 2.0)

    assert fold.fold_cost(far, 0.5, gutter=fold.gutter_in_image(box(0.0, 1.0), window)) == 0.0


# ------------------------------------------------------------- the estimate

def test_the_estimate_is_the_crop_the_placement_makes():
    """The claim the cover choice rests on: the fold is predicted with the same
    function that positions the crop."""
    crossing = box(0.0, 0.7, 0.5)
    faces = [face(0.66, 0.77)]
    row = pd.Series({'image_as': 1.5, 'faces_info': faces, 'bodies_info': None})

    assert fold.estimate_window(1.5, crossing, 2.0, faces, None) == customize_box(
        row, crossing, album_ar=2)


# ------------------------------------------------ the opening and closing pages

class _Layouts:
    """The one attribute `pick_cover_layout` reads off a layouts frame."""

    def __init__(self, rows):
        self.loc = rows


COVER_ROWS = {
    0: {'id': 900, 'left_box_ids': [1], 'right_box_ids': []},   # crosses, fold at 0.5
    1: {'id': 901, 'left_box_ids': [1], 'right_box_ids': []},   # crosses, fold at 0.86
    2: {'id': 902, 'left_box_ids': [1], 'right_box_ids': []},   # a whole page, no fold
}
COVER_BOXES = {
    (900, 1): dict(box(0.0, 1.0), area=2.0),
    (901, 1): dict(box(0.2, 0.35), area=0.7),
    (902, 1): dict(box(0.0, 0.5), area=1.0),
}


def cover_row(faces):
    return pd.Series({'image_as': 1.5, 'faces_info': faces, 'bodies_info': [],
                      'background_centroid': None})


def test_the_opening_page_avoids_a_fold_through_the_face():
    """53316807 opened on a face 0.005 of the image from the fold, because
    `_find_single_box_layout` returns candidates and the caller took the first."""
    chosen = fold.pick_cover_layout([0, 1, 2], _Layouts(COVER_ROWS), COVER_BOXES,
                                    cover_row([face(0.44, 0.56)]), album_ar=2.0)

    assert chosen != 0


def test_a_cover_candidate_that_crosses_no_fold_is_good_enough():
    chosen = fold.pick_cover_layout([0, 2], _Layouts(COVER_ROWS), COVER_BOXES,
                                    cover_row([face(0.44, 0.56)]), album_ar=2.0)

    assert chosen == 2


def test_the_covers_keep_their_own_order_when_the_fold_is_clear():
    """`_find_single_box_layout` prefers large boxes; that preference stands
    wherever the fold is not the deciding question."""
    assert fold.pick_cover_layout([0, 1, 2], _Layouts(COVER_ROWS), COVER_BOXES,
                                  cover_row([face(0.05, 0.15)]), album_ar=2.0) == 0


def test_a_cover_photo_with_no_face_keeps_the_first_candidate():
    assert fold.pick_cover_layout([0, 1, 2], _Layouts(COVER_ROWS), COVER_BOXES,
                                  cover_row([]), album_ar=2.0) == 0


def test_no_cover_candidates_is_handled():
    assert fold.pick_cover_layout([], _Layouts(COVER_ROWS), COVER_BOXES,
                                  cover_row([face(0.44, 0.56)]), album_ar=2.0) is None


def test_the_whole_thing_can_be_turned_off(monkeypatch):
    monkeypatch.setitem(CONFIGS['fold_awareness'], 'enabled', False)

    assert fold.pick_cover_layout([0, 1, 2], _Layouts(COVER_ROWS), COVER_BOXES,
                                  cover_row([face(0.44, 0.56)]), album_ar=2.0) == 0


# ------------------------------------------------- a design out of the catalogue

#: Design 4518 of `files/designs.json`, verbatim: box 14976 covers the left page
#: and runs 0.058 past the fold, the "page and a lip" family. 14979 is its text
#: box, which `generate_layouts_df` drops.
DESIGN_4518 = {'4518': {'boxes': [
    {'id': 14976, 'x': 0.09466666666666666, 'y': 0.10208333333333333,
     'width': 0.4583333333333333, 'height': 0.7179166666666666,
     'layer': 10, 'type': 0, 'tag': None},
    {'id': 14977, 'x': 0.544, 'y': 0.55375,
     'width': 0.35, 'height': 0.34875, 'layer': 20, 'type': 0, 'tag': None},
    {'id': 14978, 'x': 0.592, 'y': 0.11708333333333333,
     'width': 0.2786666666666667, 'height': 0.43666666666666665,
     'layer': 30, 'type': 0, 'tag': None},
    {'id': 14979, 'x': 0.09766666666666667, 'y': 0.8354166666666667,
     'width': 0.434, 'height': 0.06125, 'layer': 2013, 'type': 10, 'tag': None},
]}}


def test_a_real_design_carries_its_fold_through_to_the_box_data():
    """`box_id2data` is what `pick_cover_layout` reads, and it is keyed by the
    signed design id -- so a mirrored row carries the mirrored geometry."""
    layouts = generate_layouts_df(DESIGN_4518, [4518], do_mirror=True)
    _, box_id2data = get_layouts_data(layouts, None, None)

    def fractions(design_id):
        row = layouts[layouts['id'] == design_id].iloc[0]
        return {b: fold.fold_fraction(box_id2data[(design_id, b)])
                for b in list(row['left_box_ids']) + list(row['right_box_ids'])}

    original, mirrored = fractions(4518), fractions(-4518)

    assert original[14976] == pytest.approx(0.884, abs=0.001)
    assert mirrored[14976] == pytest.approx(1 - original[14976])
    assert original[14977] is None and original[14978] is None, "the others stay on their page"
