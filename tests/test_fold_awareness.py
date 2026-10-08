"""The album's fold does not run down a person's face.

A box whose x-range straddles 0.5 prints across the fold. `files/designs.json`
carries 13826 of them over 10813 designs; 8986 are a full-spread landscape with
the fold dead centre, which is where two people standing together are.

    python -m pytest tests/test_fold_awareness.py -v
"""

import types

import pandas as pd
import pytest

from src.core.photos import Photo
from src.request_processing import customize_box
from src.smart_cropping import placement_window
from src.spreads_layout import fold
from src.spreads_layout.group_layouts import GroupSingleLayout
from src.spreads_layout.main import select_best_layout_for_subgroup
from src.spreads_layout.spreads.spread import SingleSpreadLayout
from utils.configs import CONFIGS
from utils.layouts_tools import generate_layouts_df, has_offcentre_fold_box


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


def test_mirroring_a_centred_box_changes_nothing():
    """Why `mirror_offcentre` only fires off-centre: f = 0.5 maps to itself."""
    centred = box(0.1, 0.8)
    mirrored = box(1 - 0.1 - 0.8, 0.8)

    assert fold.fold_fraction(centred) == pytest.approx(0.5)
    assert fold.fold_fraction(mirrored) == pytest.approx(fold.fold_fraction(centred))


# ------------------------------------------------------------------- faces

def test_a_not_a_face_detection_is_not_a_face():
    """Face-Recognition writes blurLevel -1 for what it judged not to be one."""
    assert fold.face_spans([face(0.4, 0.6, blur=-1.0)]) == ()


def test_the_widest_face_counts_most():
    spans = fold.face_spans([face(0.1, 0.3), face(0.6, 0.65)])

    assert [round(weight, 3) for _, _, weight in spans] == [1.0, 0.25]


def test_a_fold_down_the_middle_of_a_face_costs_the_most():
    spans = fold.face_spans([face(0.4, 0.6)])

    assert fold.fold_cost(spans, 0.5) == pytest.approx(1.0)


def test_a_fold_clear_of_every_face_costs_nothing():
    spans = fold.face_spans([face(0.4, 0.6)])

    assert fold.fold_cost(spans, 0.8) == 0.0


def test_a_fold_grazing_a_face_costs_less_than_one_through_it():
    spans = fold.face_spans([face(0.4, 0.6)])

    assert 0 < fold.fold_cost(spans, 0.57) < fold.fold_cost(spans, 0.5)


def test_the_gutter_is_a_band_not_a_line():
    """`face_margin`: just outside the face still costs something."""
    spans = fold.face_spans([face(0.4, 0.6)])

    assert fold.fold_cost(spans, 0.605) > 0


def test_a_fold_through_two_faces_is_worse_than_through_one():
    one = fold.face_spans([face(0.45, 0.55)])
    two = fold.face_spans([face(0.42, 0.52), face(0.48, 0.58)])

    assert fold.fold_cost(two, 0.5) > fold.fold_cost(one, 0.5)


# ------------------------------------------------------------ sliding the crop

def test_a_crop_with_slack_slides_off_the_face():
    """A 3:2 photo in a 1:1 box keeps only 2/3 of its width, so it can move."""
    spans = fold.face_spans([face(0.45, 0.55)])
    window = placement_window(None, None, 1.5, 1.0)

    slid = fold.clear_fold(window, 0.5, spans)

    assert fold.fold_cost(spans, fold.fold_image_x(slid, 0.5)) == 0.0
    assert slid[2:] == window[2:], "the same window, moved -- not a different crop"


def test_a_full_spread_box_has_nothing_to_slide():
    """The 2:1 box shows the photo's whole width; the fold is stuck at 0.5."""
    spans = fold.face_spans([face(0.45, 0.55)])
    window = placement_window(None, None, 1.5, 2.0)

    assert window[2] == 1.0
    assert fold.clear_fold(window, 0.5, spans) == window


def test_a_crop_already_clear_of_the_faces_does_not_move():
    spans = fold.face_spans([face(0.05, 0.15)])
    window = placement_window(None, None, 1.5, 1.0)

    assert fold.clear_fold(window, 0.5, spans) == window


def test_a_face_the_slack_cannot_clear_takes_the_least_bad_position():
    """Three faces across the frame: no position is free, so take the cheapest."""
    spans = fold.face_spans([face(0.20, 0.35), face(0.42, 0.58), face(0.65, 0.80)])
    window = placement_window(None, None, 1.2, 1.0)

    slid = fold.clear_fold(window, 0.5, spans)

    assert (fold.fold_cost(spans, fold.fold_image_x(slid, 0.5))
            <= fold.fold_cost(spans, fold.fold_image_x(window, 0.5)))


# ------------------------------------------------------------- the placement

def test_the_placement_slides_the_crop_in_a_cross_fold_box():
    row = pd.Series({'image_as': 1.5, 'faces_info': [face(0.45, 0.55)], 'bodies_info': []})
    crossing = box(0.25, 0.5)

    x, y, w, h = customize_box(row, crossing, album_ar=2, spread_box=crossing)

    assert fold.fold_cost(fold.face_spans(row['faces_info']), x + 0.5 * w) == 0.0


def test_a_box_on_one_page_is_placed_exactly_as_before():
    """Nothing about the fold touches the boxes that do not cross it."""
    row = pd.Series({'image_as': 1.5, 'faces_info': [face(0.45, 0.55)], 'bodies_info': []})
    page = box(0.0, 0.5)

    assert customize_box(row, page, album_ar=2) == placement_window(
        row['faces_info'], row['bodies_info'], 1.5, 1.0)


def test_a_photo_with_no_face_is_placed_as_before():
    row = pd.Series({'image_as': 1.5, 'faces_info': [], 'bodies_info': []})
    crossing = box(0.25, 0.5)

    assert customize_box(row, crossing, album_ar=2) == placement_window(None, None, 1.5, 1.0)


def test_the_mirrored_geometry_decides_where_the_fold_is():
    """`box_id2data` is keyed by the design's positive id, so it holds the
    unmirrored box; `spread_box` is the one actually printed. The two put the
    fold on opposite sides of the photo, and only one of them cuts a face.
    """
    faces = [face(0.58, 0.69), face(0.05, 0.12)]
    row = pd.Series({'image_as': 1.5, 'faces_info': faces, 'bodies_info': []})
    looked_up = box(0.2, 0.4)                   # f = 0.75: the fold hits the face
    printed = box(1 - 0.2 - 0.4, 0.4)           # f = 0.25: it does not

    untouched = placement_window(faces, [], 1.5, 0.8)

    assert customize_box(row, looked_up, 2, spread_box=printed) == untouched
    assert customize_box(row, looked_up, 2, spread_box=looked_up) != untouched


def test_the_slide_can_be_turned_off(monkeypatch):
    monkeypatch.setitem(CONFIGS['fold_awareness'], 'slide_crop', False)
    row = pd.Series({'image_as': 1.5, 'faces_info': [face(0.45, 0.55)], 'bodies_info': []})
    crossing = box(0.25, 0.5)

    assert customize_box(row, crossing, 2, spread_box=crossing) == placement_window(
        row['faces_info'], row['bodies_info'], 1.5, 1.0)


# ------------------------------------------------------- choosing the layout

#: Two layouts holding the same two photos. CROSSING puts its big box over the
#: fold; CLEAN gives each photo its own page.
CROSSING_ID, CLEAN_ID = 100, 200

LAYOUT_ID2DATA = {
    0: {'layout_id': CROSSING_ID, 'left_box_ids': [1], 'right_box_ids': [2],
        'boxes_areas': [{'id': 1, 'area': 0.7}, {'id': 2, 'area': 0.5}]},
    1: {'layout_id': CLEAN_ID, 'left_box_ids': [1], 'right_box_ids': [2],
        'boxes_areas': [{'id': 1, 'area': 1.0}, {'id': 2, 'area': 1.0}]},
}

BOX_ID2DATA = {
    (CROSSING_ID, 1): dict(box(0.0, 0.7, 0.5), area=0.7),
    (CROSSING_ID, 2): dict(box(0.5, 0.5, 0.5), area=0.5),
    (CLEAN_ID, 1): dict(box(0.0, 0.5, 1.0), area=1.0),
    (CLEAN_ID, 2): dict(box(0.5, 0.5, 1.0), area=1.0),
}

#: The fold cuts CROSSING's big box at 0.5/0.7 of its width, and the photo is
#: shown full-width there, so it lands on image column 0.714.
ON_THE_FOLD = face(0.66, 0.77)


def a_photo(photo_id, rank):
    return Photo(id=photo_id, ar=1.5, color=True, rank=rank, photo_class='bride and groom',
                 cluster_label=1, general_time=float(photo_id))


PHOTOS = [a_photo(1, 0.0), a_photo(2, 1.0)]

PHOTO_INFO = {
    1: {'ar': 1.5, 'spans': fold.face_spans([ON_THE_FOLD]), 'faces': [ON_THE_FOLD],
        'bodies': None, 'centroid': None},
    2: {'ar': 1.5, 'spans': fold.face_spans([face(0.05, 0.15)]), 'faces': [face(0.05, 0.15)],
        'bodies': None, 'centroid': None},
}


def candidate(layout_idx, weight):
    spread = SingleSpreadLayout(layout_idx=layout_idx, left_page_photo_idxs={0},
                                right_page_photo_idxs={1}, number_of_squares=0)
    return GroupSingleLayout(spreads_layouts=[spread], weight=weight)


def resolved(group_layout):
    group_layout.resolve_and_order(PHOTOS, LAYOUT_ID2DATA, BOX_ID2DATA, merge_pages=False)
    return group_layout


def test_a_face_on_the_fold_is_measured():
    crossing = resolved(candidate(0, 1.0))

    assert fold.group_layout_fold_cost(crossing, LAYOUT_ID2DATA, BOX_ID2DATA, PHOTO_INFO) > 0


def test_a_layout_that_crosses_no_fold_costs_nothing():
    clean = resolved(candidate(1, 1.0))

    assert fold.group_layout_fold_cost(clean, LAYOUT_ID2DATA, BOX_ID2DATA, PHOTO_INFO) == 0


def test_a_close_rival_with_a_clean_fold_is_taken():
    best, rival = resolved(candidate(0, 1.0)), candidate(1, 0.8)

    chosen = fold.choose_cleaner_fold([best, rival], best, PHOTOS,
                                      LAYOUT_ID2DATA, BOX_ID2DATA, PHOTO_INFO)

    assert chosen is rival


def test_a_rival_the_search_scored_far_lower_is_not_taken():
    """`min_weight_ratio`: the fold never buys a layout the search disliked."""
    best, rival = resolved(candidate(0, 1.0)), candidate(1, 0.1)

    assert fold.choose_cleaner_fold([best, rival], best, PHOTOS,
                                    LAYOUT_ID2DATA, BOX_ID2DATA, PHOTO_INFO) is None


def test_nothing_happens_when_the_best_layout_already_clears_the_fold():
    best, rival = resolved(candidate(1, 1.0)), candidate(0, 0.9)

    assert fold.choose_cleaner_fold([best, rival], best, PHOTOS,
                                    LAYOUT_ID2DATA, BOX_ID2DATA, PHOTO_INFO) is None


def test_a_photo_with_no_face_on_the_fold_needs_no_rival():
    best = resolved(candidate(0, 1.0))
    faceless = {1: dict(PHOTO_INFO[1], spans=fold.face_spans([face(0.05, 0.15)]))}

    assert fold.choose_cleaner_fold([best, candidate(1, 0.9)], best, PHOTOS,
                                    LAYOUT_ID2DATA, BOX_ID2DATA, faceless) is None


def test_the_whole_thing_can_be_turned_off(monkeypatch):
    monkeypatch.setitem(CONFIGS['fold_awareness'], 'enabled', False)
    best = resolved(candidate(0, 1.0))

    assert fold.choose_cleaner_fold([best, candidate(1, 0.9)], best, PHOTOS,
                                    LAYOUT_ID2DATA, BOX_ID2DATA, PHOTO_INFO) is None


def test_the_layout_estimate_is_the_crop_the_placement_makes(monkeypatch):
    """The claim the re-ranking rests on, for everything but a square box."""
    monkeypatch.setitem(CONFIGS['fold_awareness'], 'slide_crop', False)
    crossing = BOX_ID2DATA[(CROSSING_ID, 1)]
    row = pd.Series({'image_as': 1.5, 'faces_info': [ON_THE_FOLD], 'bodies_info': None})

    assert fold.estimate_window(1.5, crossing, 2.0, [ON_THE_FOLD], None, None) == customize_box(
        row, crossing, album_ar=2)


# ------------------------------------------------------------- mirrored designs

def design_box(box_id, x, width, y=0.0, height=0.5):
    return {'id': box_id, 'x': x, 'y': y, 'width': width, 'height': height, 'type': 0}


def test_an_off_centre_cross_fold_box_is_worth_mirroring():
    assert has_offcentre_fold_box([design_box(1, 0.2, 0.35)])


def test_a_box_centred_on_the_fold_is_not():
    assert not has_offcentre_fold_box([design_box(1, 0.0, 1.0)])


def test_a_design_that_stays_on_its_pages_is_not():
    assert not has_offcentre_fold_box([design_box(1, 0.0, 0.5), design_box(2, 0.5, 0.5)])


def test_a_balanced_design_with_an_off_centre_crosser_gets_a_mirror():
    """One box per page, so the old rule would not have mirrored it."""
    data = {'7': {'boxes': [design_box(1, 0.2, 0.35), design_box(2, 0.7, 0.2)]}}

    layouts = generate_layouts_df(data, [7], do_mirror=True)

    assert sorted(layouts['id'].tolist()) == [-7, 7]


# --------------------------------------------------------------- the wiring

def a_row(image_id, faces):
    return {'image_id': image_id, 'image_as': 1.5, 'faces_info': faces,
            'bodies_info': [], 'background_centroid': None}


def test_the_group_frame_is_read_into_fold_info():
    frame = pd.DataFrame([a_row(1, [ON_THE_FOLD]), a_row(2, [face(0.05, 0.15)])])

    info = fold.photo_fold_info(frame)

    assert set(info) == {1, 2}
    assert info[1]['ar'] == 1.5 and info[1]['spans']


def test_a_photo_with_no_face_is_left_out_of_fold_info():
    """Nothing about it can be damaged by a fold, and its absence is the
    early exit for the albums that need none of this."""
    frame = pd.DataFrame([a_row(1, []), a_row(2, [face(0.4, 0.6, blur=-1.0)])])

    assert fold.photo_fold_info(frame) == {}


def test_a_frame_without_faces_at_all_is_handled():
    assert fold.photo_fold_info(pd.DataFrame([{'image_id': 1, 'image_as': 1.5}])) == {}
    assert fold.photo_fold_info(None) == {}


def test_selection_hands_back_the_rival_with_the_cleaner_fold():
    """The path `process_group` actually takes."""
    frame = pd.DataFrame([a_row(1, [ON_THE_FOLD]), a_row(2, [face(0.05, 0.15)])])
    candidates = [candidate(0, 1.0), candidate(1, 0.8)]

    chosen = select_best_layout_for_subgroup(candidates, PHOTOS, LAYOUT_ID2DATA, BOX_ID2DATA,
                                             fold.photo_fold_info(frame), album_ar=2.0)

    assert chosen is candidates[1]


def test_selection_without_fold_info_keeps_the_best_scoring_layout():
    """Every caller that passes nothing -- `predefined`, and the old signature."""
    candidates = [candidate(0, 1.0), candidate(1, 0.8)]

    chosen = select_best_layout_for_subgroup(candidates, PHOTOS, LAYOUT_ID2DATA, BOX_ID2DATA)

    assert chosen is candidates[0]
    assert chosen.spreads_layouts[0].left_page_photos, "still resolved onto its boxes"


# ------------------------------------------------- a design out of the catalogue

#: Design 4518 of `files/designs.json`, verbatim: box 14976 covers the left page
#: and runs 0.058 past the fold, the "page and a lip" family. 14979 is its text
#: box, which `generate_layouts_df` drops.
DESIGN_4518 = {'4518': {'boxes': [
    {'id': 14976, 'x': 0.09466666666666666, 'y': 0.10208333333333333,
     'width': 0.4583333333333333, 'height': 0.7179166666666666, 'layer': 10, 'type': 0, 'tag': None},
    {'id': 14977, 'x': 0.544, 'y': 0.55375,
     'width': 0.35, 'height': 0.34875, 'layer': 20, 'type': 0, 'tag': None},
    {'id': 14978, 'x': 0.592, 'y': 0.11708333333333333,
     'width': 0.2786666666666667, 'height': 0.43666666666666665, 'layer': 30, 'type': 0, 'tag': None},
    {'id': 14979, 'x': 0.09766666666666667, 'y': 0.8354166666666667,
     'width': 0.434, 'height': 0.06125, 'layer': 2013, 'type': 10, 'tag': None},
]}}


def fold_fractions_of(design_id, box_id2data, layouts):
    row = layouts[layouts['id'] == design_id].iloc[0]
    return {box: fold.fold_fraction(box_id2data[(design_id, box)])
            for box in list(row['left_box_ids']) + list(row['right_box_ids'])}


def test_a_real_design_carries_its_fold_through_to_the_box_data():
    """`box_id2data` is what the layout reads, and it is keyed by the signed
    design id -- so the mirrored row has to carry the mirrored geometry."""
    from utils.layouts_tools import get_layouts_data

    layouts = generate_layouts_df(DESIGN_4518, [4518], do_mirror=True)
    _, box_id2data = get_layouts_data(layouts, None, None)

    original = fold_fractions_of(4518, box_id2data, layouts)
    mirrored = fold_fractions_of(-4518, box_id2data, layouts)

    assert original[14976] == pytest.approx(0.884, abs=0.001)
    assert mirrored[14976] == pytest.approx(1 - original[14976])
    assert original[14977] is None and original[14978] is None, "the other boxes stay on their page"


# ------------------------------------------- re-housing a spread in the same shape

def test_two_layouts_holding_the_same_photos_the_same_way_share_a_shape():
    """CROSSING and CLEAN both hold one landscape per page, so they are
    interchangeable: everything `get_score` measures is identical between them."""
    assert (fold.layout_shape(0, LAYOUT_ID2DATA, BOX_ID2DATA)
            == fold.layout_shape(1, LAYOUT_ID2DATA, BOX_ID2DATA) == (0, 1, 0, 0, 1, 0))


def test_a_damaged_spread_is_rehoused_even_when_no_candidate_offers_it():
    """The gap `choose_cleaner_fold` cannot close: `process_group_lists` builds
    candidates with `itertools.product`, so the leading ones all repeat one
    layout for the first spread. 53316807's group `1_other` had the damaged
    photo there and no reachable candidate ever moved it."""
    only = resolved(candidate(0, 1.0))

    before, after = fold.repair_spreads(only, PHOTOS, LAYOUT_ID2DATA, BOX_ID2DATA, PHOTO_INFO)

    assert before > 0 and after == 0
    assert only.spreads_layouts[0].layout_idx == 1, "moved to the layout of the same shape"


def test_rehousing_keeps_the_photos_where_they_were():
    only = resolved(candidate(0, 1.0))
    before_ids = ([p.id for p in only.spreads_layouts[0].left_page_photos],
                  [p.id for p in only.spreads_layouts[0].right_page_photos])

    fold.repair_spreads(only, PHOTOS, LAYOUT_ID2DATA, BOX_ID2DATA, PHOTO_INFO)
    after_ids = ([p.id for p in only.spreads_layouts[0].left_page_photos],
                 [p.id for p in only.spreads_layouts[0].right_page_photos])

    assert before_ids == after_ids


def test_a_spread_that_clears_the_fold_is_left_where_it_is():
    clean = resolved(candidate(1, 1.0))

    before, after = fold.repair_spreads(clean, PHOTOS, LAYOUT_ID2DATA, BOX_ID2DATA, PHOTO_INFO)

    assert (before, after) == (0.0, 0.0)
    assert clean.spreads_layouts[0].layout_idx == 1


def test_selection_rehouses_what_the_candidates_could_not(monkeypatch):
    """End to end through the function `process_group` calls, with a candidate
    list that offers no rival at all."""
    frame = pd.DataFrame([a_row(1, [ON_THE_FOLD]), a_row(2, [face(0.05, 0.15)])])
    only = [candidate(0, 1.0)]

    chosen = select_best_layout_for_subgroup(only, PHOTOS, LAYOUT_ID2DATA, BOX_ID2DATA,
                                             fold.photo_fold_info(frame), album_ar=2.0)

    assert fold.group_layout_fold_cost(chosen, LAYOUT_ID2DATA, BOX_ID2DATA,
                                       fold.photo_fold_info(frame)) == 0
