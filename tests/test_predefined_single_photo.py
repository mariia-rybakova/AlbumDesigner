"""A predefined spread holding one portrait photo must not vanish.

Stage 3 places a photo only in a box of its own orientation, and a design's
one-box layouts are its full-spread frames, which classify as landscape on a
2:1 album -- every one of the four in `files/layouts.csv` carries
``max portraits == 0``. So `filter_layouts` returned nothing for a lone
portrait and `predefined_layout_processing` dropped the spread: production lost
4,613 spreads over 18-27 Oct, all but one of them a single photo, which is how
a 12-spread album shipped as 9.

The designs here are built through the real reader, so the box classification
under test is the production one.
"""
import logging

import pandas as pd

from src.core.models import SpreadSearchParams
from src.predefined.models import PredefinedLayoutInput, PredefinedSpread
from src.predefined.processing import predefined_layout_processing
from src.readers.design_templates import generate_layouts_df, get_layouts_data

LOGGER = logging.getLogger("test_predefined_single_photo")

FULL_SPREAD = 101   # one box over the whole spread: printed 2.00 : 1
WIDE_BOX = 102      # one narrower box:               printed 1.47 : 1
TWO_PORTRAITS = 103  # one portrait box per page

DESIGNS = {
    str(FULL_SPREAD): {"boxes": [
        {"id": 1, "x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0, "layer": 1, "type": 0, "tag": None},
    ]},
    str(WIDE_BOX): {"boxes": [
        {"id": 2, "x": 0.15, "y": 0.025, "width": 0.7, "height": 0.95, "layer": 1, "type": 0, "tag": None},
    ]},
    str(TWO_PORTRAITS): {"boxes": [
        {"id": 3, "x": 0.1, "y": 0.025, "width": 0.3, "height": 0.95, "layer": 1, "type": 0, "tag": None},
        {"id": 4, "x": 0.6, "y": 0.025, "width": 0.3, "height": 0.95, "layer": 1, "type": 0, "tag": None},
    ]},
}

PORTRAIT_AR = 0.67
LANDSCAPE_AR = 1.5


def designs_info():
    layouts_df = generate_layouts_df(DESIGNS, [FULL_SPREAD, WIDE_BOX, TWO_PORTRAITS])
    layout_id2data, box_id2data = get_layouts_data(layouts_df, None, None)
    return {
        "anyPagelayouts_df": layouts_df,
        "anyPagelayout_id2data": layout_id2data,
        "anyPagebox_id2data": box_id2data,
        "maxPages": 40,
        "albumAr": 2.0,
    }


def gallery(*ars):
    """One row per photo, ids 1..n, in the columns `get_photos_from_df` reads."""
    return pd.DataFrame([{
        "image_id": i + 1,
        "image_as": ar,
        "image_color": 1,
        "image_order": i + 1,
        "cluster_label": 1,
        "cluster_context": "portrait",
        "general_time": 1000 + i,
    } for i, ar in enumerate(ars)])


def lay_out(gallery_df, *spreads):
    predefined = PredefinedLayoutInput(spreads=[PredefinedSpread(photo_ids=list(ids)) for ids in spreads])
    result_list, _ = predefined_layout_processing(gallery_df, designs_info(), predefined,
                                                  SpreadSearchParams(), True, LOGGER)
    return result_list


def only_spread(result_list):
    assert len(result_list) == 1, "the spread was dropped"
    (group,) = result_list
    (result,) = group.values()
    (spread,) = result.spreads
    return spread


def boxes_of(layout_idx):
    row = designs_info()["anyPagelayouts_df"].loc[layout_idx]
    return row["id"], row["number of boxes"]


# ------------------------------------------------------- the regression itself

def test_no_one_box_layout_accepts_a_portrait():
    """The premise: without the fallback there is nothing for stage 3 to find."""
    layouts_df = designs_info()["anyPagelayouts_df"]
    one_box = layouts_df[layouts_df["number of boxes"] == 1]

    assert len(one_box) == 2, "both single-box designs were read"
    assert (one_box["max portraits"] == 0).all(), "the full-spread frames classify as landscape"


def test_a_lone_portrait_still_gets_a_spread():
    spread = only_spread(lay_out(gallery(PORTRAIT_AR), [1]))

    design_id, n_boxes = boxes_of(spread.layout_id)
    assert n_boxes == 1
    assert [p.id for p in spread.left_photos + spread.right_photos] == [1]


def test_the_lone_portrait_takes_the_box_closest_to_its_own_ratio():
    """Not the full-spread frame, which would crop hardest and cross the fold."""
    spread = only_spread(lay_out(gallery(PORTRAIT_AR), [1]))

    design_id, _ = boxes_of(spread.layout_id)
    assert abs(design_id) == WIDE_BOX


def test_the_photo_sits_on_the_page_that_holds_the_box():
    """`assembly_output` pairs `left_box_ids + right_box_ids` with
    `left_photos + right_photos` positionally, so the two must agree."""
    spread = only_spread(lay_out(gallery(PORTRAIT_AR), [1]))

    row = designs_info()["anyPagelayouts_df"].loc[spread.layout_id]
    assert len(row["left_box_ids"]) == len(spread.left_photos)
    assert len(row["right_box_ids"]) == len(spread.right_photos)


# --------------------------------------------- what the fallback must not touch

def test_a_lone_landscape_goes_through_the_normal_search():
    spread = only_spread(lay_out(gallery(LANDSCAPE_AR), [1]))

    _, n_boxes = boxes_of(spread.layout_id)
    assert n_boxes == 1
    assert [p.id for p in spread.left_photos + spread.right_photos] == [1]


def test_two_portraits_still_use_the_two_box_layout():
    spread = only_spread(lay_out(gallery(PORTRAIT_AR, PORTRAIT_AR), [1, 2]))

    design_id, n_boxes = boxes_of(spread.layout_id)
    assert abs(design_id) == TWO_PORTRAITS and n_boxes == 2
    assert len(spread.left_photos) == 1 and len(spread.right_photos) == 1


def test_a_spread_whose_photos_are_not_in_the_gallery_is_still_dropped():
    assert lay_out(gallery(PORTRAIT_AR), [999]) == []


def test_every_spread_of_a_mixed_album_survives():
    """The shape production loses: one portrait spread among ordinary ones."""
    result_list = lay_out(gallery(PORTRAIT_AR, PORTRAIT_AR, LANDSCAPE_AR),
                          [1, 2], [3], [1])

    assert len(result_list) == 3
