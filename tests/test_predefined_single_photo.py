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
THREE_BOX = 104      # one box left, two right

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
    str(THREE_BOX): {"boxes": [
        {"id": 5, "x": 0.08, "y": 0.025, "width": 0.34, "height": 0.95, "layer": 1, "type": 0, "tag": None},
        {"id": 6, "x": 0.55, "y": 0.025, "width": 0.2, "height": 0.45, "layer": 1, "type": 0, "tag": None},
        {"id": 7, "x": 0.55, "y": 0.525, "width": 0.2, "height": 0.45, "layer": 1, "type": 0, "tag": None},
    ]},
}

ALL_DESIGNS = (FULL_SPREAD, WIDE_BOX, TWO_PORTRAITS, THREE_BOX)
#: The shape production still loses: nothing in the design holds one photo.
NO_ONE_BOX = (TWO_PORTRAITS, THREE_BOX)

PORTRAIT_AR = 0.67
LANDSCAPE_AR = 1.5


def designs_info(ids=ALL_DESIGNS):
    layouts_df = generate_layouts_df(DESIGNS, list(ids))
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


def lay_out(gallery_df, *spreads, designs=ALL_DESIGNS):
    predefined = PredefinedLayoutInput(spreads=[PredefinedSpread(photo_ids=list(ids)) for ids in spreads])
    result_list, _ = predefined_layout_processing(gallery_df, designs_info(designs), predefined,
                                                  SpreadSearchParams(), True, LOGGER)
    return result_list


def only_spread(result_list):
    assert len(result_list) == 1, "the spread was dropped"
    (group,) = result_list
    (result,) = group.values()
    (spread,) = result.spreads
    return spread


def boxes_of(layout_idx, designs=ALL_DESIGNS):
    row = designs_info(designs)["anyPagelayouts_df"].loc[layout_idx]
    return row["id"], row["number of boxes"]


def photo_ids(spread):
    return [p.id for p in list(spread.left_photos) + list(spread.right_photos)]


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


# ------------------------------ when no layout holds the count the plan asked for
#
# A design holds the photo counts its layouts have boxes for and no others.
# Where the request names a count the design does not offer -- one photo where
# there is no one-box layout, five where the richest layout holds four --
# nothing in the search can help and the spread used to be dropped with its
# photographs. `_fold_into_neighbour` gives them to the spread beside it: the
# album loses a spread, not a photograph.


def test_a_lone_photo_is_folded_into_the_previous_spread():
    """No one-box layout exists, so the orphan joins the two beside it."""
    result_list = lay_out(gallery(PORTRAIT_AR, PORTRAIT_AR, PORTRAIT_AR),
                          [1, 2], [3], designs=NO_ONE_BOX)

    spread = only_spread(result_list)
    design_id, n_boxes = boxes_of(spread.layout_id, NO_ONE_BOX)
    assert abs(design_id) == THREE_BOX and n_boxes == 3
    assert sorted(photo_ids(spread)) == [1, 2, 3], "no photograph was lost"


def test_the_next_spread_takes_it_when_there_is_no_previous_one():
    result_list = lay_out(gallery(PORTRAIT_AR, PORTRAIT_AR, PORTRAIT_AR),
                          [3], [1, 2], designs=NO_ONE_BOX)

    spread = only_spread(result_list)
    assert sorted(photo_ids(spread)) == [1, 2, 3]


def test_a_spread_the_design_can_hold_is_never_folded():
    """The fold is the last resort, not a shortcut past the search."""
    result_list = lay_out(gallery(PORTRAIT_AR, PORTRAIT_AR, PORTRAIT_AR),
                          [1, 2], [3])

    assert len(result_list) == 2, "the single-box layout holds the lone photo on its own"


def test_a_photograph_is_only_lost_when_no_neighbour_can_hold_it_either():
    """Two photos per spread is all this design has: 2 + 1 fits nothing."""
    result_list = lay_out(gallery(PORTRAIT_AR, PORTRAIT_AR, PORTRAIT_AR),
                          [1, 2], [3], designs=(TWO_PORTRAITS,))

    spread = only_spread(result_list)
    assert sorted(photo_ids(spread)) == [1, 2], "the neighbour is unchanged"


def test_the_other_side_gets_its_turn_when_the_first_cannot_hold_it():
    """This design stops at three boxes: 3 + 1 fits nothing, 2 + 1 does."""
    result_list = lay_out(gallery(*([PORTRAIT_AR] * 6)),
                          [1, 2, 3], [4], [5, 6], designs=NO_ONE_BOX)

    merged = [sorted(photo_ids(s)) for group in result_list for r in group.values() for s in r.spreads]
    assert len(result_list) == 2
    assert [1, 2, 3] in merged, "the three-photo spread on the left had no room"
    assert [4, 5, 6] in merged, "so the right side took it"


def test_a_spread_naming_photos_the_gallery_does_not_have_is_not_folded():
    """Nothing to save, and the neighbour must not be relaid for it."""
    result_list = lay_out(gallery(PORTRAIT_AR, PORTRAIT_AR), [1, 2], [999], designs=NO_ONE_BOX)

    spread = only_spread(result_list)
    assert sorted(photo_ids(spread)) == [1, 2]


def test_every_spread_of_a_many_spread_album_survives_the_fold():
    """Two orphans in one album, each folding into its own neighbour."""
    result_list = lay_out(gallery(*([PORTRAIT_AR] * 6)),
                          [1, 2], [3], [4, 5], [6], designs=NO_ONE_BOX)

    kept = sorted(pid for group in result_list for r in group.values()
                  for s in r.spreads for pid in photo_ids(s))
    assert kept == [1, 2, 3, 4, 5, 6], "folded, never dropped"
    assert len(result_list) == 2, "four planned spreads, two printable ones"


def test_an_orphan_does_not_fold_into_a_spread_that_was_folded_away():
    """`[3]` takes `[1,2]` and leaves a hole; `[4]` must step over it to the
    three-photo spread that is actually there, find it full, and try the other
    side -- not merge into a spread that is no longer in the album."""
    result_list = lay_out(gallery(*([PORTRAIT_AR] * 6)),
                          [1, 2], [3], [4], [5, 6], designs=NO_ONE_BOX)

    merged = [sorted(photo_ids(s)) for group in result_list for r in group.values() for s in r.spreads]
    assert [1, 2, 3] in merged
    assert [4, 5, 6] in merged
