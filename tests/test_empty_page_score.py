"""An empty page is not a mixed one.

From the 2026-09-23T09:16 dev run of project 53819935: group None|8 (two
landscapes and a portrait) could be laid out as `[2, 1]`, the lone landscape on
the design's full-bleed layout 3444, whose one box sits on the left page. The
empty right page was scored as mixed colour and mixed class, 1e-9 together, so
`[3]` won although the partition and combination scores both preferred
`[2, 1]`.
"""

import pandas as pd
import pytest

from src.core.photos import Photo
from src.spreads_layout.spreads.spread import Penalties, SingleSpreadLayout
from utils.configs import CONFIGS

LANDSCAPE = Photo(id=12490891921, ar=1.5, color=True, rank=54, photo_class='None',
                  cluster_label=1, general_time=10.0, original_context='None')


@pytest.fixture
def penalty():
    return Penalties(crop_penalty=CONFIGS['crop_penalty'], color_mix=CONFIGS['color_mix'],
                     class_mix=CONFIGS['class_mix'], orientation_mix=CONFIGS['orientation_mix'],
                     score_threshold=0.01, double_mix_color=CONFIGS['double_page_color_mix'])


@pytest.fixture
def layouts_df():
    return pd.DataFrame({'left_mixed': [False], 'right_mixed': [False]})


def test_empty_page_has_nothing_to_mix():
    props = SingleSpreadLayout.check_page_properties(set(), [LANDSCAPE])
    assert props.is_same_color and props.is_same_class and not props.is_bride_groom_mix


def test_full_bleed_spread_scores_like_one_photo_per_page(penalty, layouts_df):
    spread = SingleSpreadLayout(layout_idx=0, left_page_photo_idxs={0},
                                right_page_photo_idxs=set(), number_of_squares=0)
    assert spread.get_score([LANDSCAPE], layouts_df, penalty) == 1.0
    assert spread.penalty_breakdown['penalties_applied'] == []
