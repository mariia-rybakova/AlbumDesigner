from typing import List, Optional

import pandas as pd

from src.spreads_layout.partitions import get_partitions
from src.spreads_layout.combinations import get_combinations
from src.spreads_layout.spread_layouts import layout_combination, eval_multi_spreads, Penalties
from src.spreads_layout.group_layouts import GroupSingleLayout, list_multi_spreads
from src.core.models import SpreadSearchParams
from src.core.photos import Photo
from utils.configs import CONFIGS


def generate_filtered_multi_spreads(photos: List[Photo], layouts_df: pd.DataFrame, spread_params: List[float], params: SpreadSearchParams, logger) -> Optional[List[GroupSingleLayout]]:
    photos_df = pd.DataFrame([photo.__dict__ for photo in photos])
    photos_df = photos_df.sort_values('general_time')
    partitions = get_partitions(photos_df, spread_params, params, layouts_df=layouts_df)
    # logger.info('Number of photos: {}. Possible partitions: {}'.format(len(photos), layout_parts))

    combs = get_combinations(partitions, photos, layouts_df, spread_params, params)

    #print("Getting the filtered multi srpreads")
    group_single_layouts = []
    for idx, comb in enumerate(combs):
        multispread_layouts = layout_combination(comb, layouts_df, photos, params)
        if multispread_layouts is not None:
            if len(photos) < 13:
                penalty = Penalties(
                    crop_penalty=CONFIGS['crop_penalty'],
                    color_mix=CONFIGS['color_mix'],
                    class_mix=CONFIGS['class_mix'],
                    orientation_mix=CONFIGS['orientation_mix'],
                    score_threshold=params.score_threshold,
                    double_mix_color=CONFIGS['double_page_color_mix']
                )
            else:
                penalty = Penalties(
                    crop_penalty=0.8,
                    color_mix=CONFIGS['color_mix'],
                    class_mix=CONFIGS['class_mix'],
                    orientation_mix=CONFIGS['orientation_mix'],
                    score_threshold=params.score_threshold,
                    double_mix_color=CONFIGS['double_page_color_mix'],
                    context_mix_penalty=0.00001,
                    time_order_penalty=0.5
                )
            multispread_layouts = eval_multi_spreads(multispread_layouts, layouts_df, photos, penalty)
            group_single_layouts += list_multi_spreads(multispread_layouts)

        if len(group_single_layouts) > 10000:
            group_single_layouts = sorted(group_single_layouts, key=lambda layout: layout.score, reverse=True)[:1000]

    if len(group_single_layouts) == 0:
        return None

    filtered = sorted(group_single_layouts, key=lambda layout: layout.score, reverse=True)
    max_score = filtered[0].score
    filtered = [layout for layout in filtered if layout.score / max_score > 0.01]

    return filtered[:1000]