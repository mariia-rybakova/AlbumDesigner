import ast
import json
import numpy as np
import pandas as pd

def get_layouts_data(layouts_df):
    layout_id2data = dict()
    for idx, layout in layouts_df.iterrows():
        layout_boxes = layout['boxes_areas']
        if isinstance(layout_boxes, str):
            layout_boxes = ast.literal_eval(layout_boxes)

        left_boxes_ids = layout['left_box_ids']
        right_boxes_ids = layout['right_box_ids']
        if isinstance(left_boxes_ids, str):
            left_boxes_ids = ast.literal_eval(left_boxes_ids)
            right_boxes_ids = ast.literal_eval(right_boxes_ids)

        layout_id2data[idx] = {
            'boxes_areas': layout_boxes,
            'left_box_ids': left_boxes_ids,
            'right_box_ids': right_boxes_ids
        }
    return layout_id2data