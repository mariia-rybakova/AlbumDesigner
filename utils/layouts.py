import ast
import json
import numpy as np
import pandas as pd


def load_layouts(layouts_path):
    layouts_df = pd.read_csv(layouts_path)

    layoutPortrait = np.sum(np.array(
        layouts_df[['left_small_portrait', 'right_small_portrait', 'left_large_portrait', 'right_large_portrait']]),
        axis=1)
    layoutLandscape = np.sum(np.array(
        layouts_df[['left_small_landscape', 'right_small_landscape', 'left_large_landscape', 'right_large_landscape']]),
        axis=1)
    layoutSquares = np.sum(np.array(
        layouts_df[
            ['left_small_square', 'right_small_square', 'left_large_square', 'right_large_square',
             'full_page_square']]),
        axis=1)

    for id in layouts_df.index:
        layouts_df.at[id, 'left_portrait_ids'] = json.loads(layouts_df.at[id, 'left_portrait_ids'])
        layouts_df.at[id, 'right_portrait_ids'] = json.loads(layouts_df.at[id, 'right_portrait_ids'])
        layouts_df.at[id, 'left_landscape_ids'] = json.loads(layouts_df.at[id, 'left_landscape_ids'])
        layouts_df.at[id, 'right_landscape_ids'] = json.loads(layouts_df.at[id, 'right_landscape_ids'])
        layouts_df.at[id, 'left_square_ids'] = json.loads(layouts_df.at[id, 'left_square_ids'])
        layouts_df.at[id, 'right_square_ids'] = json.loads(layouts_df.at[id, 'right_square_ids'])

    layouts_df['max portraits'] = layoutPortrait + layoutSquares
    layouts_df['max landscapes'] = layoutLandscape + layoutSquares

    return layouts_df


def load_layouts_coordinates(layouts_coordinates_path):
    with open(layouts_coordinates_path, 'r') as file:
        data = json.load(file)
    return data


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