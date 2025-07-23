import ast

def get_layouts_data(any_layouts_df, first_page_layouts_df, last_page_layouts_df):
    layout_id2data = dict()
    box_id2data = dict()
    for layouts_df in [any_layouts_df, first_page_layouts_df, last_page_layouts_df]:
        if layouts_df is None:
            continue
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

            # get all photos orientation
            left_portrait_ids = layout['left_portrait_ids']
            right_portrait_ids = layout['right_portrait_ids']
            left_landscape_ids = layout['left_landscape_ids']
            right_landscape_ids = layout['right_landscape_ids']
            left_square_ids = layout['left_square_ids']
            right_square_ids = layout['right_square_ids']
            if isinstance(left_portrait_ids, str):
                left_portrait_ids = ast.literal_eval(left_portrait_ids)
                right_portrait_ids = ast.literal_eval(right_portrait_ids)
                left_landscape_ids = ast.literal_eval(left_landscape_ids)
                right_landscape_ids = ast.literal_eval(right_landscape_ids)
                left_square_ids = ast.literal_eval(left_square_ids)
                right_square_ids = ast.literal_eval(right_square_ids)
            for box_id in left_portrait_ids + right_portrait_ids:
                box_id2data[box_id] = {'orientation': 'portrait'}
            for box_id in left_landscape_ids + right_landscape_ids:
                box_id2data[box_id] = {'orientation': 'landscape'}
            for box_id in left_square_ids + right_square_ids:
                box_id2data[box_id] = {'orientation': 'square'}

            # get boxes areas
            for item in layout_boxes:
                box_id = item['id']
                area = item['area']
                box_id2data[box_id]['area'] = area

            # get boxes info
            boxes_info = layout['boxes_info']
            if isinstance(boxes_info, str):
                boxes_info = ast.literal_eval(boxes_info)
            for item in boxes_info:
                box_id = item['id']
                box_id2data[box_id]['x'] = item['x']
                box_id2data[box_id]['y'] = item['y']
                box_id2data[box_id]['width'] = item['width']
                box_id2data[box_id]['height'] = item['height']

    return layout_id2data, box_id2data
