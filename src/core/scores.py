from scipy.stats import pearsonr


def calculate_correlation_score(layout_id2data, photos, all_spreads_data):
    """
        Calculate the correlation score for a layout based on the correlation
        between the area of the boxes and the image importance ranks.

        :param layout: layout pandas series
        :param image_ranks: dict, the rank of importance for each image {image_id: rank}.
        :param spread_data: spread data item
        :param default_box_area: float, default area for a box if not found.
        :param default_rank: int, default rank for an image if not found.
        :return: float, the correlation score.
        """
    default_box_area, default_rank = 0.1, 0.3

    box_areas = []
    ranks = []

    for cur_spread_data in all_spreads_data:
        layout = layout_id2data[cur_spread_data[0]]
        layout_boxes = layout['boxes_areas']

        left_boxes_ids = layout['left_box_ids']
        right_boxes_ids = layout['right_box_ids']

        left_photos = list(cur_spread_data[1])
        right_photos = list(cur_spread_data[2])

        layout_boxes_dict = {box['id']: box['area'] for box in layout_boxes}

        for box_id, photo_id in zip(left_boxes_ids, left_photos):
            photo_data = photos[photo_id]
            box_area = layout_boxes_dict.get(box_id, default_box_area)
            image_rank = photo_data.rank

            box_areas.append(box_area)
            ranks.append(image_rank)

        for box_id, photo_id in zip(right_boxes_ids, right_photos):
            photo_data = photos[photo_id]
            box_area = layout_boxes_dict.get(box_id, default_box_area)
            image_rank = photo_data.rank

            box_areas.append(box_area)
            ranks.append(image_rank)

        # Check if box_areas or ranks are empty or if there is no variation
        if not box_areas or not ranks or len(set(box_areas)) == 1 or len(set(ranks)) == 1:
            return 0.1  # Return a default correlation score (e.g., 0) if data is not suitable for correlation

    # Calculate Pearson correlation coefficient
    correlation, _ = pearsonr(ranks, box_areas)

    return correlation / 2 + 0.5


def add_ranking_score(filtered_spreads, photos, layout_id2data):
    for idx, cur_group_spreads in enumerate(filtered_spreads):
        cur_spreads = cur_group_spreads[0]
        correlation_score = calculate_correlation_score(layout_id2data, photos, cur_spreads)
        filtered_spreads[idx][1] *= correlation_score

    return filtered_spreads


def assign_photos_order_one_side(photos, boxes_ids, design_box_id2data):
    photos_portrait = [photo for photo in photos if photo.ar < 1]
    photos_landscape = [photo for photo in photos if photo.ar >= 1]
    port_idx = 0
    land_idx = 0

    photos_order = [None] * len(boxes_ids)
    for idx, box_id in enumerate(boxes_ids):
        box_data = design_box_id2data[box_id]
        if box_data['orientation'] == 'portrait':
            if port_idx != len(photos_portrait):
                cur_photos = photos_portrait[port_idx]
                port_idx += 1
                photos_order[idx] = cur_photos
            elif land_idx != len(photos_landscape):
                cur_photos = photos_landscape[land_idx]
                land_idx += 1
                photos_order[idx] = cur_photos
            else:
                print("Error: no more photos to add")
        if box_data['orientation'] == 'landscape':
            if land_idx != len(photos_landscape):
                cur_photos = photos_landscape[land_idx]
                land_idx += 1
                photos_order[idx] = cur_photos
            elif port_idx != len(photos_portrait):
                cur_photos = photos_portrait[port_idx]
                port_idx += 1
                photos_order[idx] = cur_photos
            else:
                print("Error: no more photos to add")

    for idx, box_id in enumerate(boxes_ids):
        box_data = design_box_id2data[box_id]
        if box_data['orientation'] == 'square':
            if port_idx != len(photos_portrait):
                cur_photos = photos_portrait[port_idx]
                port_idx += 1
                photos_order[idx] = cur_photos
            elif land_idx != len(photos_landscape):
                cur_photos = photos_landscape[land_idx]
                land_idx += 1
                photos_order[idx] = cur_photos
            else:
                print("Error: no more photos to add")

    return photos_order


def assign_photos_order(spreads, layout_id2data, design_box_id2data):
    for idx, spread in enumerate(spreads[0]):
        layout_data = layout_id2data[spread[0]]
        left_boxes_ids = layout_data['left_box_ids']
        right_boxes_ids = layout_data['right_box_ids']
        left_photos = spread[1]
        left_photos = sorted(left_photos, key=lambda x: x.general_time)
        right_photos = spread[2]
        right_photos = sorted(right_photos, key=lambda x: x.general_time)

        left_photos_order = assign_photos_order_one_side(left_photos, left_boxes_ids, design_box_id2data)
        right_photos_order = assign_photos_order_one_side(right_photos, right_boxes_ids, design_box_id2data)

        spreads[0][idx][1] = left_photos_order
        spreads[0][idx][2] = right_photos_order

    return spreads
