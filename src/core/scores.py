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

    # Normalization (maps ranks and areas to a 0-1 scale)
    max_rank = max(ranks) if ranks else 1  # Avoid division by zero
    max_area = max(box_areas) if box_areas else 1

    normalized_ranks = [1 - (rank / max_rank) for rank in ranks]  # Lower ranks are better
    normalized_areas = [area / max_area for area in box_areas]  # Larger areas are better
    # Calculate Pearson correlation coefficient
    correlation, _ = pearsonr(normalized_ranks, normalized_areas)

    return correlation / 2 + 0.5


def add_ranking_score(filtered_spreads, photos, layout_id2data):
    for idx, cur_group_spreads in enumerate(filtered_spreads):
        cur_spreads = cur_group_spreads[0]
        correlation_score = calculate_correlation_score(layout_id2data, photos, cur_spreads)
        filtered_spreads[idx][1] *= correlation_score

    return filtered_spreads


def assign_photos_order_by_area(photos, boxes, portraits_total, landscapes_total):
    photos_portrait = [photo for photo in photos if photo.ar < 1]
    photos_landscape = [photo for photo in photos if photo.ar >= 1]
    port_idx = 0
    land_idx = 0

    photos_order = [None] * len(boxes)
    for idx, box_data in enumerate(boxes):
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

    portraits_total = portraits_total - port_idx
    landscapes_total = landscapes_total - land_idx

    for idx, box_data in enumerate(boxes):
        if box_data['orientation'] == 'square':
            if port_idx != len(photos_portrait) - portraits_total:
                cur_photos = photos_portrait[port_idx]
                port_idx += 1
                photos_order[idx] = cur_photos
            elif land_idx != len(photos_landscape) - landscapes_total:
                cur_photos = photos_landscape[land_idx]
                land_idx += 1
                photos_order[idx] = cur_photos
            else:
                print("Error: no more photos to add")


    return photos_order, portraits_total, landscapes_total


def assign_part_photos_order(boxes, photos):
    portraits_total = len([box for box in boxes if box['orientation'] == 'portrait'])
    landscapes_total = len([box for box in boxes if box['orientation'] == 'landscape'])

    area2boxes = dict()
    for box in boxes:
        cur_area = box['area']
        added = False
        for saved_area in area2boxes.keys():
            if abs(cur_area - saved_area) < 0.01:
                area2boxes[saved_area].append(box)
                added = True
                break
        if added:
            continue
        if cur_area not in area2boxes:
            area2boxes[cur_area] = list()
        area2boxes[cur_area].append(box)

    left_size = max([box['position'] for box in boxes if box['side'] == 0], default=0) + 1
    right_size = max([box['position'] for box in boxes if box['side'] == 1], default=0) + 1
    left_photos_order = [None] * left_size
    right_photos_order = [None] * right_size

    for area in sorted(area2boxes.keys(), reverse=True):
        cur_boxes = area2boxes[area]
        photos_order, portraits_total, landscapes_total = assign_photos_order_by_area(photos, cur_boxes,
                                                                                      portraits_total,
                                                                                      landscapes_total)
        photos_order = sorted(photos_order, key=lambda x: x.general_time)
        for idx, box_data in enumerate(cur_boxes):
            cur_photo = photos_order[idx]
            if box_data['side'] == 0:
                left_photos_order[box_data['position']] = cur_photo
            else:
                right_photos_order[box_data['position']] = cur_photo

        # Remove assigned photos from all_photos so they aren't reused
        assigned_photos = [p for p in photos_order if p is not None]
        if assigned_photos:
            assigned_ids = set(id(p) for p in assigned_photos)
            photos = [p for p in photos if id(p) not in assigned_ids]

    return left_photos_order, right_photos_order


def assign_photos_order(spreads, layout_id2data, design_box_id2data, merge_pages=False):
    for spread_idx, spread in enumerate(spreads[0]):
        layout_data = layout_id2data[spread[0]]
        left_boxes_ids = layout_data['left_box_ids']
        right_boxes_ids = layout_data['right_box_ids']
        left_page_boxes = [{'id': bid,
                      'side': 0,
                      'position': idx,
                      'area': design_box_id2data[(layout_data['layout_id'],bid)]['area'],
                      'orientation': design_box_id2data[(layout_data['layout_id'],bid)]['orientation']
                      } for idx, bid in enumerate(left_boxes_ids)]
        right_page_boxes = [{'id': bid,
                       'side': 1,
                       'position': idx,
                       'area': design_box_id2data[(layout_data['layout_id'],bid)]['area'],
                       'orientation': design_box_id2data[(layout_data['layout_id'],bid)]['orientation']
                       } for idx, bid in enumerate(right_boxes_ids)]

        if merge_pages:
            all_photos = sorted(list(spread[1]) + list(spread[2]), key=lambda x: (x.rank, x.general_time))
            left_photos_order, right_photos_order = assign_part_photos_order(left_page_boxes + right_page_boxes, all_photos)
        else:
            left_page_photos = sorted(list(spread[1]), key=lambda x: (x.rank, x.general_time))
            right_page_photos = sorted(list(spread[2]), key=lambda x: (x.rank, x.general_time))
            left_photos_order, _ = assign_part_photos_order(left_page_boxes, left_page_photos)
            _, right_photos_order = assign_part_photos_order(right_page_boxes, right_page_photos)

        spreads[0][spread_idx][1] = left_photos_order
        spreads[0][spread_idx][2] = right_photos_order

    return spreads
