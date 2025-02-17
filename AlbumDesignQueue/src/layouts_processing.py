

import io
import json
import csv

import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


def classify_box(box, tolerance):
    # Adjust width and height based on 1:2 aspect ratio
    adjusted_width = float(box['width'] * 2)
    adjusted_height = float(box['height'])

    area = adjusted_width * adjusted_height

    if adjusted_width == 1 and adjusted_height == 1:
        return 'full page square', area
    elif abs(adjusted_width - adjusted_height) <= tolerance:
        if area >= 0.5:  # Arbitrary threshold to differentiate small and large squares
            return 'large square', area
        else:
            return 'small square', area
    elif adjusted_width / adjusted_height < 1:  # Compare adjusted dimensions for 1:2 ratio
        return 'portrait', area
    elif adjusted_width / adjusted_height > 1:
        return 'landscape', area
    elif adjusted_width == 1 or adjusted_height:
        return 'large square', area



def classify_boxes_global(json_file_path, id_list, tolerance=0.05):
    # Read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Initialize lists to hold areas for all portrait and landscape boxes
    all_portrait_areas = []
    all_landscape_areas = []

    # First pass: Collect areas for all boxes
    for item in data:
        boxes = [box for box in item['boxes'] if box['type'] == 0]  # Filter out text boxes
        for box in boxes:
            orientation, area = classify_box(box, tolerance)
            if orientation == 'portrait':
                all_portrait_areas.append(area)
            elif orientation == 'landscape':
                all_landscape_areas.append(area)

    # Calculate the global average area for portrait and landscape boxes
    avg_portrait_area = sum(all_portrait_areas) / len(all_portrait_areas) if all_portrait_areas else 0
    avg_landscape_area = sum(all_landscape_areas) / len(all_landscape_areas) if all_landscape_areas else 0

    # Initialize the result dictionary
    result = {}

    # Create a PDF canvas with custom page size
    page_width = 2000
    page_height = page_width / 2  # Aspect ratio 1:2
    # Second pass: Classify boxes for each ID and generate visualization
    for item in data:
        if item['id'] in id_list:
            boxes = [box for box in item['boxes'] if box['type'] == 0]  # Filter out text boxes
            num_boxes = len(boxes)

            if num_boxes == 0:
                continue  # Skip if there are no image boxes

            # Initialize counts for each classification with left/right distinction
            num_left_small_portrait = 0
            num_right_small_portrait = 0
            num_left_large_portrait = 0
            num_right_large_portrait = 0
            num_left_small_landscape = 0
            num_right_small_landscape = 0
            num_left_large_landscape = 0
            num_right_large_landscape = 0
            num_left_small_square = 0
            num_right_small_square = 0
            num_left_large_square = 0
            num_right_large_square = 0
            num_full_page_square = 0

            left_box_ids = []
            right_box_ids = []
            left_portrait_ids = []
            right_portrait_ids = []
            left_landscape_ids = []
            right_landscape_ids = []
            left_square_ids = []
            right_square_ids = []
            boxes_areas = []

            for box in boxes:
                orientation, area = classify_box(box, tolerance)
                position = 'left' if box['x'] < 0.5 else 'right'

                boxes_areas.append({'id': box['id'], 'area': area})

                if position == 'left':
                    left_box_ids.append(box['id'])
                    if orientation == 'portrait':
                        left_portrait_ids.append(box['id'])
                    elif orientation == 'landscape':
                        left_landscape_ids.append(box['id'])
                    elif 'square' in orientation:
                        left_square_ids.append(box['id'])
                else:
                    right_box_ids.append(box['id'])
                    if orientation == 'portrait':
                        right_portrait_ids.append(box['id'])
                    elif orientation == 'landscape':
                        right_landscape_ids.append(box['id'])
                    elif 'square' in orientation:
                        right_square_ids.append(box['id'])

                if orientation == 'portrait':
                    if area < avg_portrait_area:
                        if position == 'left':
                            num_left_small_portrait += 1
                        else:
                            num_right_small_portrait += 1
                    else:
                        if position == 'left':
                            num_left_large_portrait += 1
                        else:
                            num_right_large_portrait += 1
                elif orientation == 'landscape':
                    if area < avg_landscape_area:
                        if position == 'left':
                            num_left_small_landscape += 1
                        else:
                            num_right_small_landscape += 1
                    else:
                        if position == 'left':
                            num_left_large_landscape += 1
                        else:
                            num_right_large_landscape += 1
                elif orientation == 'small square':
                    if position == 'left':
                        num_left_small_square += 1
                    else:
                        num_right_small_square += 1
                elif orientation == 'large square':
                    if position == 'left':
                        num_left_large_square += 1
                    else:
                        num_right_large_square += 1
                elif orientation == 'full page square':
                    num_full_page_square += 1

            # Determine if there are mixed orientations on the left or right side
            left_portrait = num_left_small_portrait + num_left_large_portrait
            right_portrait = num_right_small_portrait + num_right_large_portrait
            left_landscape = num_left_small_landscape + num_left_large_landscape
            right_landscape = num_right_small_landscape + num_right_large_landscape
            left_square = num_left_small_square + num_left_large_square
            right_square = num_right_small_square + num_right_large_square

            left_mixed = (left_portrait > 0 and left_landscape > 0) or (left_portrait > 0 and left_square > 0) or (
                        left_landscape > 0 and left_square > 0)
            right_mixed = (right_portrait > 0 and right_landscape > 0) or (right_portrait > 0 and right_square > 0) or (
                        right_landscape > 0 and right_square > 0)

            # Store the results for the current ID
            result[item['id']] = {
                "number of boxes": num_boxes,
                "left_small_portrait": num_left_small_portrait,
                "right_small_portrait": num_right_small_portrait,
                "left_large_portrait": num_left_large_portrait,
                "right_large_portrait": num_right_large_portrait,
                "left_small_landscape": num_left_small_landscape,
                "right_small_landscape": num_right_small_landscape,
                "left_large_landscape": num_left_large_landscape,
                "right_large_landscape": num_right_large_landscape,
                "left_small_square": num_left_small_square,
                "right_small_square": num_right_small_square,
                "left_large_square": num_left_large_square,
                "right_large_square": num_right_large_square,
                "full_page_square": num_full_page_square,
                "left_box_ids": left_box_ids,
                "right_box_ids": right_box_ids,
                "left_portrait_ids": left_portrait_ids,
                "right_portrait_ids": right_portrait_ids,
                "left_landscape_ids": left_landscape_ids,
                "right_landscape_ids": right_landscape_ids,
                "left_square_ids": left_square_ids,
                "right_square_ids": right_square_ids,
                "boxes_areas": boxes_areas,
                "left_mixed": left_mixed,
                "right_mixed": right_mixed,
                "boxes_info": boxes,
            }

    return result


def save_results(save_path, data):
    # File name
    csv_file = save_path + '\layouts.csv'

    # Extract the fieldnames from the first dictionary entry
    fieldnames = ['id'] + list(next(iter(data.values())).keys())

    # Write to CSV file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write the data
        for key, value in data.items():
            row = {'id': key, **value}
            writer.writerow(row)

    print(f"Data has been written to {csv_file}")


def generate_layouts_file(design_path,desings_ids,save_path):
    desings_ids = [3444, 3415, 3417, 3418, 3419, 3420, 3421, 3423, 3424, 3425, 3426, 3427, 3428, 3429, 3430, 3431, 3432,
                   3433, 3434, 3435, 3436, 3437, 3438, 3439, 3440, 3441, 3442, 3443, 3445, 3449, 3450, 3451, 3452, 3453,
                   3454, 3455, 3456, 3457, 3458, 3459, 3460, 3461, 3462, 3463, 3464, 3465, 3466, 3467, 3468, 3469, 3470,
                   3471, 3472, 3473, 3474, 3475, 3476, 3477, 3478, 3479, 3480, 3481, 3482, 3483, 3484, 3485, 3486, 3487,
                   3488, 3489, 3490, 3491, 3492, 3494, 3495, 3496, 15971, 15972, 15973, 15974, 15975, 15976, 15977,
                   15978, 15979, 15980, 15981, 15982, 15983, 15984, 15990, 15991, 15992, 15994, 15995, 15997, 15998,
                   15999, 16000, 16001, 16002, 16003, 16004, 16111, 16112, 17109, 17110]
    classified_boxes = classify_boxes_global(design_path, desings_ids)
    save_results(save_path, classified_boxes)