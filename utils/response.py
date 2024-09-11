import ast

from utils.crop import smart_cropping

def generate_json_response(cover_img, cover_img_layout_id,sub_groups, sorted_sub_groups_dict, group_name2chosen_combinations,
                           layouts_df,logger):
    result = [{
        "accountId": 9092,
        "alerts": None,
        "bundle": None,
        "compositionPackageId": -1,
        "compositions": [],
        "copies": 1,
        "countCompositions": 1,
        "engagementId": None,
        "externalReference": None,
        "fulfillerId": 0,
        "guserId": 17498886,
        "logicalSelectionsState": [],
        "packageDesignId": None,
        "packageFinishingId": 0,
        "packageStyleId": 0,
        "packageTypeId": 0,
        "placementsImg": [],
        "placementsTxt": [],
        "productId": 1154,
        "projectId": 3441383,
        "revisionCounter": 0,
        "specialOptions": None,
        "status": 0,
        "storeId": 32,
        "userId": 306800045,
        "userJobId": 560618661,
        "__type": "Pictime.Interfaces.webP2CompositionPckDTO"
    }]

    if cover_img_layout_id is not None:
        # Plot the cover image
        cover_layout_info = ast.literal_eval(layouts_df.loc[int(cover_img_layout_id)]['boxes_info'])
        cover_image_id = cover_img['image_id']
        for box in cover_layout_info:
            x, y, w, h = box['x'], box['y'], box['width'], box['height']
            # Resize the image to fit the box
            centroid = cover_img['background_centroid'].values[0]
            cropped_x,cropped_y, croped_w, cropped_h = smart_cropping(float(cover_img['image_as'].iloc[0]), list(cover_img['faces_info']), centroid, float(cover_img['diameter'].iloc[0]))

            result[0]['compositions'].append({
                "compositionId": 1,
                "compositionPackageId": -1,
                "designId": cover_img_layout_id,
                "styleId": -1,
                "revisionCounter": 0,
                "copies": 1,
                "boxes": [
                    {
                        "id": box['id'],
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "layer": -1,
                        "layerOrder": 0,
                        "type": 1
                    }
                ],
                "logicalSelectionsState": [
                    "5x7",
                    "df-3x3",
                    "white",
                    "d-portrait",
                    "d-15450"
                ]
            })
            result[0]['placementsImg'].append({
                "placementImgId": 1,
                "compositionPackageId": -1,
                "compositionId": 1,
                "boxId": box['id'],
                "photoId": cover_image_id,
                "cropX": cropped_x,
                "cropY": cropped_y,
                "cropWidth": croped_w,
                "cropHeight": cropped_h,
                "rotate": 0,
                "projectId": 3441383,
                "photoFilter": 0,
                "photo": None
            })

    for group_name in sorted_sub_groups_dict.keys():
        if group_name not in group_name2chosen_combinations.keys():
            logger.warning(f"Group Name {group_name}  has no results ")
            continue
        group_data = group_name2chosen_combinations[group_name][0]

        number_of_spreads = len(group_data)

        # Loop over each spread
        for spread_index in range(number_of_spreads):
            layout_id = group_data[spread_index][0]
            cur_layout_info = ast.literal_eval(layouts_df.loc[layout_id]['boxes_info'])
            left_box_ids = ast.literal_eval(layouts_df.loc[layout_id]['left_box_ids'])
            right_box_ids = ast.literal_eval(layouts_df.loc[layout_id]['right_box_ids'])

            left_page_photos = list(group_data[spread_index][1])
            right_page_photos = list(group_data[spread_index][2])

            all_box_ids = left_box_ids + right_box_ids
            all_photos = left_page_photos + right_page_photos
            compositionId = 1

            orig_group_name = group_name.split('*')
            parts = orig_group_name[0].split('_')
            group_id = (float(parts[0]), '_'.join(parts[1:]))
            c_group = sub_groups.get_group(group_id)

            # Loop over boxes and plot images
            for i,box in enumerate(cur_layout_info):
                box_id = box['id']
                if box_id not in all_box_ids:
                    print('Some error, cant find box with id: {}'.format(box_id))

                element_index = all_box_ids.index(box_id)
                cur_photo = all_photos[element_index]

                x, y, w, h = box['x'], box['y'], box['width'], box['height']
                c_image_id = cur_photo.id

                c_image_info = c_group[c_group['image_id'] == c_image_id]
                if len(c_image_info) != 0:
                    centroid = c_image_info['background_centroid'].values[0]
                    cropped_x,cropped_y, cropped_w, cropped_h = smart_cropping(float(c_image_info['image_as'].iloc[0]), list(c_image_info['faces_info']), centroid, float(c_image_info['diameter'].iloc[0]))
                    result[0]['compositions'].append({
                        "compositionId": compositionId + 1,
                        "compositionPackageId": -1,
                        "designId": layout_id,
                        "styleId": -1,
                        "revisionCounter": 0,
                        "copies": 1,
                        "boxes": [
                            {
                                "id": box['id'],
                                "x": x,
                                "y": y,
                                "width": w,
                                "height": h,
                                "layer": -1,
                                "layerOrder": 0,
                                "type": 1
                            }
                        ],
                        "logicalSelectionsState": [
                            "5x7",
                            "df-3x3",
                            "white",
                            "d-portrait",
                            "d-15450"
                        ]
                    })
                    result[0]['placementsImg'].append({
                        "placementImgId": i,
                        "compositionPackageId": -1,
                        "compositionId": compositionId + 1,
                        "boxId": box['id'],
                        "photoId": cur_photo.id,
                        "cropX": cropped_x,
                        "cropY": cropped_y,
                        "cropWidth": cropped_w,
                        "cropHeight": cropped_h,
                        "rotate": 0,
                        "projectId": 3441383,
                        "photoFilter": 0,
                        "photo": None
                    })

    return result