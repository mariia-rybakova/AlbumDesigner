def get_info_only_for_selected_images(images_selected, gallery_photos_info):
    images_selected_data = {}
    for image_id in images_selected:
        if image_id in gallery_photos_info.keys():
            images_selected_data[image_id] = gallery_photos_info[image_id]
        else:
            print("image has no data")
    return images_selected_data