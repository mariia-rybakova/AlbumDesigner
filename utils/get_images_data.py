def get_info_only_for_selected_images(images_selected, gallery_photos_info, logger=None):
    images_selected_data = {}
    for image_id in images_selected:
        if image_id in gallery_photos_info.keys():
            images_selected_data[image_id] = gallery_photos_info[image_id]
        else:
            logger.warning(f"image selected {image_id} has no data!")
            print("image has no data")

    if len(images_selected_data) != len(images_selected):
        logger.warning("Not all images that selected included here debug!")
        print("Not all images that selected included here debug!")

    return images_selected_data