from .wedding_selection import smart_wedding_selection
from .non_wedding_selection import smart_non_wedding_selection

def auto_images_selection(df, ten_photos, people_ids, user_relation, category, tags_features,
                          DEBUG,
                          logger=None):
    if category ==1:
        # Select images for creating an album

        images_selected, errors = smart_wedding_selection(df, ten_photos, people_ids, user_relation, tags_features, DEBUG,logger)
    else:
        # Select images for creating an album
        images_selected,errors = smart_non_wedding_selection(df, logger=logger)

    logger.info(f"Number of Selected Images To Design an Album {len(images_selected)}")

    return images_selected, errors