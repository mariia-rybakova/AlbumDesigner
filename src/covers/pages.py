"""Putting the chosen photographs onto the first and last pages.

Once `choosing` has the two photographs, this finds a one-box design that
suits each one's orientation (`find_single_box_layout`), lets
`spreads_layout.fold` pick between the candidates so the fold misses the
face, and builds the pages the reply carries.

This is the only part of `covers` that touches the layout side; the scoring
and the choosing never do.
"""



from src.spreads_layout import fold
from src.covers.choosing import (choose_good_non_wedding_images, choose_good_wedding_images)


def find_single_box_layout(layouts_df, orientation):
    """Find a single-box layout matching the image orientation.

    Prefers large boxes, falls back to any single-box layout.

    Every branch requires ``number of boxes == 1``. "max landscapes == 1" only
    says the layout accepts at most one landscape -- a three-box design with one
    large landscape and two portrait slots satisfies it. The caller fills this
    layout from the cover photos, of which a wedding has exactly one, and then
    walks `iloc` over every box: a two-box "single-box" layout indexes past the
    end and fails the whole album with `single positional indexer is
    out-of-bounds`. 53009168 died that way.
    """
    if orientation == "landscape":
        # Prefer large landscape or large square
        candidates = [key for key, layout in layouts_df.iterrows() if layout["max landscapes"] == 1 and (
            layout["number of boxes"] == 1) and (
            layout['right_large_landscape'] == 1 or layout["left_large_landscape"] == 1 or
            layout["left_large_square"] == 1 or layout["right_large_square"] == 1)]
        if not candidates:
            # Fall back: any single-box layout that accepts a landscape
            candidates = [key for key, layout in layouts_df.iterrows() if layout["max landscapes"] >= 1
                          and layout["number of boxes"] == 1]
    else:
        # Prefer large portrait or large square
        candidates = [key for key, layout in layouts_df.iterrows() if layout["max portraits"] == 1 and (
            layout["number of boxes"] == 1) and (
            layout['left_large_portrait'] == 1 or layout["right_large_portrait"] == 1 or
            layout["left_large_square"] == 1 or layout["right_large_square"] == 1)]
        if not candidates:
            # Fall back: any single-box layout that accepts a portrait
            candidates = [key for key, layout in layouts_df.iterrows() if layout["max portraits"] >= 1
                          and layout["number of boxes"] == 1]
    # Last resort: any single-box layout
    if not candidates:
        candidates = [key for key, layout in layouts_df.iterrows() if layout["number of boxes"] == 1]
    return candidates


def get_design_id(layout_df, number_of_boxes, logger):
    # Get all layout keys where "number of boxes" == 1
    img_layouts = [key for key, layout in layout_df.iterrows() if layout["number of boxes"] == number_of_boxes]

    # Ensure there are at least 2 valid layouts
    if len(img_layouts) < 1:
        logger.error("Not enough layouts with one image to select two distinct ones.")
        return None

    return img_layouts[0]


def generate_first_last_pages(message, df, logger):
    first_last_pages_data_dict = dict()

    if message.pagesInfo.get("firstPage"):
        if message.content.get('is_wedding', True):
            # The same flag ProcessStage reads a few lines later: on the manual
            # route the chosen photos are the user's, and none of them is the
            # pipeline's to discard.
            manual_selection = message.content.get('manual_selection', False)
            # What `enrich.key_pages` chose over the whole gallery. Imported
            # here for the same reason `_concept_score` imports: `src.core` must
            # not depend on the pipeline package at load time.
            from src.pipeline.contracts import KeyPages
            preferred = KeyPages.from_content(message.content.get('key_pages'))
            df, first_images_ids, first_imgs_df, last_images_ids, last_imgs_df = choose_good_wedding_images(df,
                                                                                                            message.content.get(
                                                                                                                'bride and groom'),
                                                                                                            logger,
                                                                                                            manual_selection=manual_selection,
                                                                                                            preferred=preferred)
        else:
            # Two: one for the opening, one for the closing. Asking for one
            # left the opening with nothing to place.
            df, first_images_ids, last_images_ids, first_imgs_df, last_imgs_df = choose_good_non_wedding_images(df, 2,
                                                                                                                logger)

        if message.pagesInfo.get("firstPage"):
            layouts_df = message.designsInfo[f"firstPage_layouts_df"]
            if not first_imgs_df.empty:
                cover_layouts = find_single_box_layout(layouts_df, first_imgs_df["image_orientation"].values[0])
                if cover_layouts:
                    # The opening and closing pages are placed by `cover_box`,
                    # which the layout search never sees, so the fold is chosen
                    # here or not at all.
                    cover_layouts = [fold.pick_cover_layout(
                        cover_layouts, layouts_df,
                        message.designsInfo.get('anyPagebox_id2data') or {},
                        first_imgs_df.iloc[0],
                        float(message.designsInfo.get('albumAr', 2.0)), logger)]
                    first_last_pages_data_dict["firstPage"] = {
                        'design_id': cover_layouts[0],
                        'first_images_ids': first_images_ids,
                        'first_images_df': first_imgs_df,
                    }
                else:
                    logger.warning("No matching single-box layout found for firstPage")
        else:
            logger.warning("For this album theres no first page cover image")

        if message.pagesInfo.get('lastPage'):
            layouts_df = message.designsInfo[f"lastPage_layouts_df"]
            if not last_imgs_df.empty:
                cover_layouts = find_single_box_layout(layouts_df, last_imgs_df["image_orientation"].values[0])
                if cover_layouts:
                    # The opening and closing pages are placed by `cover_box`,
                    # which the layout search never sees, so the fold is chosen
                    # here or not at all.
                    cover_layouts = [fold.pick_cover_layout(
                        cover_layouts, layouts_df,
                        message.designsInfo.get('anyPagebox_id2data') or {},
                        last_imgs_df.iloc[0],
                        float(message.designsInfo.get('albumAr', 2.0)), logger)]
                    first_last_pages_data_dict['lastPage'] = {
                        'design_id': cover_layouts[0],
                        'last_images_ids': last_images_ids,
                        'last_images_df': last_imgs_df,
                    }
                else:
                    logger.warning("No matching single-box layout found for lastPage")
        else:
            logger.warning("For this album theres no last page cover image")

    return df, first_last_pages_data_dict
