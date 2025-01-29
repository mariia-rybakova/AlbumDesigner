import random
from utils.cover_processing import get_cover_img,get_important_imgs


def get_cover_end_layout(layout_df,logger):
    # Get all layout keys where "number of boxes" == 1
    one_img_layouts = [key for key, layout in layout_df.iterrows() if layout["number of boxes"] == 1]

    # Ensure there are at least 2 valid layouts
    if len(one_img_layouts) < 2:
        logger.error("Not enough layouts with one image to select two distinct ones.")
        return None

    # Select two distinct layouts
    chosen_layouts = random.sample(one_img_layouts, 2)

    return chosen_layouts


def process_wedding_cover_end_image(df,logger):
    bride_groom_highest_images = get_important_imgs(df, top=50)
    # if we didn't find the highest ranking images then we won't be able to get cover image
    if len(bride_groom_highest_images) > 0:
        # get cover image and remove it from dataframe
        data_df, cover_image_id, cover_image_df = get_cover_img(df, bride_groom_highest_images)
        return data_df, cover_image_id, cover_image_df
    else:
        logger.error("Image Cover not selected for wedding")
        return df, None,None


def process_non_wedding_cover_image(df, logger):
    pass



