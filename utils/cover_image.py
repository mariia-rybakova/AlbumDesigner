import pandas as pd
import random
from utils.cover_processing import get_cover_img,get_important_imgs
from utils.parser import CONFIGS


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
    bride_groom_highest_images = get_important_imgs(df, top=CONFIGS['top_imges_for_cover'])
    # if we didn't find the highest ranking images then we won't be able to get cover image
    if len(bride_groom_highest_images) > 0:
        # get cover image and remove it from dataframe
        data_df, cover_image_id, cover_image_df = get_cover_img(df, bride_groom_highest_images)
        return data_df, cover_image_id, cover_image_df
    else:
        logger.error("Image Cover not selected for wedding")
        return df, None,None


def process_non_wedding_cover_image(df, logger=None):
    # Validate input DataFrame
    required_columns = {'persons_ids', 'image_order', 'image_id'}
    if not isinstance(df, pd.DataFrame):
        logger.error("Error: Input is not a valid pandas DataFrame.")
        return df, [], pd.DataFrame()

    if not required_columns.issubset(df.columns):
        missing_cols = required_columns - set(df.columns)
        logger.error(f"Error: DataFrame is missing required columns: {missing_cols}")
        return df, [], pd.DataFrame()

    try:
        # Collect all unique people IDs in the dataset
        unique_people_ids = set()
        for _, row in df.iterrows():
            if isinstance(row['persons_ids'], list):  # Ensure it's a list
                unique_people_ids.update(row['persons_ids'])

        if not unique_people_ids:
            logger.warning("Warning: No unique people IDs found in dataset.")
            return df, [], pd.DataFrame()

        # Find images that contain all unique people IDs
        selected_images_df = df[
            df['persons_ids'].apply(lambda x: set(x).issuperset(unique_people_ids) if isinstance(x, list) else False)]

        if selected_images_df.empty:
            logger.warning("Warning: No images found containing all unique people IDs.")
            return df, [], pd.DataFrame()

        # Select the top two images with the highest image_order
        selected_images_df = selected_images_df.nlargest(2, 'image_order')

        if selected_images_df.empty:
            logger.warning("Warning: No images selected based on image_order.")
            return df, [], pd.DataFrame()

        # Extract image IDs
        selected_image_ids = selected_images_df['image_id'].tolist()

        # Remove selected images from the original DataFrame
        df_without_selected = df[~df['image_id'].isin(selected_image_ids)]

        logger.info(f"Selected cover images: {selected_image_ids}")
        return df_without_selected, selected_image_ids, selected_images_df

    except Exception as e:
        logger.error(f"Unexpected error in process_non_wedding_cover_image: {str(e)}")
        return df, [], pd.DataFrame()

