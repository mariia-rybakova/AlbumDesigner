import random
import pandas as pd

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


def get_important_imgs(data_df, top=5):
    selection_q = ['bride and groom in a great moment together','bride and groom ONLY','bride and groom ONLY with beautiful background ',' intimate moment in a serene setting between bride and groom ONLY','bride and groom Only in the picture  holding hands','bride and groom Only kissing each other in a romantic way',   'bride and groom Only in a gorgeous standing ','bride and groom doing a great photosession together',' bride and groom with a fantastic standing looking to each other with beautiful scene','bride and groom kissing each other in a photoshot','bride and groom holding hands','bride and groom half hugged for a speical photo moment','groom and brides dancing together solo', 'bride and groom cutting cake', ]
    # Step 1: Filter based on the conditions
    filtered_df = data_df[
        (data_df["cluster_context"] == "bride and groom") &
        (data_df["image_subquery_content"].isin(selection_q))
        ]

    # Step 2: Take the top N rows based on the 'top' variable
    top_filtered_df = filtered_df.head(top)

    # Step 3: Extract the image_ids into a list
    image_id_list = top_filtered_df["image_id"].tolist()

    if len(image_id_list) == 0:
        # let's pick another images
        image_id_list = data_df[
            (data_df["image_query_content"] == "bride")].head(top)['image_id'].tolist()

    if len(image_id_list) == 0:
        # let's pick another images
        image_id_list = data_df.head(top)['image_id'].tolist()

    return image_id_list

def process_wedding_cover_end_image(df,logger):
    bride_groom_highest_images = get_important_imgs(df, top=CONFIGS['top_imges_for_cover'])
    # if we didn't find the highest ranking images then we won't be able to get cover image
    if len(bride_groom_highest_images) > 0:
            # Select 2 distinct images
        cover_img_ids = random.sample(bride_groom_highest_images, 2)

        # Get rows corresponding to selected images
        cover_image_df = df[df['image_id'].isin(cover_img_ids)]

        # Remove selected images from main dataframe
        df = df[~df['image_id'].isin(cover_img_ids)]

        return df, cover_img_ids, cover_image_df
    else:
        logger.error("Image Cover not selected for wedding")
        return df, None,None


def process_non_wedding_cover_image(df, logger=None):
    # Validate input DataFrame
    required_columns = {'persons_ids', 'image_order', 'image_id'}

    if not required_columns.issubset(df.columns):
        missing_cols = required_columns - set(df.columns)
        logger.error(f"Error: DataFrame is missing required columns: {missing_cols}")
        return df, [], pd.DataFrame()

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


