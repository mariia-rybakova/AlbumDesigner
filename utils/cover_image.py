import random
import pandas as pd

from utils.parser import CONFIGS


def get_design_id(layout_df, number_of_boxes, logger):
    # Get all layout keys where "number of boxes" == 1
    img_layouts = [key for key, layout in layout_df.iterrows() if layout["number of boxes"] == number_of_boxes]

    # Ensure there are at least 2 valid layouts
    if len(img_layouts) < 1:
        logger.error("Not enough layouts with one image to select two distinct ones.")
        return None

    return img_layouts[0]


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
            (data_df["cluster_context"] == "bride and groom")].head(top)['image_id'].tolist()

    if len(image_id_list) == 0:
        # let's pick another images
        image_id_list = data_df[
            (data_df["image_query_content"] == "bride")].head(top)['image_id'].tolist()

    if len(image_id_list) == 0:
        # let's pick another images
        image_id_list = data_df.head(top)['image_id'].tolist()

    return image_id_list


def choose_good_wedding_images(df, number_of_images, logger):
    good_images = get_important_imgs(df, top=CONFIGS['top_imges_for_cover'])
    # if we didn't find the highest ranking images then we won't be able to get cover image
    if len(good_images) >= number_of_images:
        # Select 2 distinct images
        cover_img_ids = random.sample(good_images, number_of_images)

        # Get rows corresponding to selected images
        cover_image_df = df[df['image_id'].isin(cover_img_ids)]

        # Remove selected images from main dataframe
        df = df[~df['image_id'].isin(cover_img_ids)]

        return df, cover_img_ids, cover_image_df
    else:
        logger.error("Image Cover not selected for wedding")
        return df, None, None


def choose_good_non_wedding_images(df, number_of_images, logger):
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
        selected_images_df = df.nlargest(number_of_images, 'image_order')

    # Find images that contain all unique people IDs
    selected_images_df = df[
        df['persons_ids'].apply(lambda x: set(x).issuperset(unique_people_ids) if isinstance(x, list) else False)]

    if selected_images_df.empty:
        logger.warning("Warning: No images found containing all unique people IDs. selectign based on max faces")
        max_faces = df['n_faces'].max()
        selected_images_df = df[df['n_faces'] == max_faces]
        # return df, [], pd.DataFrame()

    # Select the top two images with the highest image_order
    selected_images_df = selected_images_df.nlargest(number_of_images, 'image_order')

    if selected_images_df.empty:
        logger.warning("Warning: No images selected based on image_order.")
        return df, [], pd.DataFrame()

    # Extract image IDs
    selected_image_ids = selected_images_df['image_id'].tolist()

    # Remove selected images from the original DataFrame
    df_without_selected = df[~df['image_id'].isin(selected_image_ids)]

    logger.info(f"Selected cover images: {selected_image_ids}")

    return df_without_selected, selected_image_ids, selected_images_df


def generate_first_last_pages(message, df, logger):
    first_last_pages_data_dict = dict()
    for page_type in ['firstPage', 'lastPage']:
        if not message.pagesInfo.get(page_type):
            continue
        layouts_sizes = [layout["number of boxes"] for key, layout in message.designsInfo[f'{page_type}_layouts_df'].iterrows()]
        number_of_images = min(layouts_sizes)

        if message.content.get('is_wedding', True):
            df, first_last_images_ids, first_last_imgs_df = choose_good_wedding_images(df, number_of_images, logger)
        else:
            df, first_last_images_ids, first_last_imgs_df = choose_good_non_wedding_images(df, number_of_images, logger)

        cur_design_id = get_design_id(message.designsInfo[f'{page_type}_layouts_df'], number_of_images, logger)

        first_last_pages_data_dict[page_type] = {
            'design_id': cur_design_id,
            'images_ids': first_last_images_ids,
            'images_df': first_last_imgs_df
        }

    return df, first_last_pages_data_dict
