import random
import pandas as pd
from utils.configs import CONFIGS


def get_design_id(layout_df, number_of_boxes, logger):
    # Get all layout keys where "number of boxes" == 1
    img_layouts = [key for key, layout in layout_df.iterrows() if layout["number of boxes"] == number_of_boxes]

    # Ensure there are at least 2 valid layouts
    if len(img_layouts) < 1:
        logger.error("Not enough layouts with one image to select two distinct ones.")
        return None

    return img_layouts[0]

def get_important_imgs(data_df,logger, top=3):
    try:
        first_page_ids = []
        last_page_ids = []
        bride_id = data_df["bride_id"].values[0]
        groom_id = data_df["groom_id"].values[0]

        first_cover_queries = [
            'bride and groom smiling at each other',
            'bride and groom posing for a portrait'
        ]
        fallback_first_queries = [
            'bride and groom during the ceremony',
            'bride and groom kissing'
        ]
        last_cover_queries = [
            'bride and groom dancing',
            'bride and groom smiling at each other'
        ]

        first_filtered_1 = data_df[
            (data_df["cluster_context"] == "bride and groom") &
            (data_df["image_subquery_content"].isin(first_cover_queries)) &
            (data_df["persons_ids"].apply(
        lambda x: isinstance(x, list) and bride_id in x and groom_id in x
           ))&
            (data_df["number_bodies"] == 2) &  (data_df["image_orientation"] == "landscape")
            ]

        if first_filtered_1.empty:
            logger.info("We could'nt find a good  first cover from query 1")
            first_filtered_1 = data_df[
                (data_df["cluster_context"] == "bride and groom") &
                (data_df["image_subquery_content"].isin(fallback_first_queries)) &
                (data_df["persons_ids"].apply(lambda x: isinstance(x, list) and len(x) == 2)) &
                (data_df["number_bodies"] == 2) & (data_df["image_orientation"] == "landscape")
                ]

        if first_filtered_1.empty:
            first_filtered_1 = data_df[
                (data_df["cluster_context"] == "bride and groom") &
                (data_df["persons_ids"].apply(lambda x: isinstance(x, list) and len(x) == 2)) &
                (data_df["number_bodies"] == 2) & (data_df["image_orientation"] == "landscape")
                ]

        ids = first_filtered_1.sort_values(by='image_order', ascending=True)['image_id'].tolist()
        for i in range(top):
            if i < len(ids) - 1:
                break
            first_page_ids.extend(ids[:i])

        # second cover image
        df_sorted = data_df.sort_values(by="image_time_date", ascending=False)

        second_filtered = data_df[
            (df_sorted["image_subquery_content"].isin(last_cover_queries)) &
            data_df["persons_ids"].apply(
                lambda x: isinstance(x, list) and bride_id in x and groom_id in x
            )
            ]

        if second_filtered.empty:
            if len(first_filtered_1) <= top:
                second_filtered = data_df[
                    (data_df["cluster_context"] == "bride and groom")
                    ]
                ids = second_filtered["image_id"].tolist()
                last_page_ids.extend([id for id in ids if id not in first_page_ids])
        else:
            #ids = second_filtered.sort_values(by='image_order', ascending=True)['image_id'].tolist()
            ids = second_filtered['image_id'].tolist()
            last_page_ids.extend([id for id in ids if id not in first_page_ids][:top])
            if len(last_page_ids) < 1:
                 if len(first_filtered_1) !=0:
                     ids = first_filtered_1["image_id"].tolist()
                     last_page_ids.extend([id for id in ids if id not in first_page_ids])

    except Exception as e:
        logger.error(f"Error inside the function get_important_imgs {e}")
        return None, None

    return first_page_ids, last_page_ids

def choose_good_wedding_images(df, number_of_images, logger):
    first_page_ids, last_page_ids = get_important_imgs(df,logger, top=CONFIGS['top_imges_for_cover'])

    # if we didn't find the highest ranking images then we won't be able to get cover image
    if len(first_page_ids) >= number_of_images and len(last_page_ids) >= number_of_images:
        # Select 2 distinct images
        first_cover_img_ids = random.sample(first_page_ids, number_of_images)
        last_cover_img_ids = random.sample(last_page_ids, number_of_images)

        # Get rows corresponding to selected images
        first_cover_image_df = df[df['image_id'].isin(first_cover_img_ids)]
        last_cover_image_df = df[df['image_id'].isin(last_cover_img_ids)]

        # Remove selected images from main dataframe
        df = df[~df['image_id'].isin(first_cover_img_ids+last_cover_img_ids)]


        return df, first_cover_img_ids, first_cover_image_df, last_cover_img_ids, last_cover_image_df
    else:
        logger.error("Image Cover not selected for wedding")
        return df, None, None, None, None


def choose_good_non_wedding_images(df, number_of_images, logger):
    # Validate input DataFrame
    required_columns = {'persons_ids', 'image_order', 'image_id'}

    if not required_columns.issubset(df.columns):
        missing_cols = required_columns - set(df.columns)
        logger.error(f"Error: DataFrame is missing required columns: {missing_cols}")
        return df, None, None, None, None

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
        logger.warning("No images matched. Selecting top by n_faces.")
        # Take top-N by number of faces
        selected_images_df = df.nlargest(number_of_images, 'n_faces')
    else:
        # Keep your current priority by image_order first
        selected_images_df = selected_images_df.nlargest(number_of_images, 'image_order')

        # If still fewer than required, fill the remaining by highest n_faces from the whole df (excluding already chosen)
        if len(selected_images_df) < number_of_images:
            remaining = number_of_images - len(selected_images_df)
            chosen_ids = set(selected_images_df['image_id'].tolist())
            fill_pool = df[~df['image_id'].isin(chosen_ids)]
            fill_df = fill_pool.nlargest(remaining, 'n_faces')

            selected_images_df = pd.concat([selected_images_df, fill_df], ignore_index=False).drop_duplicates(subset='image_id').head(number_of_images)

    if selected_images_df.empty:
        logger.warning("Warning: No images selected based on image_order.")
        return df, None, None, None, None

    # Extract image IDs
    selected_image_ids = selected_images_df['image_id'].tolist()
    mid_index = len(selected_image_ids) // 2
    first_image_id = selected_image_ids[:mid_index]
    last_image_id = selected_image_ids[mid_index:]

    first_image_df = df[df['image_id'].isin(first_image_id)]
    last_image_df = df[df['image_id'].isin(last_image_id)]

    # Remove selected images from the original DataFrame
    df_without_selected = df[~df['image_id'].isin(selected_image_ids)]

    logger.info(f"Selected cover images: {selected_image_ids}")

    return df_without_selected, first_image_id, last_image_id, first_image_df, last_image_df


def generate_first_last_pages(message, df, logger):
    first_last_pages_data_dict = dict()
    for page_type in ['firstPage', 'lastPage']:
        if not message.pagesInfo.get(page_type):
            continue
        layouts_sizes = [layout["number of boxes"] for key, layout in message.designsInfo[f'{page_type}_layouts_df'].iterrows()]
        number_of_images = min(layouts_sizes)

        if message.content.get('is_wedding', True):
            df, first_images_ids, first_imgs_df, last_images_ids, last_imgs_df = choose_good_wedding_images(df, number_of_images, logger)
        else:
            df, first_images_ids, last_images_ids, first_imgs_df, last_imgs_df = choose_good_non_wedding_images(df, 2*number_of_images, logger)

        cur_design_id = get_design_id(message.designsInfo[f'{page_type}_layouts_df'], number_of_images, logger)

        first_last_pages_data_dict[page_type] = {
            'design_id': cur_design_id,
            'first_images_ids': first_images_ids,
            'first_images_df': first_imgs_df,
            'last_images_ids': last_images_ids,
            'last_images_df': last_imgs_df,

        }

    return df, first_last_pages_data_dict
