import random
from utils.configs import CONFIGS


def get_design_id(layout_df, number_of_boxes, logger):
    # Get all layout keys where "number of boxes" == 1
    img_layouts = [key for key, layout in layout_df.iterrows() if layout["number of boxes"] == number_of_boxes]

    # Ensure there are at least 2 valid layouts
    if len(img_layouts) < 1:
        logger.error("Not enough layouts with one image to select two distinct ones.")
        return None

    return img_layouts[0]

def get_important_imgs(data_df, top=3):
    first_page_ids = []
    last_page_ids = []

    first_cover_queries = [
        "an intimate portrait of just the bride and groom",
        "creative and artistic wedding portrait of the bride and groom",
        "a private moment captured between the bride and groom on their wedding day"
    ]

    filtered = data_df[
        (data_df["cluster_context"] == "bride and groom") &
        (data_df["image_subquery_content"].isin(first_cover_queries)) &
        (data_df["persons_ids"].apply(lambda x: isinstance(x, list) and len(x) == 2))&
        (data_df["number_bodies"] == 2)
        ]

    if len(filtered) == 0:
        q_2= ['Smiling bride and groom are Only in the photo','bride and groom holding hands', 'bride and groom with a fantastic standing looking to each other with beautiful scene','bride and groom ONLY with beautiful background']
        filtered = data_df[
            (data_df["cluster_context"] == "bride and groom") &
            (data_df["image_subquery_content"].isin(q_2)) &
            (data_df["persons_ids"].apply(lambda x: isinstance(x, list) and len(x) == 2)) &
            (data_df["number_bodies"] == 2)
            ]

    ids = filtered.sort_values(by='image_order', ascending=True)['image_id'].tolist()

    if len(ids) >= top:
        first_page_ids.extend(ids[:top])
    else:
        # Strategy 2: Any image with "bride and groom" context
        filtered = data_df[
            (data_df["cluster_context"] == "bride and groom") &
            (data_df["persons_ids"].apply(lambda x: isinstance(x, list) and len(x) == 2)) &
            (data_df["number_bodies"] == 2)
            ]
        ids = filtered.sort_values(by='image_order', ascending=True)["image_id"].tolist()
        if len(ids) >= top:
            first_page_ids.extend(ids[:top])
        else:
            # Strategy 3: Any image with "bride" in the main query
            filtered = data_df[data_df["cluster_context"] == "bride"]
            ids = filtered.sort_values(by='image_order', ascending=True)['image_id'].tolist()
            if len(ids) >= top:
                first_page_ids.extend(ids[:top])
            else:
                first_page_ids = data_df.head(top)['image_id'].tolist()

    keywords = ["bride and groom", "groom and bride","groom and brides dancing together solo"]
    queries_not_choose = ["waiting", "posing"]
    df_sorted = data_df.sort_values(by="image_time_date", ascending=False)

    for row in df_sorted.itertuples(index=False):
        text = str(row.image_subquery_content).lower()
        persons_ids = str(row.persons_ids).lower()

        # Check if any keyword is in the text
        has_keyword = any(k.lower() in text for k in keywords)

        # Check if none of the excluded keywords are in text or persons_text
        has_no_excluded = all(q not in text for q in queries_not_choose)

        if has_keyword and has_no_excluded:
            if row.image_id not in first_page_ids:
                last_page_ids.append(row.image_id)

            if len(last_page_ids) >= top:
                break

    if len(last_page_ids) < 1:
        last_q = [
            "bride and groom sharing a heartfelt laugh",
            "bride and groom alone together after the wedding ceremony"
        ]
        filtered = data_df[
            (df_sorted["image_subquery_content"].isin(last_q))
            ]
        ids = filtered.sort_values(by='image_order', ascending=True)['image_id'].tolist()
        last_page_ids.extend([id for id in ids if id not in first_page_ids])

    return first_page_ids, last_page_ids

def choose_good_wedding_images(df, number_of_images, logger):
    first_page_ids, last_page_ids = get_important_imgs(df, top=CONFIGS['top_imges_for_cover'])

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
        logger.warning("Warning: No images found containing all unique people IDs. selectign based on max faces")
        max_faces = df['n_faces'].max()
        selected_images_df = df[df['n_faces'] == max_faces]
        # return df, [], pd.DataFrame()

    # Select the top two images with the highest image_order
    selected_images_df = selected_images_df.nlargest(number_of_images, 'image_order')

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
            df, first_images_ids, first_imgs_df, last_images_ids, last_imgs_df = choose_good_non_wedding_images(df, number_of_images, logger)

        cur_design_id = get_design_id(message.designsInfo[f'{page_type}_layouts_df'], number_of_images, logger)

        first_last_pages_data_dict[page_type] = {
            'design_id': cur_design_id,
            'first_images_ids': first_images_ids,
            'first_images_df': first_imgs_df,
            'last_images_ids': last_images_ids,
            'last_images_df': last_imgs_df,

        }

    return df, first_last_pages_data_dict
