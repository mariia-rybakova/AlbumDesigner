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


def _pick_time_cluster(df, position="first"):
    if df.empty:
        return None
    return df["time_cluster"].min() if position == "first" else df["time_cluster"].max()

def _select_by_priority(df, queries_primary, queries_fallback, time_cluster_value):
    """
    Priority:
      1) landscape @ time_cluster_value with queries_primary
      2) landscape @ time_cluster_value with queries_fallback
      3) portrait  @ time_cluster_value with queries_primary
      4) portrait  @ time_cluster_value with queries_fallback
    Returns a list of image_ids (ordered by query priority).
    """
    if time_cluster_value is None:
        return []

    def pick(orientation, queries):
        sub = df[
            (df["image_orientation"] == orientation) &
            (df["time_cluster"] == time_cluster_value) &
            (df["image_subquery_content"].isin(queries))
        ].copy()

        if sub.empty:
            return []

        # Sort by query priority
        sub["q_order"] = pd.Categorical(
            sub["image_subquery_content"], categories=queries, ordered=True
        )
        sub = sub.sort_values(["q_order"])
        return sub["image_id"].tolist()

    # Try in order
    for orientation, queries in [
        ("landscape", queries_primary),
        ("landscape", queries_fallback),
        ("portrait",  queries_primary),
        ("portrait",  queries_fallback),
    ]:
        picked = pick(orientation, queries)
        if picked:
            return picked

    picked = df[
        (df["time_cluster"] == time_cluster_value) &
        (df["image_subquery_content"].isin(["unknown_bride_and_groom"]))
        ].copy()

    if picked.empty:
        return []
    else:
        return picked["image_id"].tolist()


def get_important_imgs(data_df,logger, top=3):
    try:
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

        ############################################################
        base = data_df[
            (data_df["cluster_context"] == "bride and groom") &
            (data_df["persons_ids"].apply(lambda x: isinstance(x, list) and bride_id in x and groom_id in x))
            ].copy()

        # 2) FIRST cover: earliest time_cluster
        first_tc = _pick_time_cluster(base, position="first")
        first_page_ids = _select_by_priority(
            base,
            queries_primary=first_cover_queries,
            queries_fallback=fallback_first_queries,
            time_cluster_value=first_tc,
        )

        # 3) LAST cover: latest time_cluster
        last_tc = _pick_time_cluster(base, position="last")
        last_page_ids = _select_by_priority(
            base,
            queries_primary=last_cover_queries,
            queries_fallback=fallback_first_queries,  # reuse same fallback list if desired
            time_cluster_value=last_tc,
        )
        return first_page_ids[0], last_page_ids[0]

    except Exception as e:
        logger.error(f"Error inside the function get_important_imgs {e}")
        return None, None

    return first_page_ids, last_page_ids

def choose_good_wedding_images(df, number_of_images, logger):
    first_page_ids, last_page_ids = get_important_imgs(df,logger, top=CONFIGS['top_imges_for_cover'])

    # if we didn't find the highest ranking images then we won't be able to get cover image
    # if len(first_page_ids) >= number_of_images and len(last_page_ids) >= number_of_images:
    #     # Select 2 distinct images
    #     first_cover_img_ids = random.sample(first_page_ids, number_of_images)
    #     last_cover_img_ids = random.sample(last_page_ids, number_of_images)

    # Get rows corresponding to selected images
    first_cover_image_df = df[df['image_id'].isin([first_page_ids])]
    last_cover_image_df = df[df['image_id'].isin([last_page_ids])]

    # Remove selected images from main dataframe
    df = df[~df['image_id'].isin([first_page_ids]+[last_page_ids])]


    return df, first_page_ids, first_cover_image_df, last_page_ids, last_cover_image_df
    # else:
    #     logger.error("Image Cover not selected for wedding")
    #     return df, None, None, None, None


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
    # PAGE_KEYS = ("firstPage", "lastPage")
    # # 1) Gather per-page layout metadata (min images and design_id)
    # page_meta = {}
    # for key in PAGE_KEYS:
    #     if not message.pagesInfo.get(key):
    #         continue
    #     layouts_df = message.designsInfo[f"{key}_layouts_df"]
    #     # minimal number of images required by any layout for that page
    #     n_images = int(layouts_df["number of boxes"].min())
    #     cover_layouts = [key for key, layout in layouts_df.iterrows() if layout["number of boxes"] == n_images]
    #
    #     if cover_layouts:
    #         design_id = cover_layouts[0]
    #     else:
    #         design_id = None
    #
    #     page_meta[key] = {"n_images": n_images, "design_id": design_id}

    if message.pagesInfo.get("firstPage"):
        if message.content.get('is_wedding', True):
            df, first_images_ids, first_imgs_df, last_images_ids, last_imgs_df = choose_good_wedding_images(df,1, logger)
        else:
            df, first_images_ids, last_images_ids, first_imgs_df, last_imgs_df = choose_good_non_wedding_images(df, 1, logger)

        if message.pagesInfo.get("firstPage"):
            layouts_df = message.designsInfo[f"firstPage_layouts_df"]
            if not first_imgs_df.empty:
                if first_imgs_df["image_orientation"].values[0] == "landscape":
                    cover_layouts = [key for key, layout in layouts_df.iterrows() if layout["max landscapes"] == 1 and (layout['right_large_landscape'] == 1 or layout["left_large_landscape"] == 1 or layout[
                            "left_large_square"] == 1 or layout["right_large_square"]  == 1) ]
                    design_id = cover_layouts[0]
                else:
                    cover_layouts = [key for key, layout in layouts_df.iterrows() if layout["max portraits"] == 1 and (layout['left_large_portrait'] == 1 or layout["right_large_portrait"] == 1 or layout["left_large_square"] == 1 or layout["right_large_square"]== 1)]
                    design_id = cover_layouts[0]

                first_last_pages_data_dict["firstPage"]  = {
                    'design_id': design_id,
                    'first_images_ids': [first_images_ids],
                    'first_images_df': first_imgs_df,

                }
        else:
            logger.warning("For this album theres no first page cover image")

        if message.pagesInfo.get('lastPage'):
            layouts_df = message.designsInfo[f"lastPage_layouts_df"]
            # minimal number of images required by any layout for that page
            if not last_imgs_df.empty:
                if last_imgs_df["image_orientation"].values[0] == "landscape":
                    cover_layouts = [key for key, layout in layouts_df.iterrows() if layout["max landscapes"] == 1 and (
                                layout['right_large_landscape'] == 1 or layout["left_large_landscape"] == 1 or layout[
                            "left_large_square"] == 1 or layout["right_large_square"]== 1)]
                    design_id = cover_layouts[0]
                else:
                    cover_layouts = [key for key, layout in layouts_df.iterrows() if layout["max portraits"] == 1 and (
                                layout['left_large_portrait'] == 1 or layout["right_large_portrait"] == 1 or layout[
                            "left_large_square"] == 1 or layout["right_large_square"]== 1)]
                    design_id = cover_layouts[0]

                first_last_pages_data_dict['lastPage'] = {
                        'design_id': design_id,
                        'last_images_ids': [last_images_ids],
                        'last_images_df': last_imgs_df,

                }
        else:
            logger.warning("For this album theres no last page cover image")

    return df, first_last_pages_data_dict
