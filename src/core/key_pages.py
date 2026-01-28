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


def _pick_cover_subset(df, position="first", window_size=10):
    """
    Returns a subset DataFrame for the cover selection:
      - If 'time_cluster' exists: all rows at min/max cluster.
      - Else if 'time' exists: first/last N rows after sorting by 'time' asc.
      - Else: first/last N rows by current order.
    """
    if df.empty:
        return df

    if "time_cluster" in df.columns:
        tc = df["time_cluster"].min() if position == "first" else df["time_cluster"].max()
        return df[df["time_cluster"] == tc].copy()

    if "image_time" in df.columns:
        # Ensure numeric (don’t convert to datetime)
        tmp = df.copy()
        tmp["__t"] = pd.to_numeric(tmp["image_time"], errors="coerce")
        tmp = tmp.sort_values("__t", ascending=True)
        subset = tmp.head(window_size) if position == "first" else tmp.tail(window_size)
        return subset.drop(columns="__t")

    # Fallback: by current order
    return (df.head(window_size) if position == "first" else df.tail(window_size)).copy()


def _select_by_priority_from_subset(df_subset, queries_primary, queries_fallback):
    """
    Try in order:
      1) landscape + primary
      2) landscape + fallback
      3) portrait  + primary
      4) portrait  + fallback
    Returns ordered list of image_ids.
    """
    if df_subset.empty:
        return []

    def pick(orientation, queries):
        sub = df_subset[
            (df_subset["image_orientation"] == orientation) &
            (df_subset["image_subquery_content"].isin(queries))
            ].copy()
        if sub.empty:
            return []
        sub["__q"] = pd.Categorical(sub["image_subquery_content"], categories=queries, ordered=True)
        sub = sub.sort_values("__q")
        return sub["image_id"].tolist()

    for orientation, queries in [
        ("landscape", queries_primary),
        ("landscape", queries_fallback),
        ("portrait", queries_primary),
        ("portrait", queries_fallback),
    ]:
        ids = pick(orientation, queries)
        if ids:
            return ids

    picked = df_subset[
        (df_subset["image_subquery_content"].isin(["unknown_bride_and_groom"]))
    ].copy()

    if picked.empty:
        return []
    else:
        return picked["image_id"].tolist()


def get_important_imgs(data_df, bride_groom_df, logger):
    try:
        if bride_groom_df is not None:
            chosen_df = data_df.copy()
            if not bride_groom_df.empty:
                chosen_df = bride_groom_df.copy()
        else:
            chosen_df = data_df.copy()

        bride_id = chosen_df["bride_id"].values[0]
        groom_id = chosen_df["groom_id"].values[0]

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
        base = chosen_df[
            (chosen_df["cluster_context"] == "bride and groom") &
            (chosen_df["persons_ids"].apply(lambda x: isinstance(x, list) and bride_id in x and groom_id in x)) &
            (chosen_df["n_faces"] > 0)
            ].copy()

        # 2) FIRST cover: earliest time_cluster
        subset_first = _pick_cover_subset(base, position="first", window_size=10)
        first_page_ids = _select_by_priority_from_subset(
            subset_first, first_cover_queries, fallback_first_queries
        )

        # 3) LAST cover: latest time_cluster
        subset_last = _pick_cover_subset(base, position="last", window_size=10)
        last_page_ids = _select_by_priority_from_subset(
            subset_last, last_cover_queries, fallback_first_queries
        )
        if len(first_page_ids) == 0 or len(last_page_ids) == 0:
            logger.error("No images selected for Cover images.")
            return [], []

        return [first_page_ids[0]], [last_page_ids[0]]

    except Exception as e:
        logger.error(f"Error inside the function get_important_imgs {e}")
        return None, None


def choose_good_wedding_images(df, bride_groom_df, logger):
    first_page_ids, last_page_ids = get_important_imgs(df, bride_groom_df, logger)

    if bride_groom_df is not None:
        if not bride_groom_df.empty:
            first_cover_image_df = bride_groom_df[bride_groom_df['image_id'].isin(first_page_ids)]
            last_cover_image_df = bride_groom_df[bride_groom_df['image_id'].isin(last_page_ids)]
        else:
            first_cover_image_df = df[df['image_id'].isin(first_page_ids)]
            last_cover_image_df = df[df['image_id'].isin(last_page_ids)]
    else:
        # Get rows corresponding to selected images
        first_cover_image_df = df[df['image_id'].isin(first_page_ids)]
        last_cover_image_df = df[df['image_id'].isin(last_page_ids)]

    # Remove selected images from main dataframe
    df = df[~df['image_id'].isin(first_page_ids + last_page_ids)]

    return df, first_page_ids, first_cover_image_df, last_page_ids, last_cover_image_df


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

            selected_images_df = pd.concat([selected_images_df, fill_df], ignore_index=False).drop_duplicates(
                subset='image_id').head(number_of_images)

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

    if message.pagesInfo.get("firstPage"):
        if message.content.get('is_wedding', True):
            df, first_images_ids, first_imgs_df, last_images_ids, last_imgs_df = choose_good_wedding_images(df,
                                                                                                            message.content.get(
                                                                                                                'bride and groom'),
                                                                                                            logger)
        else:
            df, first_images_ids, last_images_ids, first_imgs_df, last_imgs_df = choose_good_non_wedding_images(df, 1,
                                                                                                                logger)

        if message.pagesInfo.get("firstPage"):
            layouts_df = message.designsInfo[f"firstPage_layouts_df"]
            if not first_imgs_df.empty:
                if first_imgs_df["image_orientation"].values[0] == "landscape":
                    cover_layouts = [key for key, layout in layouts_df.iterrows() if layout["max landscapes"] == 1 and (
                                layout['right_large_landscape'] == 1 or layout["left_large_landscape"] == 1 or layout[
                            "left_large_square"] == 1 or layout["right_large_square"] == 1)]
                    design_id = cover_layouts[0]
                else:
                    cover_layouts = [key for key, layout in layouts_df.iterrows() if layout["max portraits"] == 1 and (
                                layout['left_large_portrait'] == 1 or layout["right_large_portrait"] == 1 or layout[
                            "left_large_square"] == 1 or layout["right_large_square"] == 1)]
                    design_id = cover_layouts[0]

                first_last_pages_data_dict["firstPage"] = {
                    'design_id': design_id,
                    'first_images_ids': first_images_ids,
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
                        "left_large_square"] == 1 or layout["right_large_square"] == 1)]
                    design_id = cover_layouts[0]
                else:
                    cover_layouts = [key for key, layout in layouts_df.iterrows() if layout["max portraits"] == 1 and (
                            layout['left_large_portrait'] == 1 or layout["right_large_portrait"] == 1 or layout[
                        "left_large_square"] == 1 or layout["right_large_square"] == 1)]
                    design_id = cover_layouts[0]

                first_last_pages_data_dict['lastPage'] = {
                    'design_id': design_id,
                    'last_images_ids': last_images_ids,
                    'last_images_df': last_imgs_df,

                }
        else:
            logger.warning("For this album theres no last page cover image")

    return df, first_last_pages_data_dict
