import random

def get_cover_img(data_df, important_imgs):
    # Ensure there are at least 2 images to choose from
    if len(important_imgs) < 2:
        raise ValueError("important_imgs must contain at least 2 distinct images.")

    # Select 2 distinct images
    cover_img_ids = random.sample(important_imgs, 2)

    # Get rows corresponding to selected images
    cover_image_df = data_df[data_df['image_id'].isin(cover_img_ids)]

    # Remove selected images from main dataframe
    df = data_df[~data_df['image_id'].isin(cover_img_ids)]

    return df, cover_img_ids, cover_image_df

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

    return image_id_list

