from utils.configs import CONFIGS
from utils.selection.non_wedding_selection_tools import generate_dict_key,get_appearance_percentage,select_images_of_group,select_images_of_one_person

def smart_non_wedding_selection(df, logger):
    # convert df to dict
    if df is None:
        logger.info("the dataframe is empty! for non wedding selection")
        return None, "Error fetching the data"

    total_images = len(df)

    if total_images <= CONFIGS['small_gallery_number']:
        selected_images = df['image_id'].values.tolist()
        return selected_images, None

    df['people_cluster'] = df.apply(lambda row: generate_dict_key(row['persons_ids'], row['number_bodies']), axis=1)

    photos_info_dict = df.set_index('image_id').to_dict(orient='index')
    images_ids = list(photos_info_dict.keys())
    clusters_class_imgs = df.groupby("cluster_label")["image_id"].apply(list).to_dict()
    persons_images_clustering = df.groupby('people_cluster')['image_id'].apply(list).to_dict()

    persons_percentage = get_appearance_percentage(persons_images_clustering, total_images)

    # cover image selection for non wedding gallery
    count_percentage = 0

    for percent_person in persons_percentage.keys():
        if persons_percentage[percent_person] >= CONFIGS['person_count_percentage']:
            count_percentage += 1

    if count_percentage == 1:
        # one person gallery
        auto_selected_images = select_images_of_one_person(images_ids, photos_info_dict, logger)
    else:
        # more than one person gallery
        auto_selected_images = select_images_of_group(persons_images_clustering, photos_info_dict, clusters_class_imgs,
                                                      logger)

    return auto_selected_images, None
