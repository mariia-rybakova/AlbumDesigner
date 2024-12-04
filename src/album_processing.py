import math
import time
import numpy as np
import psutil
import statistics
import pandas as pd

from utils import get_photos_from_db,generate_filtered_multi_spreads,add_ranking_score, get_layouts_data, get_important_imgs, get_cover_img, get_cover_layout,generate_json_response, process_illegal_groups
from utils.clusters_labels import label_list
from utils.load_layouts import load_layouts
from utils.time_proessing import process_image_time
from utils.plotting_results import  plot_album
from utils.clustering_time import cluster_by_time
from utils.time_outliers import handle_edited_time
from utils.process_auto_groups   import process_auto_groups

from utils.lookup_table import genreate_look_up,lookup_table

def sort_groups_by_photo_time(data_dict,logger):
    def get_mean_time(sublist):
        # Extract all general_time values from the photos in the sublist
        total_spread_time = []
        for page in sublist[1:3]:
            time = [photo.general_time for photo in page]
            total_spread_time.extend(time)

        if total_spread_time:
            #logger.info(f"layout id {sublist[0]}, average time, {statistics.mean(total_spread_time)}")
            return statistics.median(total_spread_time)
        else:
            return float('inf')

    # Iterate over each group in the dictionary
    for group_key in data_dict:
        data_dict[group_key][0].sort(key=get_mean_time)

    return data_dict


def get_ranking(images_dict):
    image_id2rank_score = dict()
    for idx, (image_id, info_dict) in enumerate(images_dict.items()):
        image_id2rank_score[image_id] = info_dict['ranking']

    return image_id2rank_score


def sort_sub_groups(sub_grouped, group_names):
    # Define the priority list for secondary sorting
    priority_list = ["bride getting dressed", "getting hair-makeup", "bride", "groom getting dress", "groom",
                     "portrait",
                     "bride and groom", "walking the aisle", "ceremony", "settings", "dancing"]

    # Create a dictionary to map group names to their priority
    priority_dict = {name: i for i, name in enumerate(priority_list)}

    time_group_dict = {}
    for group_name in group_names:
        orig_group_name = group_name.split('*')
        parts = orig_group_name[0].split('_')
        group_id = (float(parts[0]), '_'.join(parts[1:]))
        group = sub_grouped.get_group(group_id)

        # Calculate the median instead of the mean
        #group_time_median = group["edited_general_time"].median()
        group_time_median = group["general_time"].median()
        #group_time_median = group["image_orderInScene"].median()
        time_group_dict[group_name] = group_time_median

    sorted_time_groups = dict(
        sorted(time_group_dict.items(),
               key=lambda item: (item[1],
                                 priority_dict.get(item[0].split('_')[1].split('*')[0], float('inf'))))
    )

    return sorted_time_groups


def get_images_per_group(data_df):
    """
    Return: dict group_name to list of images data
    """
    group2images_data_list = dict()
    grouped_by_content = data_df.groupby('time_cluster')
    #grouped_by_content = data_df.groupby('scene_order')
    for main_content_cluster, group_df in grouped_by_content:
        sub_grouped = group_df.groupby('cluster_context')
        for sub_cluster, sub_group_df in sub_grouped:
            num_images = len(sub_group_df)
            group2images_data_list[f'{main_content_cluster}_{sub_cluster}'] = num_images
    return group2images_data_list


def genreate_look_up(group2images):
    for group_name, images in group2images.items():
        parts = group_name.split('_')
        group_id = parts[1]
        if group_id in lookup_table:
            # mean = calculate_flexible_mean(images,lookup_table[group_id][0] )
            # lookup_table[group_name] = (mean,lookup_table[group_id][1])
            continue
        else:
            lookup_table[group_name] = (5, 0.2)
    return lookup_table

def gallery_processing(data_df, layouts_df,is_auto, logger):
    logger.info(f'============================')
    logger.info(f'Start Processing the Gallery')

    ERROR = None

    group2images = get_images_per_group(data_df)
    sub_grouped = data_df.groupby(['time_cluster', 'cluster_context'])

    start_time = time.time()

    if is_auto:
        updated_sub_grouped, group2images, lookup_table = process_auto_groups(sub_grouped)
    else:
        updated_sub_grouped, group2images, lookup_table = process_illegal_groups(group2images, sub_grouped, logger)

    illegal_time = (time.time() - start_time) / 60
    logger.info(f'Illegal groups processing time: {illegal_time:.2f} minutes')

    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()

    logger.info(f"CPU Usage AFTER Illegal Processing: {cpu_usage}%")
    logger.info(f"Memory Usage AFTER Illegal Processing: {memory_info.percent}%")

    layout_id2data = get_layouts_data(layouts_df)

    group_name2chosen_combinations = dict()
    for group_name in group2images.keys():
        logger.info("Starting with group_name {}".format(group_name))

        parts = group_name.split('_')
        group_id = (float(parts[0]), '_'.join(parts[1:]))
        try:
            group_images_df = updated_sub_grouped.get_group(group_id)

            cur_group_photos = get_photos_from_db(group_images_df)

            spread_params = list(lookup_table.get(parts[1], (10, 1.5)))

            cur_group_photos_list = list()
            if (len(cur_group_photos) / (spread_params[0] - 2*spread_params[1]) >= 4 or
               # len(cur_group_photos) / spread_params[0] >= 3 and len(cur_group_photos) > 11 or
                len(cur_group_photos) / (spread_params[0] - 2*spread_params[1]) < 3 and len(cur_group_photos) > 24):
                split_size = min(spread_params[0] * 3, max(spread_params[0], 11))
                number_of_splits = math.ceil(len(cur_group_photos) / split_size)
                logger.info('Using splitting to {} parts'.format(number_of_splits))
                for split_num in range(number_of_splits):
                    cur_group_photos_list.append(cur_group_photos[
                                                 split_num * split_size: min((split_num + 1) * split_size,
                                                                             len(cur_group_photos))])
            else:
                cur_group_photos_list.append(cur_group_photos)

            for idx, group_photos in enumerate(cur_group_photos_list):
                filter_start = time.time()

                logger.info('Photos: {}'.format([[item.id, item.ar, item.color, item.rank, item.photo_class, item.cluster_label, item.general_time]
                     for item in group_photos]))
                filtered_spreads = generate_filtered_multi_spreads(group_photos, layouts_df, spread_params,logger)
                if filtered_spreads is None:
                    continue
                logger.info('Filtered spreads size: {}'.format(len(filtered_spreads)))
                logger.info('Filtered spreads time: {}'.format(time.time() - filter_start))

                ranking_start = time.time()
                filtered_spreads = add_ranking_score(filtered_spreads, group_photos, layout_id2data)
                filtered_spreads = sorted(filtered_spreads, key=lambda x: x[1], reverse=True)
                logger.info('Ranking time: {}'.format(time.time() - ranking_start))

                best_spread = filtered_spreads[0]
                cur_spreads = best_spread[0]
                for spread_id, spread in enumerate(cur_spreads):
                    best_spread[0][spread_id][1] = set([group_photos[photo_id] for photo_id in spread[1]])
                    best_spread[0][spread_id][2] = set([group_photos[photo_id] for photo_id in spread[2]])

                group_name2chosen_combinations[group_name + '*' + str(idx)] = best_spread
                logger.info('{} results:'.format(group_name + '*' + str(idx)))
                logger.info(group_name2chosen_combinations[group_name + '*' + str(idx)])

                del filtered_spreads

            logger.info("############################################################")

        except Exception as e:
            logger.error("Theres Error with group_name {}".format(group_name), e)
            ERROR = "Theres Error with group_name {}".format(group_name), e
            continue

    logger.info("Album Generation Finished ^_^")
    return group_name2chosen_combinations, updated_sub_grouped, ERROR


def map_cluster_label(cluster_label):
    if cluster_label == -1:
        return "None"
    elif cluster_label >= 0 and cluster_label < len(label_list):
        return label_list[cluster_label]
    else:
        return "Unknown"


def create_automatic_album(images_data_dict, layouts_path,gallery_path,relation_type,is_auto, logger=None):
    # Start time
    logger.info("Start creating album...")
    layouts_df = load_layouts(layouts_path)

    start_time = time.time()
    data_df = pd.DataFrame.from_dict(images_data_dict, orient='index')
    # Convert the index to a column
    data_df['image_id'] = data_df.index
    # Reset the index to a numerical range
    data_df.reset_index(drop=True, inplace=True)
    # Optionally reorder columns if you want 'image_id' to be the first column
    data_df = data_df[['image_id'] + [col for col in data_df.columns if col != 'image_id']]
    data_df.fillna(value='None', inplace=True)
    sorted_ranking_df = data_df.sort_values(by="image_order", ascending=False)
    # convert cluster class number into text value and make copy when we need merging
    # apply the mapping function to create the new cluster_context column
    sorted_ranking_df["cluster_context"] = sorted_ranking_df["cluster_class"].apply(map_cluster_label)
    # copy column to make edit when we merge
    sorted_ranking_df["cluster_context_2nd"] = sorted_ranking_df['cluster_context']

    # get most import images for cover image
    bride_groom_highest_images = get_important_imgs(sorted_ranking_df, top=50)

    # if we didn't find the highest ranking images then we won't be able to get cover image
    if len(bride_groom_highest_images) > 0:
        # get cover image and remove it from dataframe
        data_df, cover_img = get_cover_img(sorted_ranking_df, bride_groom_highest_images)
        # Get layout for cover image
        cover_img_layout = get_cover_layout(layouts_df)

        sorted_by_time_df, image_id2general_time = process_image_time(data_df)

        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()

        logger.info(f"CPU Usage before gallery processing: {cpu_usage}%")
        logger.info(f"Memory Usage before gallery processing: {memory_info.percent}%")

        #df = handle_edited_time(sorted_by_time_df)
        df = cluster_by_time(sorted_by_time_df)

        group_name2chosen_combinations, sub_groups, error = gallery_processing(df, layouts_df,is_auto, logger)

        comb_generation_time = (time.time() - start_time) / 60
        logger.info(f'Combination generation time: {comb_generation_time:.2f} minutes')

        sorted_group_name2chosen_combinations = sort_groups_by_photo_time(group_name2chosen_combinations, logger)
        sorted_sub_groups = sort_sub_groups(sub_groups, sorted_group_name2chosen_combinations.keys())

        result = generate_json_response(cover_img, cover_img_layout,sub_groups, sorted_sub_groups, sorted_group_name2chosen_combinations, layouts_df, logger)
        output_save_path = r'results/new_galles'
        plot_album(cover_img, cover_img_layout, sub_groups, sorted_sub_groups, group_name2chosen_combinations,
                   layouts_df, logger, gallery_path, output_save_path,relation_type)
        # End time
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        logger.info(f"Elapsed time: {elapsed_time:.2f} minutes")
        return result, error
    else:
        logger.error(f"Cant process the album without cover image")
        print("Cant process the album without cover image of bride and groom")
        return None,"No Cover image found"

# if __name__ == '__main__':
#     layouts_file_path = r'C:\Users\karmel\Desktop\PicTime\Projects\AlbumDesigner\layouts.csv'
#     layouts_df = load_layouts(layouts_file_path)
#     with open("my_dict.pkl", "rb") as file:
#         loaded_dict = pickle.load(file)
#         # image_time  image_as image_orderInScene image_orientation scene_order
#     create_automatic_album(loaded_dict, layouts_df)
