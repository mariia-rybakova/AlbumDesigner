import math
import time
import numpy as np
import psutil
import statistics
from multiprocessing import Pool
from functools import partial
import pandas as pd

from utils import get_photos_from_db,generate_filtered_multi_spreads,add_ranking_score, get_layouts_data, get_important_imgs, get_cover_img, get_cover_layout,generate_json_response, process_illegal_groups
from utils.lookup_table_tools import get_lookup_table
from utils.album_tools import get_none_wedding_groups,get_wedding_groups,get_images_per_groups

class create_automatic_album:
    def __init__(self,df,layouts_df,layout_id2data,is_wedding, logger):
             self.df = df
             self.is_wedding = is_wedding
             self.logger = logger
             self.original_groups = None
             self.look_up_table = None
             self.group_name2chosen_combinations = dict()
             self.updated_groups = None
             self.layouts_df =layouts_df
             self.layout_id2data = layout_id2data

    def process_groups(self,group_name):
        self.logger.info("Starting with group_name {}".format(group_name))
        try:
            group_images_df =  self.updated_groups.get_group(group_name)
            cur_group_photos = get_photos_from_db(group_images_df)
            spread_params = list(self.look_up_table.get(group_name[1], (10, 1.5)))

            cur_group_photos_list = list()
            if (len(cur_group_photos) / (spread_params[0] - 2 * spread_params[1]) >= 4 or
                    # len(cur_group_photos) / spread_params[0] >= 3 and len(cur_group_photos) > 11 or
                    len(cur_group_photos) / (spread_params[0] - 2 * spread_params[1]) < 3 and len(
                        cur_group_photos) > 24):
                split_size = min(spread_params[0] * 3, max(spread_params[0], 11))
                number_of_splits = math.ceil(len(cur_group_photos) / split_size)
                self.logger.info('Using splitting to {} parts'.format(number_of_splits))
                for split_num in range(number_of_splits):
                    cur_group_photos_list.append(cur_group_photos[
                                                 split_num * split_size: min((split_num + 1) * split_size,
                                                                             len(cur_group_photos))])
            else:
                cur_group_photos_list.append(cur_group_photos)

            for idx, group_photos in enumerate(cur_group_photos_list):
                filter_start = time.time()

                self.logger.info('Photos: {}'.format(
                    [[item.id, item.ar, item.color, item.rank, item.photo_class, item.cluster_label, item.general_time]
                     for item in group_photos]))
                filtered_spreads = generate_filtered_multi_spreads(group_photos, self.layouts_df, spread_params, self.logger)
                if filtered_spreads is None:
                    continue
                self.logger.info('Filtered spreads size: {}'.format(len(filtered_spreads)))
                self.logger.info('Filtered spreads time: {}'.format(time.time() - filter_start))

                ranking_start = time.time()
                filtered_spreads = add_ranking_score(filtered_spreads, group_photos, self.layout_id2data)
                filtered_spreads = sorted(filtered_spreads, key=lambda x: x[1], reverse=True)
                self.logger.info('Ranking time: {}'.format(time.time() - ranking_start))

                best_spread = filtered_spreads[0]
                cur_spreads = best_spread[0]
                for spread_id, spread in enumerate(cur_spreads):
                    best_spread[0][spread_id][1] = set([group_photos[photo_id] for photo_id in spread[1]])
                    best_spread[0][spread_id][2] = set([group_photos[photo_id] for photo_id in spread[2]])

                self.group_name2chosen_combinations[group_name + '*' + str(idx)] = best_spread
                self.logger.info('{} results:'.format(group_name + '*' + str(idx)))
                self.logger.info(self.group_name2chosen_combinations[group_name + '*' + str(idx)])

                del filtered_spreads
            self.logger.info("############################################################")

        except Exception as e:
            self.logger.error("Theres Error with group_name {}".format(group_name), e)
            return None

        return self.group_name2chosen_combinations

    def group_processing(self,group2images):
        start_time = time.time()
        if self.is_wedding:
            self.updated_groups, group2images = process_illegal_groups(group2images, self.original_groups, self.logger)
        illegal_time = (time.time() - start_time) / 60
        self.logger.info(f'Illegal groups processing time: {illegal_time:.2f} minutes')

        with Pool(processes=4) as pool:
            # Process the groups in parallel
            results = pool.map(self.process_groups, [group_name for group_name, _ in group2images.keys()])

        self.logger.info("Album Generation Finished ^_^")

        return results

    def start_processing_album(self):
        if self.is_wedding:
            self.original_groups = get_wedding_groups(self.df)
        else:
            self.original_groups = get_none_wedding_groups(self.df)

        group2images = get_images_per_groups(self.original_groups)
        self.look_up_table = get_lookup_table(group2images, self.is_wedding)

        return self.group_processing(group2images)



