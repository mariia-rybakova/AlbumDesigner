import math
import time
import copy
import queue
import traceback
import threading
from gc import collect

from networkx import is_weighted

from utils import get_photos_from_db, generate_filtered_multi_spreads, add_ranking_score, process_illegal_groups
from utils.lookup_table_tools import get_lookup_table
from utils.album_tools import get_none_wedding_groups,get_wedding_groups,get_images_per_groups,organize_groups,sort_groups_by_name



class AutomaticAlbum:
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

    def process_group(self,args):
        group_name, group_images_df, spread_params, layouts_df, layout_id2data = args
        self.logger.info(f"Processing group name {group_name}, and # of images {len(group_images_df)}")
        try:
            cur_group_photos = get_photos_from_db(group_images_df,self.is_wedding)
            cur_group_photos_list = copy.deepcopy(list())
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

            local_result = {}
            for idx, group_photos in enumerate(cur_group_photos_list):
                filter_start = time.time()
                filtered_spreads = generate_filtered_multi_spreads(group_photos, layouts_df, spread_params,None)
                if filtered_spreads is None:
                    continue
                self.logger.info('Filtered spreads size: {}'.format(len(filtered_spreads)))
                self.logger.info('Filtered spreads time: {}'.format(time.time() - filter_start))

                ranking_start = time.time()
                filtered_spreads = add_ranking_score(filtered_spreads, group_photos, layout_id2data)
                filtered_spreads = sorted(filtered_spreads, key=lambda x: x[1], reverse=True)
                self.logger.info('Ranking time: {}'.format(time.time() - ranking_start))

                best_spread = filtered_spreads[0]
                cur_spreads = best_spread[0]
                for spread_id, spread in enumerate(cur_spreads):
                    best_spread[0][spread_id][1] = set([group_photos[photo_id] for photo_id in spread[1]])
                    best_spread[0][spread_id][2] = set([group_photos[photo_id] for photo_id in spread[2]])

                if self.is_wedding:
                    local_result[str(group_name[0]) + '_' + group_name[1] + '*' + str(idx)] = best_spread
                else:
                    local_result[str(group_name[0]) + '*' + str(idx)] = best_spread


                del cur_group_photos, filtered_spreads

            del cur_group_photos_list
            collect()

            return local_result

        except Exception as e:
            self.logger.error(f"Error with group_name {group_name}: {e}")
            print(traceback.format_exc())
            return None

    def process_all_groups_parallel(self,args):
        jobs = []
        q = queue.Queue()  # thread safe queue to store results

        def worker(arg):
            result = self.process_group(arg)
            q.put(result)  # put the result in queue

        for arg in args:
            thread = threading.Thread(target=worker, args=(arg,))
            jobs.append(thread)

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()

        results = [q.get() for _ in range(len(args))]  # retrieve results from the queue
        return results

    def groups_processing(self,group2images,look_up_table):
        start_time = time.time()
        if self.is_wedding:
            self.updated_groups, group2images,self.look_up_table = process_illegal_groups(group2images, self.original_groups,look_up_table,self.is_wedding, self.logger)
            if self.updated_groups is None:
                 return 'Error: couldn\'t process illegal groups'
            illegal_time = (time.time() - start_time) / 60
            self.logger.info(f'Illegal groups processing time: {illegal_time:.2f} minutes')
        else:
            self.look_up_table = look_up_table
            self.updated_groups = self.original_groups

        args = [
            (group_name,
                copy.deepcopy(self.updated_groups.get_group(group_name)),
                copy.deepcopy(list(self.look_up_table.get(group_name[1].split('_')[0], (10, 1.5)))) if self.is_wedding else copy.deepcopy(list(self.look_up_table.get(group_name[0].split('_')[0], (1, 2)))),
                copy.deepcopy(self.layouts_df),
                copy.deepcopy(self.layout_id2data))
         for group_name in group2images.keys()
        ]
        all_results = self.process_all_groups_parallel(args)

        return all_results

    def start_processing_album(self):
        if self.is_wedding:
            self.original_groups = get_wedding_groups(self.df,self.logger)
        else:
            self.original_groups = get_none_wedding_groups(self.df,self.logger)

        if isinstance(self.original_groups, str):  # Check if it's an error message, report it
            return self.original_groups

        group2images = get_images_per_groups(self.original_groups,self.logger)

        if isinstance(group2images, str):  # Check if it's an error message, report it
            return group2images

        look_up_table = get_lookup_table(group2images, self.is_wedding,self.logger)

        if isinstance(look_up_table, str):  # Check if it's an error message, report it
            return look_up_table

        result_list = self.groups_processing(group2images,look_up_table)

        if isinstance(result_list, str):  # Check if it's an error message, report it
            return result_list

        #sorintg & formating & cropping
        if self.is_wedding:
            sorted_result_list = sort_groups_by_name(result_list,self.is_wedding)
            result = organize_groups(sorted_result_list,self.layouts_df,self.updated_groups, self.is_wedding,self.logger)
        else:
            result = "^_^"
        return result





