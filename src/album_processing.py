import math
import time
import copy
import queue
import traceback
import threading
from gc import collect

from utils import get_photos_from_db, generate_filtered_multi_spreads, add_ranking_score, process_illegal_groups
from utils.lookup_table_tools import get_lookup_table
from utils.album_tools import get_none_wedding_groups,get_wedding_groups,get_images_per_groups,organize_and_sort_groups



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
        print("group name", group_name, "number of imgs", len(group_images_df))
        print("1")
        try:
            cur_group_photos = get_photos_from_db(group_images_df)
            print("2")
            cur_group_photos_list = copy.deepcopy(list())
            if (len(cur_group_photos) / (spread_params[0] - 2 * spread_params[1]) >= 4 or
                    # len(cur_group_photos) / spread_params[0] >= 3 and len(cur_group_photos) > 11 or
                    len(cur_group_photos) / (spread_params[0] - 2 * spread_params[1]) < 3 and len(
                        cur_group_photos) > 24):
                split_size = min(spread_params[0] * 3, max(spread_params[0], 11))
                number_of_splits = math.ceil(len(cur_group_photos) / split_size)
                print('Using splitting to {} parts'.format(number_of_splits))
                for split_num in range(number_of_splits):
                    cur_group_photos_list.append(cur_group_photos[
                                                 split_num * split_size: min((split_num + 1) * split_size,
                                                                             len(cur_group_photos))])
            else:
                cur_group_photos_list.append(cur_group_photos)
            print("3")
            local_result = {}
            for idx, group_photos in enumerate(cur_group_photos_list):
                filter_start = time.time()

                # print('Photos: {}'.format(
                #     [[item.id, item.ar, item.color, item.rank, item.photo_class, item.cluster_label, item.general_time]
                #      for item in group_photos]))
                filtered_spreads = generate_filtered_multi_spreads(group_photos, layouts_df, spread_params,None)
                if filtered_spreads is None:
                    continue
                print('Filtered spreads size: {}'.format(len(filtered_spreads)))
                print('Filtered spreads time: {}'.format(time.time() - filter_start))

                ranking_start = time.time()
                filtered_spreads = add_ranking_score(filtered_spreads, group_photos, layout_id2data)
                filtered_spreads = sorted(filtered_spreads, key=lambda x: x[1], reverse=True)
                print('Ranking time: {}'.format(time.time() - ranking_start))

                best_spread = filtered_spreads[0]
                cur_spreads = best_spread[0]
                for spread_id, spread in enumerate(cur_spreads):
                    best_spread[0][spread_id][1] = set([group_photos[photo_id] for photo_id in spread[1]])
                    best_spread[0][spread_id][2] = set([group_photos[photo_id] for photo_id in spread[2]])
                print("4")
                local_result[str(group_name[0]) +'_' + group_name[1] + '*' + str(idx)] = best_spread

                del cur_group_photos, filtered_spreads

            del cur_group_photos_list
            collect()
            print("############################################################")
            # Critical section: update the shared result dictionary

            print("5")
            return local_result

        except Exception as e:
            print(f"Error with group_name {group_name}: {e}")
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
        illegal_time = (time.time() - start_time) / 60
        print(f'Illegal groups processing time: {illegal_time:.2f} minutes')

        args = [
            (group_name,
                copy.deepcopy(self.updated_groups.get_group(group_name)),
                copy.deepcopy(list(self.look_up_table.get(group_name[1].split('_')[0], (10, 1.5)))),
                copy.deepcopy(self.layouts_df),
                copy.deepcopy(self.layout_id2data))
         for group_name in group2images.keys()
        ]
        all_results = self.process_all_groups_parallel(args)

        return all_results

    def start_processing_album(self):
        if self.is_wedding:
            self.original_groups = get_wedding_groups(self.df)
        else:
            self.original_groups = get_none_wedding_groups(self.df)

        group2images = get_images_per_groups(self.original_groups)
        look_up_table = get_lookup_table(group2images, self.is_wedding)

        result_list = self.groups_processing(group2images,look_up_table)

        #sorintg & formating & cropping
        result = organize_and_sort_groups(result_list,self.layouts_df,self.updated_groups, self.is_wedding)

        return result





