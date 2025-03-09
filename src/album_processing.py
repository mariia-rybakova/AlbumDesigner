import math
import time
import copy
import queue
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor
from gc import collect

from utils import get_photos_from_db, generate_filtered_multi_spreads, add_ranking_score, process_illegal_groups
from utils.lookup_table_tools import get_lookup_table
from utils.album_tools import get_none_wedding_groups,get_wedding_groups,get_images_per_groups,organize_groups,sort_groups_by_name
from utils.parser import CONFIGS


def get_group_photos_list(cur_group_photos, spread_params, logger):
    cur_group_photos_list = copy.deepcopy(list())
    if (len(cur_group_photos) / (spread_params[0] - 2 * spread_params[1]) >= 4 or
            round(len(cur_group_photos) / spread_params[0]) >= 3 and len(cur_group_photos) > 11 or
            len(cur_group_photos) / (spread_params[0] - 2 * spread_params[1]) < 3 and len(
                cur_group_photos) > CONFIGS['max_imges_per_spread']):
        split_size = min(spread_params[0] * 3, max(spread_params[0], 11))
        number_of_splits = math.ceil(len(cur_group_photos) / split_size)
        logger.info('Condition we split!. Using splitting to {} parts'.format(number_of_splits))
        for split_num in range(number_of_splits):
            cur_group_photos_list.append(cur_group_photos[
                                         split_num * split_size: min((split_num + 1) * split_size,
                                                                     len(cur_group_photos))])
    else:
        cur_group_photos_list.append(cur_group_photos)

    return cur_group_photos_list


def process_group(args):
    group_name, group_images_df, spread_params, layouts_df, layout_id2data ,is_wedding,logger= args
    logger.info(f"Processing group name {group_name}, and # of images {len(group_images_df)}")
    try:
        cur_group_photos = get_photos_from_db(group_images_df,is_wedding)
        logger.info("Number of photos inside cur photos {} for group name {}".format(len(cur_group_photos), group_name))
        cur_group_photos_list = get_group_photos_list(cur_group_photos, spread_params, logger)

        local_result = {}
        group_idx = 0
        for group_photos in cur_group_photos_list:
            filter_start = time.time()
            filtered_spreads = generate_filtered_multi_spreads(group_photos, layouts_df, spread_params,logger)

            final_groups_and_spreads = None
            if filtered_spreads is None:
                print("Filtered spreads not found we try again with different params")
                for divider in [2, 3, 4]:
                    new_group_photos_list = get_group_photos_list(group_photos, spread_params, logger)
                    groups_filtered_spreads_list = list()
                    for cur_sub_group_photos in new_group_photos_list:
                        cur_filtered_spreads = generate_filtered_multi_spreads(cur_sub_group_photos, layouts_df,
                                                                           [spread_params[0] / divider, spread_params[1]], logger)
                        if cur_filtered_spreads is None:
                            groups_filtered_spreads_list = None
                            break
                        else:
                            groups_filtered_spreads_list.append((cur_sub_group_photos, cur_filtered_spreads))

                    if groups_filtered_spreads_list is None:
                        continue
                    else:
                        final_groups_and_spreads = groups_filtered_spreads_list
                        break
                if final_groups_and_spreads is None:
                    logger.info('It is hopeless. Skipping group: {}'.format(group_name))
                    continue
            else:
                final_groups_and_spreads = [(group_photos, filtered_spreads)]

            logger.info('Number of filtered spreads: {}. Their sizes: {}'.format(len(final_groups_and_spreads), [len(spr) for _, spr in final_groups_and_spreads]))
            logger.info('Filtered spreads time: {}'.format(time.time() - filter_start))

            for sub_group_photos, filtered_spreads in final_groups_and_spreads:
                ranking_start = time.time()
                filtered_spreads = add_ranking_score(filtered_spreads, sub_group_photos, layout_id2data)
                filtered_spreads = sorted(filtered_spreads, key=lambda x: x[1], reverse=True)
                logger.info(f"Number of filtered spreads after ranking sorted {len(filtered_spreads)}")
                logger.info('Ranking time: {}'.format(time.time() - ranking_start))
                if len(filtered_spreads) == 0:
                    continue
                best_spread = filtered_spreads[0]
                cur_spreads = best_spread[0]
                for spread_id, spread in enumerate(cur_spreads):
                    best_spread[0][spread_id][1] = set([sub_group_photos[photo_id] for photo_id in spread[1]])
                    best_spread[0][spread_id][2] = set([sub_group_photos[photo_id] for photo_id in spread[2]])

                if is_wedding:
                    local_result[str(group_name[0]) + '_' + group_name[1] + '*' + str(group_idx)] = best_spread
                else:
                    local_result[str(group_name[0]) + '*' + str(group_idx)] = best_spread
                group_idx += 1


                # del filtered_spreads

        logger.info("Finished with cur photos {} for group name {}".format(len(cur_group_photos), group_name))

        del cur_group_photos_list
        collect()

        return local_result

    except Exception as e:
        logger.error(f"Error with group_name {group_name}: {e}")
        print(traceback.format_exc())
        return None


def update_lookup_table(group2images, lookup_table, is_wedding):
    for key, number_images in group2images.items():
        # Extract the correct lookup key
        content_key = key[1].split("_")[0] if is_wedding and "_" in key[1] else key[1] if is_wedding else \
        key[0].split("_")[0]

        # Get group value, default to 0 if not found
        group_value = lookup_table.get(content_key, (0,))[0]

        # Calculate spreads safely
        spreads = round(number_images / group_value) if group_value else 0

        # Cap spreads at 3 but ensure at least 1 if necessary
        spreads = max(1, min(spreads, 3))

        # Update the lookup table with the new spreads while keeping the second tuple value unchanged
        lookup_table[key] = (spreads, lookup_table[key][1])

    return lookup_table

def update_lookup_table_with_limit(group2images, is_wedding, lookup_table):
    total_spreads = 0
    groups_with_three_spreads = []

    # First pass: Compute initial spreads and track groups with 3 spreads
    for key, number_images in group2images.items():
        # Extract the correct lookup key
        content_key = key[1].split("_")[0] if is_wedding and "_" in key[1] else key[1] if is_wedding else key[0].split("_")[0]

        # Get group value, default to 0 if not found
        group_value = lookup_table.get(content_key, (0,))[0]

        # Calculate spreads safely
        spreads = 1 if round(number_images / group_value) == 0 else round(number_images / group_value)

        if spreads > CONFIGS['max_group_spread'] :
            # Update lookup table
            max_images_per_spread = math.ceil(number_images / CONFIGS['max_group_spread'])
            lookup_table[content_key] = (max_images_per_spread , lookup_table[content_key][1])
            spreads = round(number_images / max_images_per_spread)

        total_spreads += spreads

        # Track groups with 3 spreads for later reduction
        if spreads > 1 :
            groups_with_three_spreads.append(key)

    # If total spreads exceed limit, reduce spreads in groups with 3 spreads
    excess_spreads = total_spreads - CONFIGS['max_total_spreads']

    if excess_spreads > 0:
        for key in groups_with_three_spreads:
            if excess_spreads <= 0:
                break  # Stop once the total spreads is within limit

            content_key = key[1].split("_")[0] if is_wedding and "_" in key[1] else key[1] if is_wedding else \
            key[0].split("_")[0]
            current_max_imges , extra_value = lookup_table[content_key]
            spread = 1 if round(group2images[key] / current_max_imges) - 1 == 0 else round(group2images[key] / current_max_imges)
            lookup_table[content_key] = (round(group2images[key] / spread), extra_value)
            excess_spreads -= 1  # Reduce excess count

    return lookup_table

def process_all_groups_parallel(args):
    jobs = []
    q = queue.Queue()  # thread safe queue to store results

    def worker(arg):
        result = process_group(arg)
        q.put(result)  # put the result in queue
    #("Process with layout workers ", CONFIGS['max_lay_workers'])
    with ThreadPoolExecutor(max_workers=CONFIGS['max_lay_workers']) as executor:
        executor.map(worker, args)

    results = [q.get() for _ in range(len(args))]  # retrieve results from the queue
    return results

def groups_processing(group2images,original_groups,look_up_table,layouts_df,layout_id2data,is_wedding,logger):
    start_time = time.time()
    if is_wedding:
        updated_groups, group2images,look_up_table = process_illegal_groups(group2images, original_groups,look_up_table,is_wedding, logger)
        if updated_groups is None:
             return 'Error: couldn\'t process illegal groups'
        illegal_time = (time.time() - start_time)
        logger.info(f'Illegal groups processing time: {illegal_time:.2f} seconds')
    else:
        look_up_table = look_up_table
        updated_groups = original_groups

    print("Groups", group2images)
    # make sure that each group has no more than 3 spreads
    look_up_table = update_lookup_table_with_limit(group2images,is_wedding,look_up_table)


    args = [
        (group_name,
            copy.deepcopy(updated_groups.get_group(group_name)),
            copy.deepcopy(list(look_up_table.get(group_name[1].split('_')[0], (10, 1.5)))) if is_wedding else copy.deepcopy(list(look_up_table.get(group_name[0].split('_')[0], (1, 2)))),
            copy.deepcopy(layouts_df),
            copy.deepcopy(layout_id2data),
            copy.deepcopy(is_wedding),
            copy.deepcopy(logger))
     for group_name in group2images.keys()
    ]
    all_results = process_all_groups_parallel(args)
    print("Results", all_results)

    return all_results,updated_groups

def start_processing_album(df, layouts_df, layout_id2data, is_wedding, logger):
        if is_wedding:
            original_groups = get_wedding_groups(df,logger)
        else:
            original_groups = get_none_wedding_groups(df,logger)

        if isinstance(original_groups, str):  # Check if it's an error message, report it
            return original_groups

        group2images = get_images_per_groups(original_groups,logger)

        if isinstance(group2images, str):  # Check if it's an error message, report it
            return group2images

        look_up_table = get_lookup_table(group2images,is_wedding,logger)

        if isinstance(look_up_table, str):  # Check if it's an error message, report it
            return look_up_table

        result_list,updated_groups = groups_processing(group2images,original_groups,look_up_table,layouts_df, layout_id2data, is_wedding, logger)

        if isinstance(result_list, str):  # Check if it's an error message, report it
            return result_list

        #sorintg & formating & cropping
        if is_wedding:
            sorted_result_list = sort_groups_by_name(result_list)
            #result = organize_groups(sorted_result_list,layouts_df,updated_groups, is_wedding,logger)
        else:
            sorted_result_list = result_list

        result = format_output(sorted_result_list)

        return result





