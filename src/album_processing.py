import math
import time
import copy
from gc import collect

from src.core.photos import get_photos_from_db
from src.core.spreads import generate_filtered_multi_spreads
from src.core.scores import add_ranking_score, assign_photos_order
from src.groups_operations.groups_management import process_illegal_groups
from utils.lookup_table_tools import get_lookup_table
from utils.album_tools import get_none_wedding_groups, get_wedding_groups, get_images_per_groups
from utils.time_processing import sort_groups_by_time
from utils.configs import CONFIGS


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


def get_current_spread_parameters(group_key, number_of_images, is_wedding, lookup_table):
    # Extract the correct lookup key
    content_key = group_key[1].split("_")[0] if is_wedding and "_" in group_key[1] else group_key[1] if is_wedding else \
    group_key[0].split("_")[0]

    group_params = lookup_table.get(content_key, (10, 1.5))
    group_value = group_params[0]
    if group_value == 0:
        spreads = 0
    else:
        spreads = 1 if round(number_of_images / group_value) == 0 else round(number_of_images / group_value)

    if spreads > CONFIGS['max_group_spread']:
        max_images_per_spread = math.ceil(number_of_images / CONFIGS['max_group_spread'])
        if max_images_per_spread > CONFIGS['max_imges_per_spread']:
            max_images_per_spread = CONFIGS['max_imges_per_spread']
        return max_images_per_spread, group_params[1]

    return group_params


def update_lookup_table_with_limit(group2images, is_wedding, lookup_table, max_total_spreads):
    total_spreads = 0
    groups_with_three_spreads = []

    # First pass: Compute initial spreads and track groups with 3 spreads
    for key, number_images in group2images.items():
        spread_params = get_current_spread_parameters(key, number_images, is_wedding, lookup_table)
        spreads = round(number_images / spread_params[0])

        total_spreads += spreads

        # Track groups with 3 spreads for later reduction
        if spreads > 1 :
            groups_with_three_spreads.append(key)

    # If total spreads exceed limit, reduce spreads in groups with 3 spreads
    excess_spreads = total_spreads - max_total_spreads

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


def get_group_photos_list(cur_group_photos, spread_params, logger):
    cur_group_photos_list = copy.deepcopy(list())
    if ( (len(cur_group_photos) / (spread_params[0] - 2 * spread_params[1]) >= 4) or
            (math.ceil(len(cur_group_photos) / spread_params[0]) >= 3 and len(cur_group_photos) > 11) or
            (len(cur_group_photos) / (spread_params[0] - 2 * spread_params[1]) < 3 and len(
                cur_group_photos) > CONFIGS['max_imges_per_spread']) ):
        split_size = min(spread_params[0] * 3, max(spread_params[0], 11))
        number_of_splits = math.ceil(len(cur_group_photos) / split_size)
        logger.info('Condition we split!. Using splitting to {} parts'.format(number_of_splits))

        # Split as equally as possible
        total_items = len(cur_group_photos)
        base_size = total_items // number_of_splits
        remainder = total_items % number_of_splits

        start_idx = 0
        for split_num in range(number_of_splits):
            # Add 1 extra item to the first 'remainder' splits
            current_size = base_size + (1 if split_num < remainder else 0)
            end_idx = start_idx + current_size
            cur_group_photos_list.append(cur_group_photos[start_idx:end_idx])
            start_idx = end_idx

        # for split_num in range(number_of_splits):
        #     cur_group_photos_list.append(cur_group_photos[
        #                                  split_num * split_size: min((split_num + 1) * split_size,
        #                                                              len(cur_group_photos))])
    else:
        cur_group_photos_list.append(cur_group_photos)

    return cur_group_photos_list


def process_group(group_name, group_images_df, spread_params, designs_info ,is_wedding, params, logger):
    layouts_df = designs_info['anyPagelayouts_df']
    layout_id2data = designs_info['anyPagelayout_id2data']
    design_box_id2data = designs_info['anyPagebox_id2data']


    # logger.info(f"Processing group name {group_name}, and # of images {len(group_images_df)}")
    try:
        cur_group_photos = get_photos_from_db(group_images_df,is_wedding)
        # logger.info("Number of photos inside cur photos {} for group name {}".format(len(cur_group_photos), group_name))
        cur_group_photos_list = get_group_photos_list(cur_group_photos, spread_params, logger)

        local_result = {}
        group_idx = 0
        for group_photos in cur_group_photos_list:
            filter_start = time.time()
            filtered_spreads = generate_filtered_multi_spreads(group_photos, layouts_df, spread_params,params,logger)

            final_groups_and_spreads = None
            if filtered_spreads is None:

                for divider in [2, 3, 4]:
                    new_group_photos_list = get_group_photos_list(group_photos, spread_params, logger)
                    groups_filtered_spreads_list = list()
                    for cur_sub_group_photos in new_group_photos_list:
                        logger.debug("Filtered spreads not found we try again with different params. Group: {}. Params: {}".format(group_name, [spread_params[0] / divider, spread_params[1]]))
                        cur_filtered_spreads = generate_filtered_multi_spreads(cur_sub_group_photos, layouts_df,
                                                                           [spread_params[0] / divider, spread_params[1]],params, logger)
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
                    logger.warning('It is hopeless. Skipping group: {}'.format(group_name))
                    continue
            else:
                final_groups_and_spreads = [(group_photos, filtered_spreads)]

            # logger.info('Number of filtered spreads: {}. Their sizes: {}'.format(len(final_groups_and_spreads), [len(spr) for _, spr in final_groups_and_spreads]))
            # logger.info('Filtered spreads time: {}'.format(time.time() - filter_start))

            best_spread = None
            for sub_group_photos, filtered_spreads in final_groups_and_spreads:
                ranking_start = time.time()
                filtered_spreads = add_ranking_score(filtered_spreads, sub_group_photos, layout_id2data)
                filtered_spreads = sorted(filtered_spreads, key=lambda x: x[1], reverse=True)
                # logger.info(f"Number of filtered spreads after ranking sorted {len(filtered_spreads)}")
                # logger.info('Ranking time: {}'.format(time.time() - ranking_start))
                if len(filtered_spreads) == 0:
                    continue
                best_spread = filtered_spreads[0]
                cur_spreads = best_spread[0]
                for spread_id, spread in enumerate(cur_spreads):
                    best_spread[0][spread_id][1] = set([sub_group_photos[photo_id] for photo_id in spread[1]])
                    best_spread[0][spread_id][2] = set([sub_group_photos[photo_id] for photo_id in spread[2]])

                # TODO: add smart photos selection for boxes
                best_spread = assign_photos_order(best_spread, layout_id2data, design_box_id2data)

                if is_wedding:
                    local_result[str(group_name[0]) + '_' + group_name[1] + '*' + str(group_idx)] = best_spread
                else:
                    local_result[str(group_name[0]) + '*' + str(group_idx)] = best_spread
                group_idx += 1
            # logger.debug('Current group: {}. Best spread: {}'.format(group_name, best_spread))

        # logger.info("Finished with cur photos {} for group name {}".format(len(cur_group_photos), group_name))

        del cur_group_photos_list
        collect()

        return local_result

    except Exception as e:
        logger.error(f"Error processing group_name {group_name}: {e}")
        return None


def album_processing(df, designs_info, is_wedding, params, logger , density =3):
    if is_wedding:
        original_groups = get_wedding_groups(df,logger)
    else:
        original_groups = get_none_wedding_groups(df,logger)

    group2images = get_images_per_groups(original_groups)
    # logger.info('Detected groups: {}'.format(group2images))

    look_up_table = get_lookup_table(group2images,is_wedding,logger,density)

    # groups processing
    start_time = time.time()
    if is_wedding:
        updated_groups, group2images, look_up_table = process_illegal_groups(group2images, original_groups,
                                                                             look_up_table, is_wedding, logger)
        illegal_time = (time.time() - start_time)
        logger.info(f'Illegal groups processing time: {illegal_time:.2f} seconds')
    else:
        look_up_table = look_up_table
        updated_groups = original_groups

    logger.debug("Groups: {}".format(group2images))
    # make sure that each group has no more than 3 spreads
    look_up_table = update_lookup_table_with_limit(group2images, is_wedding, look_up_table, max_total_spreads=max(CONFIGS['max_total_spreads'], designs_info['maxPages']))

    result_list = []
    for group_name in group2images.keys():
        spread_params = get_current_spread_parameters(group_name, group2images[group_name], is_wedding, look_up_table)
        cur_result = process_group(group_name=group_name,
                                   group_images_df=updated_groups.get_group(group_name),
                                   spread_params=list(spread_params),
                                   designs_info=designs_info,
                                   is_wedding=is_wedding,
                                   params=params,
                                   logger=logger)
        result_list.append(cur_result)

    general_time = (time.time() - start_time)
    logger.info(f'General groups processing time: {general_time:.2f} seconds')

    # sorintg
    if is_wedding:
        # return sort_groups_by_name(result_list)
        return sort_groups_by_time(result_list, logger)
    else:
        return result_list









