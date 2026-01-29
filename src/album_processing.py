import math
import time
import copy
from gc import collect
from typing import List, Dict, Any, Optional, Tuple

from src.core.photos import get_photos_from_db, Photo
from src.core.spreads import generate_filtered_multi_spreads
from src.core.scores import add_ranking_score, assign_photos_order
from src.groups_operations.groups_management import process_wedding_illegal_groups
from src.core.models import AlbumDesignResources, Spread, GroupProcessingResult
from utils.lookup_table_tools import WeddingLookUpTable, NonWeddingLookUpTable
from utils.album_tools import get_none_wedding_groups, get_wedding_groups, get_images_per_groups
from utils.time_processing import sort_groups_by_time
from utils.configs import CONFIGS


def get_group_photos_list(cur_group_photos: List[Photo], spread_params: List[float], largest_layout_size: int, logger) -> List[List[Photo]]:
    cur_group_photos_list = []

    optimal_spread_param = min(largest_layout_size, spread_params[0])
    if len(cur_group_photos) / (max(optimal_spread_param - 2 * spread_params[1], 1)) >= 4:
        split_size = min(optimal_spread_param * 3, max(optimal_spread_param, 11))
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
    else:
        cur_group_photos_list.append(cur_group_photos)

    return cur_group_photos_list


def process_group(group_name: Tuple, group_images_df, spread_params: List[float], 
                  resources: AlbumDesignResources, is_wedding: bool, params: List[float], logger) -> Optional[Dict[str, GroupProcessingResult]]:
    layouts_df = resources.layouts_df
    layout_id2data = resources.layout_id2data
    design_box_id2data = resources.box_id2data

    largest_layout_size = max(list(layouts_df['number of boxes'].unique()))
    start = time.time()
    try:
        if is_wedding and 'dancing' in group_name[1]:
            group_images_df = group_images_df.sort_values(['image_as', 'image_time'])
        else:
            group_images_df = group_images_df.sort_values(['image_time'])

        cur_group_photos = get_photos_from_db(group_images_df, is_wedding)
        cur_group_photos_list = get_group_photos_list(cur_group_photos, spread_params, largest_layout_size, logger)
        if len(cur_group_photos_list) > 1:
            logger.info('Group: {} with size: {} was split into {} parts.'.format(group_name, len(cur_group_photos), len(cur_group_photos_list)))

        local_result = {}
        group_idx = 0
        for group_photos in cur_group_photos_list:
            final_groups_and_spreads = _find_spreads_for_group(group_photos, layouts_df, spread_params, params, largest_layout_size, group_name, logger)

            if final_groups_and_spreads is None:
                continue

            for sub_group_photos, filtered_spreads in final_groups_and_spreads:
                best_spread_data = _rank_and_select_best_spread(filtered_spreads, sub_group_photos, layout_id2data, design_box_id2data)
                
                if best_spread_data is None:
                    continue

                spreads_list, score = best_spread_data
                
                structured_spreads = []
                for s in spreads_list:
                    structured_spreads.append(Spread(layout_id=s[0], left_photos=list(s[1]), right_photos=list(s[2])))
                
                group_id_str = str(group_name[0]) + '_' + group_name[1] if is_wedding else str(group_name[0])
                group_id_str += '*' + str(group_idx)
                
                local_result[group_id_str] = GroupProcessingResult(group_name=group_id_str, spreads=structured_spreads, score=score)
                group_idx += 1

        collect()
        end = time.time()
        logger.info(f"Processed group name {group_name} in {end - start:.2f} seconds.")

        return local_result

    except Exception as ex:
        import traceback
        tb = traceback.extract_tb(ex.__traceback__)
        filename, lineno, func, text = tb[-1]
        logger.error(f"Error processing group_name {group_name}: {ex}. Exception in function: {func}, line {lineno}, file {filename}.")
        return None


def _find_spreads_for_group(group_photos, layouts_df, spread_params, params, largest_layout_size, group_name, logger):
    filtered_spreads = generate_filtered_multi_spreads(group_photos, layouts_df, spread_params, params, logger)
    
    if filtered_spreads is not None:
        return [(group_photos, filtered_spreads)]

    # Retry with smaller spread params
    for divider in [0.8, 0.6, 0.4, 0.2]:
        new_spread_params = [round(spread_params[0] * divider), spread_params[1]]
        new_group_photos_list = get_group_photos_list(group_photos, new_spread_params, largest_layout_size, logger)
        groups_filtered_spreads_list = []
        
        for cur_sub_group_photos in new_group_photos_list:
            logger.debug("Filtered spreads not found we try again with different params. Group: {}. Params: {}. Divider: {}.".format(group_name, new_spread_params, divider))
            cur_filtered_spreads = generate_filtered_multi_spreads(cur_sub_group_photos, layouts_df, new_spread_params, params, logger)
            if cur_filtered_spreads is None:
                groups_filtered_spreads_list = None
                break
            else:
                groups_filtered_spreads_list.append((cur_sub_group_photos, cur_filtered_spreads))

        if groups_filtered_spreads_list is not None:
            return groups_filtered_spreads_list

    # Last resort: Add dummy photo
    logger.info('Coundnt find spread for the group: {}. Adding dummy photo to photo list.'.format(group_name))
    dummy_photo = Photo(id=-1, ar=1.5, color=True, rank=1000000, photo_class='None', cluster_label=1,
                        general_time=1000000, original_context='None')
    group_photos_with_dummy = group_photos + [dummy_photo]
    filtered_spreads = generate_filtered_multi_spreads(group_photos_with_dummy, layouts_df, spread_params, params, logger)
    
    if filtered_spreads is not None:
        logger.info('Spread created using dummy photo for the group: {}.'.format(group_name))
        return [(group_photos_with_dummy, filtered_spreads)]
    
    logger.warning('It is hopeless. Skipping group: {}'.format(group_name))
    return None


def _rank_and_select_best_spread(filtered_spreads, sub_group_photos, layout_id2data, design_box_id2data):
    filtered_spreads = add_ranking_score(filtered_spreads, sub_group_photos, layout_id2data)
    filtered_spreads = sorted(filtered_spreads, key=lambda x: x[1], reverse=True)
    
    if not filtered_spreads:
        return None
    
    best_spread = filtered_spreads[0]
    cur_spreads = best_spread[0]
    
    # Transform photo IDs to Photo objects
    for spread_id, spread in enumerate(cur_spreads):
        best_spread[0][spread_id][1] = set([sub_group_photos[photo_id] for photo_id in spread[1]])
        best_spread[0][spread_id][2] = set([sub_group_photos[photo_id] for photo_id in spread[2]])

    best_spread = assign_photos_order(best_spread, layout_id2data, design_box_id2data, merge_pages=False)
    return best_spread


def album_processing(df, designs_info, is_wedding, modified_lut, params, logger, density=3, manual_selection=False):
    group2images_initial = get_images_per_groups(get_wedding_groups(df, manual_selection, logger) if is_wedding else get_none_wedding_groups(df, logger))

    LookUpTable = WeddingLookUpTable if is_wedding else NonWeddingLookUpTable
    if modified_lut is not None:
        look_up_table = LookUpTable(modified_lut)
    else:
        look_up_table = LookUpTable()
        look_up_table.get_table(group2images_initial, logger, density)

    look_up_table.update_with_layouts_size(designs_info['anyPagelayouts_df'])
    
    max_total_spreads = max(CONFIGS['max_total_spreads'], designs_info['maxPages'])
    look_up_table.update_with_limit(group2images_initial, max_total_spreads=max_total_spreads)

    resources = AlbumDesignResources.from_dict(designs_info, look_up_table)
    
    if is_wedding:
        original_groups = get_wedding_groups(df, manual_selection, logger)
    else:
        original_groups = get_none_wedding_groups(df, logger)

    group2images = get_images_per_groups(original_groups)
    logger.info('Detected groups: {}'.format(group2images))

    start_time = time.time()
    if is_wedding:
        updated_groups, group2images = process_wedding_illegal_groups(df, resources, manual_selection, logger)
        resources.look_up_table = look_up_table
        logger.info(f'Illegal groups processing time: {time.time() - start_time:.2f} seconds')
    else:
        updated_groups = original_groups

    resources.look_up_table.update_with_limit(group2images,  max_total_spreads=max_total_spreads)

    result_list = []
    for group_name in group2images.keys():
        spread_params = resources.look_up_table.get_current_spread_parameters(group_name, group2images[group_name])
        if spread_params[0] > 24:
            spread_params = (24.0, spread_params[1])
            
        cur_result = process_group(group_name=group_name,
                                   group_images_df=updated_groups.get_group(group_name),
                                   spread_params=list(spread_params),
                                   resources=resources,
                                   is_wedding=is_wedding,
                                   params=params,
                                   logger=logger)
        if cur_result:
            result_list.append(cur_result)

    logger.info(f'General groups processing time: {time.time() - start_time:.2f} seconds')

    if is_wedding:
        return sort_groups_by_time(result_list, logger)
    else:
        return result_list









