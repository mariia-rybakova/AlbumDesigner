import time
import copy
from typing import List, Dict, Any, Optional, Tuple
import os
import json
import shutil

from src.groups_operations.groups_management import process_wedding_illegal_groups
from src.core.models import AlbumDesignResources, SpreadSearchParams
from utils.lookup_table_tools import WeddingLookUpTable, NonWeddingLookUpTable
from utils.album_tools import get_none_wedding_groups, get_wedding_groups, get_images_per_groups
from utils.time_processing import sort_groups_by_time
from src.spreads_layout.main import process_group
from utils.configs import CONFIGS
from utils.stages_recorder import set_is_artificial_time


def _json_default(o):
    if hasattr(o, 'item'):
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def album_processing(df, designs_info, is_wedding, modified_lut, params: SpreadSearchParams, logger, density=3,
                     manual_selection=False, all_gallery_df=None, selection_min_total_spreads=None,
                     is_artificial_time=False):
    # Make the artificial-time flag available to the stage recorders so the
    # split/merge/subgroups JSONs carry it and the visualizers can pick the
    # right time field (mirrors album1.pdf in process_gallery.py).
    set_is_artificial_time(is_artificial_time)
    group2images_initial = get_images_per_groups(get_wedding_groups(df, manual_selection, logger) if is_wedding else get_none_wedding_groups(df, logger))

    LookUpTable = WeddingLookUpTable if is_wedding else NonWeddingLookUpTable
    if modified_lut is not None:
        look_up_table = LookUpTable(modified_lut)
    else:
        look_up_table = LookUpTable()
        look_up_table.get_table(group2images_initial, logger, density)

    look_up_table.update_with_layouts_size(designs_info['anyPagelayouts_df'])

    max_total_spreads = max(CONFIGS['max_total_spreads'], designs_info['maxPages']) - 3
    min_total_spreads = min(max_total_spreads, designs_info['minPages']+6)
    look_up_table.update_with_limit(group2images_initial, max_total_spreads=max_total_spreads,
                                    min_total_spreads=min_total_spreads,logger = logger)

    resources = AlbumDesignResources.from_dict(designs_info, look_up_table)
    
    if is_wedding:
        original_groups = get_wedding_groups(df, manual_selection, logger)
    else:
        original_groups = get_none_wedding_groups(df, logger)

    group2images = get_images_per_groups(original_groups)
    logger.info('Detected groups: {}'.format(group2images))

    start_time = time.time()
    updated_photos_df = df
    if is_wedding:
        updated_groups, group2images, updated_photos_df = process_wedding_illegal_groups(df, resources, manual_selection, logger,
                                                                         all_gallery_df=all_gallery_df,
                                                                         selection_min_total_spreads=selection_min_total_spreads)
        resources.look_up_table = look_up_table
        logger.info(f'Illegal groups processing time: {time.time() - start_time:.2f} seconds')
    else:
        updated_groups = original_groups

    resources.look_up_table.update_with_limit(group2images, max_total_spreads=max_total_spreads,
                                              min_total_spreads=min_total_spreads,logger = logger)

    if CONFIGS['save_files']['spreads']:
        # Wipe stale per-group jsons from any previous run. Without this, files
        # whose group key isn't reused by the new run linger and mix into the
        # next visualization with photo ids that won't exist on disk.
        shutil.rmtree('files/stages_info/spreads', ignore_errors=True)
        os.makedirs('files/stages_info/spreads', exist_ok=True)
        with open('files/stages_info/spreads/_layouts.json', 'w') as f:
            json.dump(resources.printlab_data.to_dict(), f, indent=2, default=_json_default)

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
        return sort_groups_by_time(result_list, logger), updated_photos_df
    else:
        return result_list, updated_photos_df
