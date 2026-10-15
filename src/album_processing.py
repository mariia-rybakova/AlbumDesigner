import time
import copy
from typing import List, Dict, Any, Optional, Tuple
import os
import json
import shutil

from src.groups_operations.groups_management import process_wedding_illegal_groups
from src.core.models import AlbumDesignResources, SpreadSearchParams
from utils.lookup_table_tools import WeddingLookUpTable, NonWeddingLookUpTable
from src.groups_operations.grouping import get_none_wedding_groups, get_wedding_groups, get_images_per_groups
from src.spreads_layout.ordering import sort_groups_by_time, reorder_disjoint_spreads
from src.spreads_layout.main import process_group
from utils.configs import CONFIGS
from src.stages_recorder import set_is_artificial_time, set_general_time_index, reset_combinations_dir
from src.spreads_layout.spreads.spread import set_neutral_contexts
from src.spreads_layout.twins import apply_twin_colors
from src.spreads_layout import fold


#: Compositions the reply always carries besides the content spreads: the
#: cover, the first page and the last page (see `src/request_processing.py`).
#: A printlab page count converts to a content-spread budget by subtracting
#: these, which is what the historical `- 3` has always been.
FIXED_COMPOSITIONS = 3


def _persons_in(group_name) -> int:
    """How many people a non-wedding group is of: the count its
    ``people_cluster`` key starts with (``'3_pple_2_12_4'`` -> 3,
    ``'1_person_10'`` -> 1); ``'No PEOPLE'`` and anything unparsable -> 0."""
    key = group_name[0] if isinstance(group_name, tuple) else group_name
    head = str(key).split('_', 1)[0]
    return int(head) if head.isdigit() else 0


def _group_rank(group2images):
    """Non-wedding groups, most people first; a tie goes to the bigger group."""
    return sorted(group2images, key=lambda g: (-_persons_in(g), -group2images[g], str(g)))


def cap_non_wedding_groups(group2images, spreads_per_group, max_total_spreads, logger=None):
    """The groups a non-wedding album keeps so its spreads fit the design.

    The non-wedding route groups photos by who is in them, and a gallery the
    content model never classified splits into one small group per
    combination of people. Each needs at least one spread, and the reduction in
    `update_with_limit` only ever takes spreads *from* a group, never a group
    away -- so 44090624 (2026-10-12; every one of its 3,034 photos
    ``image_class -1``) was laid out at 142 spreads, sent 146 compositions, and
    its product allows 41.

    Groups are taken most people first, as long as their spreads still fit
    ``max_total_spreads`` (the design's maximum less `FIXED_COMPOSITIONS`); a
    group too big for what is left is skipped and smaller ones still try.
    Returns the kept group names, in ``group2images`` order.
    """
    total = sum(spreads_per_group.values())
    if total <= max_total_spreads:
        return list(group2images)

    kept, used = set(), 0
    for group in _group_rank(group2images):
        need = spreads_per_group.get(group, 1)
        if used + need <= max_total_spreads:
            kept.add(group)
            used += need
    if not kept:
        # Not even the first group fits: an album of it beats an empty one.
        first = _group_rank(group2images)[0]
        kept, used = {first}, spreads_per_group.get(first, 1)
    if logger:
        logger.warning(f"Non-wedding cap: {len(group2images)} groups need {total} spreads, "
                       f"the design allows {max_total_spreads}; kept {len(kept)} groups "
                       f"({used} spreads), most people first")
    return [g for g in group2images if g in kept]


def trim_non_wedding_results(named_results, group2images, max_total_spreads, logger=None):
    """Drop the fewest-people groups until the laid-out spreads fit.

    The layout search may give a group a spread more than it was planned at, so
    the cap above can still land a spread or two over. ``named_results`` is
    ``[(group_name, {group_id: GroupProcessingResult}), ...]``.
    """
    def spreads(result):
        return sum(len(r.spreads) for r in result.values())

    total = sum(spreads(r) for _, r in named_results)
    if total <= max_total_spreads:
        return named_results
    order = {g: i for i, g in enumerate(_group_rank(group2images))}
    dropped = set()
    for i, (group, result) in sorted(enumerate(named_results),
                                     key=lambda item: -order.get(item[1][0], len(order))):
        if total <= max_total_spreads:
            break
        dropped.add(i)
        total -= spreads(result)
    if logger:
        logger.warning(f"Non-wedding cap: laid out over {max_total_spreads} spreads; "
                       f"dropped {len(dropped)} fewest-people group(s), {total} spreads left")
    return [nr for i, nr in enumerate(named_results) if i not in dropped]


def _json_default(o):
    if hasattr(o, 'item'):
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def size_album(designs_info, selection_max_total_spreads=None) -> Tuple[int, int]:
    """The content-spread budget for one album, as (min_total, max_total).

    Content spreads, not compositions: the reply adds `FIXED_COMPOSITIONS` on
    top, so a caller wanting N compositions asks for N pages and gets
    N - FIXED_COMPOSITIONS spreads here.

    A design whose ``minPages`` equals its ``maxPages`` is stating an exact
    album rather than a range, and is taken literally -- neither the
    `min_pages_headroom` on the floor nor selection's gallery-aware ceiling
    applies, and
    the design's own page count is used instead of
    ``CONFIGS['max_total_spreads']``. Without that last part a design asking
    for fewer pages than the config default was silently widened to it, because
    the ceiling takes whichever of the two is *larger*.
    """
    min_pages, max_pages = designs_info['minPages'], designs_info['maxPages']

    if min_pages == max_pages:
        exact = max(1, min_pages - FIXED_COMPOSITIONS)
        return exact, exact

    # Hard ceiling from the album design - the album can never exceed this.
    hard_max_total_spreads = max(CONFIGS['max_total_spreads'], max_pages) - FIXED_COMPOSITIONS
    # Wedding selection proposes a tighter, gallery-aware limit; clamp it to the hard ceiling.
    # Non-wedding has no selection target and keeps the design limit alone.
    max_total_spreads = (min(selection_max_total_spreads, hard_max_total_spreads)
                         if selection_max_total_spreads is not None else hard_max_total_spreads)
    # The floor sits `min_pages_headroom` above the design's minimum. It was 6,
    # and once selection's range stopped counting the opening and closing pages
    # that floor sat above its target: 49994361's design (minPages 12) floored
    # the album at 18 content spreads against a plan of 17.
    headroom = int(CONFIGS.get('min_pages_headroom', 3))
    return min(max_total_spreads, min_pages + headroom), max_total_spreads


def layout_target(selection_target_spreads, min_total_spreads, max_total_spreads):
    """The spread total the layout plan reduces toward, or None for "the ceiling".

    Selection sizes its photos for a number of spreads -- the sum of its
    per-category plan -- and the layout used to see only the range around it.
    Rounded to whole spreads and held inside the album's own [min, max], so a
    design that demands more pages still gets them.
    """
    if selection_target_spreads is None or not selection_target_spreads > 0:
        return None
    return int(min(max(round(selection_target_spreads), min_total_spreads), max_total_spreads))


def album_processing(df, designs_info, is_wedding, modified_lut, params: SpreadSearchParams, logger, density=3,
                     manual_selection=False, all_gallery_df=None, selection_min_total_spreads=None,
                     selection_max_total_spreads=None, is_artificial_time=False,
                     selection_target_spreads=None, neutral_contexts=None):
    # Make the artificial-time flag available to the stage recorders so the
    # split/merge/subgroups JSONs carry it and the visualizers can pick the
    # right time field (mirrors album1.pdf in process_gallery.py).
    set_is_artificial_time(is_artificial_time)
    # `Photo` carries only `general_time`, so recorders downstream of
    # `get_photos_from_df` need this to report the wall clock album1.pdf shows.
    set_general_time_index(df)
    group2images_initial = get_images_per_groups(get_wedding_groups(df, manual_selection, logger) if is_wedding else get_none_wedding_groups(df, logger))

    LookUpTable = WeddingLookUpTable if is_wedding else NonWeddingLookUpTable
    if modified_lut is not None:
        look_up_table = LookUpTable(modified_lut)
    else:
        look_up_table = LookUpTable()
        look_up_table.get_table(group2images_initial, logger, density)

    look_up_table.update_with_layouts_size(designs_info['anyPagelayouts_df'])

    # Wedding contexts only: a non-wedding group key is not a content class.
    set_neutral_contexts(neutral_contexts if is_wedding else None)

    min_total_spreads, max_total_spreads = size_album(designs_info, selection_max_total_spreads)
    logger.info(f"Printlab data: minPages={designs_info['minPages']}. Calculated: min_total_spreads={min_total_spreads}")
    logger.info(f"Printlab data: maxPages={designs_info['maxPages']}. Calculated: max_total_spreads={max_total_spreads}")
    target_total_spreads = layout_target(selection_target_spreads, min_total_spreads, max_total_spreads)
    if target_total_spreads is not None:
        logger.info(f"Layout target: {target_total_spreads} spreads "
                    f"(selection planned {selection_target_spreads:.2f})")
    look_up_table.update_with_limit(group2images_initial, max_total_spreads=max_total_spreads,
                                    min_total_spreads=min_total_spreads,logger = logger,
                                    target_total_spreads=target_total_spreads)

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
                                              min_total_spreads=min_total_spreads,logger = logger,
                                              per_group=True, target_total_spreads=target_total_spreads)

    if not is_wedding:
        kept = cap_non_wedding_groups(
            group2images, resources.look_up_table._compute_initial_spreads(group2images),
            max_total_spreads, logger)
        if len(kept) < len(group2images):
            group2images = {g: group2images[g] for g in kept}

    if CONFIGS['save_files']['spreads']:
        # Wipe stale per-group jsons from any previous run. Without this, files
        # whose group key isn't reused by the new run linger and mix into the
        # next visualization with photo ids that won't exist on disk.
        shutil.rmtree('files/stages_info/spreads', ignore_errors=True)
        os.makedirs('files/stages_info/spreads', exist_ok=True)
        with open('files/stages_info/spreads/_layouts.json', 'w') as f:
            json.dump(resources.printlab_data.to_dict(), f, indent=2, default=_json_default)

    # Same reason as above: stale per-subgroup traces from a previous run would
    # otherwise mix into combinations.pdf.
    reset_combinations_dir()

    fold.reset_tally()
    named_results = []
    for group_name in group2images.keys():
        spread_params = resources.look_up_table.get_current_spread_parameters(group_name, group2images[group_name])
        if spread_params[0] > 24:
            spread_params = (24.0, spread_params[1])
            
        pinned = (resources.look_up_table.pinned_spreads(group_name, group2images[group_name])
                  if hasattr(resources.look_up_table, 'pinned_spreads') else None)
        cur_result = process_group(group_name=group_name,
                                   group_images_df=updated_groups.get_group(group_name),
                                   spread_params=list(spread_params),
                                   resources=resources,
                                   is_wedding=is_wedding,
                                   params=params,
                                   logger=logger,
                                   pinned_spreads=pinned)
        if cur_result:
            named_results.append((group_name, cur_result))

    if not is_wedding:
        named_results = trim_non_wedding_results(named_results, group2images, max_total_spreads, logger)
    result_list = [result for _, result in named_results]

    logger.info(f'General groups processing time: {time.time() - start_time:.2f} seconds')

    moved, fold_before, fold_after = fold.tally()
    if moved:
        logger.info(f'fold: {moved} spread(s) moved off the fold '
                    f'({fold_before:.2f} -> {fold_after:.2f} faces on the fold)')

    # The layout scored every twin as colour-neutral; now that the pages are
    # fixed, each one's twins take the colour it needs.
    if CONFIGS.get('twin_colors', True):
        apply_twin_colors(result_list, logger)

    if is_wedding:
        # Group order first, then the one relaxation of it the clock can settle
        # on its own: a spread that is wholly earlier than the one in front of
        # it moves ahead, whatever group each belongs to.
        ordered = sort_groups_by_time(result_list, logger)
        return reorder_disjoint_spreads(ordered, logger), updated_photos_df
    else:
        return result_list, updated_photos_df
