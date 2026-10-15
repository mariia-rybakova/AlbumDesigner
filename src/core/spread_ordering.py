"""Putting the album's spreads into the order the day happened in.

The last word on sequence, after `spreads_layout` has decided which photos
share a spread. `sort_groups_by_time` orders whole groups by their median
time and the spreads inside a group by their first photo, which makes a group
atomic; `reorder_disjoint_spreads` then relaxes that, but only between two
spreads whose times cannot overlap, so a late spread of an early group can
fall back behind an early spread of a late one.

It works on `Spread` and `GroupProcessingResult` objects, long after the
DataFrame stage is over, and it runs after `spreads_layout` has returned --
which is why it sits here and not in that package: that is a tree entered at
its own `main`, and this is called from outside it.
"""

from statistics import median

from src.core.models import GroupProcessingResult, Spread


def _spread_time_range(spread):
    """`(first, last)` photo time in a spread, or None when it has no real photo.

    The dummy photo the layout search pads with carries `id == -1` and a
    sentinel time; it is skipped here exactly as `sort_groups_by_time` skips it.
    """
    times = [photo.general_time
             for photo in list(spread.left_photos) + list(spread.right_photos)
             if photo is not None and photo.id != -1 and photo.general_time is not None]
    return (min(times), max(times)) if times else None


def reorder_disjoint_spreads(groups_list, logger):
    """Move a spread ahead of the one before it when the two cannot overlap.

    `sort_groups_by_time` orders whole groups by their median time and the
    spreads inside a group by their first photo. A group is therefore an atomic
    block, and a spread can never pass one belonging to another group -- so a
    group that merely *ends* late is placed late in full, and its opening spread
    sits behind spreads shot after it.

    This pass relaxes that, but only where the times leave nothing to argue
    about: two spreads are swapped when every photo of the later one was taken
    before every photo of the earlier one. Spreads that overlap even by a second
    keep the order the group sort gave them, which is what holds a group
    together -- the spreads within one are almost always interleaved in time, so
    the rule is silent on them and only fires across a boundary.

    Groups stay contiguous wherever nothing moves. A spread that does move
    splits its group into two results carrying the same `group_name`, which the
    downstream page loop handles as it already handles the one-result-per-spread
    shape `predefined.processing` emits.

    Cost is `O(n^2)` comparisons of two precomputed floats, over the spreads of
    one album -- tens of them, not thousands. Each swap removes one pair that
    violates "entirely before" and introduces none, since the two spreads keep
    their order relative to everything else, so the pass terminates.
    """
    try:
        # Flatten to `(group_dict_index, group_id, spread, range)`, keeping each
        # spread's own group so a moved one can be filed back under it.
        flat = []
        for dict_idx, group_dict in enumerate(groups_list):
            for group_id, result in group_dict.items():
                if not isinstance(result, GroupProcessingResult):
                    continue
                for spread in result.spreads:
                    flat.append([dict_idx, group_id, result.group_name,
                                 spread, _spread_time_range(spread)])

        if len(flat) < 2:
            return groups_list

        swaps = 0
        changed = True
        while changed:
            changed = False
            for i in range(len(flat) - 1):
                earlier, later = flat[i][4], flat[i + 1][4]
                # Only an unambiguous pair moves: the whole of the later spread
                # before the whole of the earlier one. A spread with no usable
                # time never qualifies and so never moves.
                if earlier is not None and later is not None and later[1] < earlier[0]:
                    flat[i], flat[i + 1] = flat[i + 1], flat[i]
                    changed = True
                    swaps += 1

        if not swaps:
            return groups_list

        # Rebuild, merging consecutive spreads of the same group back into one
        # result. With no swaps this reproduces the input exactly; with swaps it
        # splits only the groups a spread actually crossed.
        rebuilt = []
        for dict_idx, group_id, group_name, spread, _ in flat:
            if rebuilt and rebuilt[-1][0] == (dict_idx, group_id):
                rebuilt[-1][1].spreads.append(spread)
            else:
                rebuilt.append(((dict_idx, group_id),
                                GroupProcessingResult(group_name=group_name, spreads=[spread])))

        logger.info(f"Reordered {swaps} spread pair(s) whose times do not overlap; "
                    f"{len(groups_list)} group block(s) became {len(rebuilt)}.")
        return [{group_id: result} for (_, group_id), result in rebuilt]

    except Exception as ex:
        logger.warning(f"Error reordering disjoint spreads: {ex}. Keeping the group order.")
        return groups_list


def sort_groups_by_time(groups_list, logger):
    try:
        groups_time_list = list()
        for group_dict in groups_list:
            photos_time_list = list()
            # group_dict is a Dict[str, GroupProcessingResult]
            for group_id, result in group_dict.items():
                if not isinstance(result, GroupProcessingResult):
                    continue
                
                # Sort spreads inside this result by min general_time of photos
                def spread_time_key(spread: Spread):
                    all_photos = spread.left_photos + spread.right_photos
                    times = [photo.general_time for photo in all_photos if photo and photo.id != -1]
                    return min(times) if times else float('inf')
                
                result.spreads.sort(key=spread_time_key)
                
                # Collect times for sorting the groups themselves
                for spread in result.spreads:
                    all_photos = spread.left_photos + spread.right_photos
                    for cur_photo in all_photos:
                        if cur_photo and cur_photo.id != -1:
                            photos_time_list.append(cur_photo.general_time)
            
            groups_time_list.append((group_dict, median(photos_time_list) if photos_time_list else float('inf')))
        
        groups_time_list.sort(key=lambda x: x[1])
        return [item[0] for item in groups_time_list]
    except Exception as ex:
        logger.warning(f"Error in sorting groups by time: {ex}. Return groups without sorting by time.")
        return groups_list
