import random
import numpy as np

from itertools import combinations, product, groupby, permutations

from utils.configs import CONFIGS
import pandas as pd

def classWeight(nPhotos, classSpredParams):
    # calculates the class contribution to score.
    # score is gaussian with provided array of [mean,std]
    # input parameter nPhotos is an array of number of photos for all spreads for specific context class
    # the result classWeight is the product of all gaussians for the context class

    nPhotos = np.array(nPhotos)
    nPhotos = nPhotos[nPhotos > 0]
    classWeight = np.prod(np.exp(-0.5 * np.power(((nPhotos - classSpredParams[0]) / classSpredParams[1]), 2)))
    return classWeight


def printAllUniqueParts(n):
    p = [0] * n  # An array to store a partition
    k = 0  # Index of last element in a partition
    p[k] = n  # Initialize first partition
    # as number itself

    # This loop first prints current partition,
    # then generates next partition.The loop
    # stops when the current partition has all 1s

    parts = []

    while True:

        parts.append(p[:k + 1].copy())
        # Generate next partition

        # Find the rightmost non-one value in p[].
        # Also, update the rem_val so that we know
        # how much value can be accommodated
        rem_val = 0
        while k >= 0 and p[k] == 1:
            rem_val += p[k]
            k -= 1

        # if k < 0, all the values are 1 so
        # there are no more partitions
        if k < 0:
            return parts

        # Decrease the p[k] found above
        # and adjust the rem_val
        p[k] -= 1
        rem_val += 1

        # If rem_val is more, then the sorted
        # order is violated. Divide rem_val in
        # different values of size p[k] and copy
        # these values at different positions after p[k]
        while rem_val > p[k]:
            p[k + 1] = p[k]
            rem_val = rem_val - p[k]
            k += 1

        # Copy rem_val to next position
        # and increment position
        p[k + 1] = rem_val
        k += 1


def selectPartitions(photos_df, classSpreadParams,params,layouts_df):
    # finds all available partitions for a class cluster of size nPhotos
    # eliminates unlikely partitions based ont the cluster class score parameters
    # parameter: nPhotos - total number of photos for a class cluster
    # parameter: classSpreadParams - array size 2 [mean,std] containing the gaussian parameter for the context class score

    nPhotos = len(photos_df.index)
    available_n = set(layouts_df['number of boxes'].unique())

    nPortrait = len(photos_df[photos_df['ar'] < 1].index)
    nLandscape = nPhotos - nPortrait

    classSpreadParams[1] = max(classSpreadParams[1], 0.5)

    parts = printAllUniqueParts(nPhotos)
    parts = [part for part in parts if set(part).issubset(available_n)]

    weights = np.zeros(len(parts))
    for idx, part in enumerate(parts):
        weights[idx] = classWeight(part, classSpreadParams)

    if np.all(weights == 0):
        classSpreadParams[1] = np.abs(nPhotos - classSpreadParams[0]) / 3
        for idx, part in enumerate(parts):
            weights[idx] = classWeight(part, classSpreadParams)
    else:
        weights /= np.max(weights)

    sorted_indices = np.argsort(weights)[::-1]
    parts = [parts[i] for i in sorted_indices]
    weights = weights[sorted_indices]

    layouts_dict = dict()
    for item in list(available_n):
        layouts_dict[item] = layouts_df[layouts_df['number of boxes']==item][['max portraits','max landscapes']].drop_duplicates()

    filtered_parts = []
    filtered_weights = []
    for idx1,part in enumerate(parts):

        part_landscape = nLandscape
        part_portrait = nPortrait

        for spread in part:
            n_layouts = layouts_dict[spread]
            match_layout=False
            for idx2, row in n_layouts.iterrows():
                rem_portrait = max(part_portrait - row['max portraits'],0)
                rem_landscape = max(part_landscape - row['max landscapes'],0)
                if (part_landscape+part_portrait)-spread >= (rem_portrait+rem_landscape):
                    match_layout=True
                    part_portrait = rem_portrait
                    part_landscape = rem_landscape
                    break
            if not match_layout:
                break
        if match_layout:
            filtered_parts.append(part)
            filtered_weights.append(weights[idx1])
            if len(filtered_parts)>2 and weights[idx1] < np.max(weights) / params[1]:
                break

    parts_len = [len(part) for part in filtered_parts]
    if len(parts_len) > 0 and np.max(parts_len) > np.min(parts_len) :
        idxs = [idx for idx, part in enumerate(filtered_parts) if (len(part) - np.min(parts_len) <= 1 and len(part)<=2 and nPhotos < 16) or (len(part) == np.min(parts_len)) ]
        partsAboveThresh = [filtered_parts[idx] for idx in idxs]
        weightsAboveThresh = [filtered_weights[idx] for idx in idxs]
    else:
        partsAboveThresh = filtered_parts
        weightsAboveThresh = filtered_weights

    # parts = filtered_parts
    #
    # #print("The partition_score_threshold is {} ".format(CONFIGS['partition_score_threshold']))
    # #aboveThresh = np.where(weights > np.max(weights) / CONFIGS['partition_score_threshold'])[0]
    # aboveThresh = np.where(weights > np.max(weights) / params[1])[0]
    # if len(weights) > 2:
    #     args = np.argsort(weights)[::-1]
    #     if len(aboveThresh) > 2:
    #         partsAboveThresh = [parts[args[idx]] for idx in range(len(aboveThresh))]
    #         weightsAboveThresh = [weights[args[idx]] for idx in range(len(aboveThresh))]
    #     else:
    #         partsAboveThresh = [parts[args[idx]] for idx in range(3)]
    #         weightsAboveThresh = [weights[args[idx]] for idx in range(3)]
    # else:
    #     partsAboveThresh = parts
    #     weightsAboveThresh = weights
    return partsAboveThresh, weightsAboveThresh


def partitions_with_swaps(seq, sizes, m):
    """
    Generate all partitions of `seq` into len(sizes) groups of given sizes.
    Order inside a group doesn't affect the cost.
    Cost = minimal #adjacent swaps needed to restore the default consecutive split,
           which equals the number of inversions between group labels along the
           original index order.

    Returns: list of (groups, swaps) where groups is a list of lists of elements.
    """
    n = len(seq)
    assert sum(sizes) == n, "sizes must sum to len(seq)"
    G = len(sizes)
    indices = tuple(range(n))

    # assignment over original positions: -1 = unassigned, else group id 0..G-1
    assign = [-1] * n
    assigned_positions = []  # list of positions already assigned
    results = []

    def add_inversions_for_new(pos, g):
        """Count inversions introduced by assigning position `pos` -> group `g`,
        against all previously assigned positions."""
        inc = 0
        for j in assigned_positions:
            gj = assign[j]
            if j < pos and gj > g:
                inc += 1
            elif j > pos and g > gj:
                inc += 1
        return inc

    def backtrack(group_id, remaining_idx_set, swaps_so_far, groups_idx):
        if swaps_so_far > m:
            return
        if group_id == G:
            # Build concrete groups (keep each group's indices in ascending original order)
            groups = []
            for gi, idxs in enumerate(groups_idx):
                groups.append([seq[i] for i in sorted(idxs)])
            results.append((groups, swaps_so_far))
            return

        s = sizes[group_id]
        rem_list = sorted(remaining_idx_set)
        # choose s indices (as a set) for this group
        for chosen in combinations(rem_list, s):
            # assign them (order within chosen doesn't change cost since same label)
            inc = 0
            # assign one-by-one so we can update swaps incrementally
            for pos in chosen:
                assign[pos] = group_id
                inc += add_inversions_for_new(pos, group_id)
                assigned_positions.append(pos)

            backtrack(
                group_id + 1,
                remaining_idx_set - set(chosen),
                swaps_so_far + inc,
                groups_idx + [chosen],
            )

            # undo
            for pos in chosen:
                assigned_positions.pop()  # last appended
                assign[pos] = -1

    backtrack(0, set(indices), 0, [])
    # sort nicely (by swaps, then lexicographically)
    results.sort(key=lambda x: (x[1], x[0]))
    return results


def listSingleCombinations(photos, layout_part, maxCombs):
    photos_ids = list(range(len(photos)))
    photos_ids = set(photos_ids)

    if len(layout_part) > 1:
        layout_combs = partitions_with_swaps(list(photos_ids), layout_part, 2)
        layout_combs = [[set(part) for part in comb] for comb,v in layout_combs]
    else:
        l0_combs = list(combinations(photos_ids, layout_part[0]))

        if len(l0_combs) > maxCombs:
            sample_idxs = random.sample(range(len(l0_combs)), maxCombs)
            l0_combs = [l0_combs[i] for i in sample_idxs]

        l0_combs = [[set(l0_comb)] for l0_comb in l0_combs]
        rem_photos = [photos_ids - l0_comb[0] for l0_comb in l0_combs]
        layout_combs = l0_combs

        for layout_index in range(1, len(layout_part) - 1):
            merged_combs = []
            merged_rem_photos = []
            for comb_idx in range(len(layout_combs)):
                next_combs = list(combinations(rem_photos[comb_idx], layout_part[layout_index]))
                next_combs = [set(next_comb) for next_comb in next_combs]
                if len(layout_combs)>maxCombs:
                    next_combs=[next_combs[0]]
                single_comb = [layout_combs[comb_idx].copy() for _ in range(len(next_combs))]
                single_rem_photos = [rem_photos[comb_idx].copy() for _ in range(len(next_combs))]
                for single_idx in range(len(single_comb)):
                    single_comb[single_idx].append(next_combs[single_idx])
                    single_rem_photos[single_idx] = single_rem_photos[single_idx] - next_combs[single_idx]
                merged_combs += single_comb
                merged_rem_photos += single_rem_photos
            layout_combs = merged_combs
            rem_photos = merged_rem_photos
        if len(layout_part) > 1:
            if len(layout_combs) > maxCombs:
                #print(f"Sampling {maxCombs} combinations from {len(layout_combs)}")
                sample_idxs = random.sample(range(len(layout_combs)), maxCombs)
                layout_combs = [layout_combs[i] for i in sample_idxs]
                # rem_photos = [rem_photos[i] for i in sample_idxs]
            for comb_idx in range(len(layout_combs)):
                layout_combs[comb_idx].append(rem_photos[comb_idx])

    return layout_combs


def greedy_combination_search(photos, layout_part, layout_df):
    photos_ids = list(range(len(photos)))

    n_photos = len(photos)
    landscapes = 0
    landscape_photos_ids = list()
    portrait_photos_ids = list()

    for i in range(len(photos)):
        if photos[i].ar < 1:
            portrait_photos_ids.append(photos_ids[i])
        else:
            landscapes += 1
            landscape_photos_ids.append(photos_ids[i])
    portraits = n_photos - landscapes
    context2photos_number = dict()
    for photo in photos:
        if photo.original_context not in context2photos_number:
            context2photos_number[photo.original_context] = list()
        context2photos_number[photo.original_context].append(photo.general_time)
    landscape_photos_ids = sorted(landscape_photos_ids, key=lambda x: [np.mean(context2photos_number[photos[x].original_context]), photos[x].general_time])
    portrait_photos_ids = sorted(portrait_photos_ids, key=lambda x: [np.mean(context2photos_number[photos[x].original_context]), photos[x].general_time])

    spread_layouts_list = list()
    for spread_size in layout_part:
        layouts = layout_df.loc[(layout_df['number of boxes'] == spread_size)]
        # &
        # (len(layout_df['left_portrait_ids']) + len(layout_df['right_portrait_ids']) <= portraits) &
        # (len(layout_df['left_landscape_ids']) + len(layout_df['right_landscape_ids']) <= landscapes)
        list_of_single_row_layouts = []
        for index, row in layouts.iterrows():
            single_row_df = row.to_frame().T
            list_of_single_row_layouts.append(single_row_df)
        spread_layouts_list.append(list_of_single_row_layouts)

    if len(spread_layouts_list) >= 4:
        all_combinations_of_layouts = [[sublist[0] for sublist in spread_layouts_list]]
    else:
        all_combinations_of_layouts = list(product(*spread_layouts_list))
    if len(all_combinations_of_layouts) > 1000:
        all_combinations_of_layouts = random.sample(all_combinations_of_layouts, 1000)

    final_layout_combs_list = list()
    for layouts_comb in all_combinations_of_layouts:
        total_number_of_boxes = sum([int(cur_layout['number of boxes'].iloc[0]) for cur_layout in layouts_comb])
        total_number_of_portraits = sum([len(cur_layout['left_portrait_ids'].iloc[0]) + len(cur_layout['right_portrait_ids'].iloc[0]) for cur_layout in layouts_comb])
        total_number_of_landscapes = sum([len(cur_layout['left_landscape_ids'].iloc[0]) + len(cur_layout['right_landscape_ids'].iloc[0]) for cur_layout in layouts_comb])
        if total_number_of_boxes != n_photos or total_number_of_portraits > portraits or total_number_of_landscapes > landscapes:
            continue

        cur_comb = [set() for _ in range(len(layouts_comb))]
        portraits_idx = 0
        landscapes_idx = 0
        for cur_idx, cur_layout in enumerate(layouts_comb):
            for _ in range(len(cur_layout['left_portrait_ids'].iloc[0]) + len(cur_layout['right_portrait_ids'].iloc[0])):
                cur_comb[cur_idx].add(portrait_photos_ids[portraits_idx])
                portraits_idx += 1
        for cur_idx, cur_layout in enumerate(layouts_comb):
            for _ in range(len(cur_layout['left_landscape_ids'].iloc[0]) + len(cur_layout['right_landscape_ids'].iloc[0])):
                cur_comb[cur_idx].add(landscape_photos_ids[landscapes_idx])
                landscapes_idx += 1

        # add squares
        for cur_idx, cur_layout in enumerate(layouts_comb):
            while len(cur_comb[cur_idx]) < int(cur_layout['number of boxes'].iloc[0]):
                if portraits_idx == len(portrait_photos_ids) and landscapes_idx == len(landscape_photos_ids):
                    raise Exception('Something wrong. Not enough photos in greedy layouts search.')
                elif portraits_idx == len(portrait_photos_ids):
                    cur_comb[cur_idx].add(landscape_photos_ids[landscapes_idx])
                    landscapes_idx += 1
                elif landscapes_idx == len(landscape_photos_ids):
                    cur_comb[cur_idx].add(portrait_photos_ids[portraits_idx])
                    portraits_idx += 1
                else:
                    next_portrait = photos[portrait_photos_ids[portraits_idx]].general_time
                    next_landscape = photos[landscape_photos_ids[landscapes_idx]].general_time
                    if next_portrait < next_landscape:
                        cur_comb[cur_idx].add(portrait_photos_ids[portraits_idx])
                        portraits_idx += 1
                    else:
                        cur_comb[cur_idx].add(landscape_photos_ids[landscapes_idx])
                        landscapes_idx += 1
        final_layout_combs_list.append(cur_comb)

    cleaned_comb_data = []
    seen = set()

    for inner_list in final_layout_combs_list:
        frozen_inner_list = tuple(frozenset(s) for s in inner_list)

        if frozen_inner_list not in seen:
            seen.add(frozen_inner_list)
            cleaned_comb_data.append(inner_list)
    return cleaned_comb_data


def layoutSingleCombination(singleClassComb, layout_df, photos,params):
    n_spreads = len(singleClassComb)
    multi_spreads = []
    for spread_idx in range(n_spreads):
        spread_photos = list(singleClassComb[spread_idx])
        if len(spread_photos) == 0:
            spread_photos
        landscape_set = set()
        portrait_set = set()

        n_photos = len(spread_photos)
        landscapes = 0

        for i in range(len(spread_photos)):
            if photos[spread_photos[i]].ar < 1:
                portrait_set.add(spread_photos[i])
            else:
                landscapes += 1
                landscape_set.add(spread_photos[i])

        portraits = n_photos - landscapes

        layouts = layout_df.loc[
            (layout_df['number of boxes'] == n_photos) & (layout_df['max portraits'] >= portraits) & (
                    layout_df['max landscapes'] >= landscapes)].copy()
        
        if not layouts.empty:
            layouts['number of squares'] = layouts.apply(lambda x: len(list(x['left_square_ids'])) + len(list(x['right_square_ids'])), axis=1)
        else:
            layouts['number of squares'] = 0

        # large spreads with squares gets trivial layout
        if n_photos > 13 and len(layouts[layouts['number of squares']==n_photos]) > 0 and n_spreads == 1:
            selectedLayout = layouts[layouts['number of squares']==n_photos]
            single_spreads=[]
            for layout_idx, layout in selectedLayout.iterrows():
                single_spreads.append([layout_idx, set(range(0,len(layout['left_square_ids']))),set(range(len(layout['left_square_ids']),n_photos)),n_photos])
            multi_spreads.append(single_spreads)
            return multi_spreads

        ### greedy attempt to find layout based on seperation of time, class and color

        try:
            if len(layouts) > 0 :
                greedy_layouts = layouts.copy()

                greedy_layouts['max_left_portraits'] = greedy_layouts.apply(
                    lambda x: len(list(x['left_portrait_ids'])) + len(list(x['left_square_ids'])), axis=1)
                greedy_layouts['max_left_landscapes'] = greedy_layouts.apply(
                    lambda x: len(list(x['left_landscape_ids'])) + len(list(x['left_square_ids'])), axis=1)
                greedy_layouts['max_right_portraits'] = greedy_layouts.apply(
                    lambda x: len(list(x['right_portrait_ids'])) + len(list(x['right_square_ids'])), axis=1)
                greedy_layouts['max_right_landscapes'] = greedy_layouts.apply(
                    lambda x: len(list(x['right_landscape_ids'])) + len(list(x['right_square_ids'])), axis=1)
                greedy_layouts['left_total_capacity'] = greedy_layouts.apply(
                    lambda x: len(list(x['left_portrait_ids'])) + len(list(x['left_landscape_ids'])) + len(list(x['left_square_ids'])), axis=1)
                greedy_layouts['right_total_capacity'] = greedy_layouts.apply(
                    lambda x: len(list(x['right_portrait_ids'])) + len(list(x['right_landscape_ids'])) + len(list(x['right_square_ids'])), axis=1)

                greedy_single_spreads = []
                time_sequeces = [(photo_id, photos[photo_id].general_time, (photos[photo_id].original_context,photos[photo_id].color)) for photo_id in spread_photos]
                time_sequeces = sorted(time_sequeces, key=lambda x: x[1])

                grouped = groupby(time_sequeces, key=lambda x: x[2])

                grouped_sequences = []
                for key, group in grouped:
                    grouped_sequences.append(list(group))
                if len(grouped_sequences) == 2:
                    left_landscapes = np.sum([photos[item[0]].ar > 1 for item in grouped_sequences[0]])
                    left_portraits = len(grouped_sequences[0]) - left_landscapes
                    right_landscapes = np.sum([photos[item[0]].ar > 1 for item in grouped_sequences[1]])
                    right_portraits = len(grouped_sequences[1]) - right_landscapes

                    mask = ((greedy_layouts['max_left_landscapes'] >= left_landscapes) &
                            (greedy_layouts['max_left_portraits'] >= left_portraits) &
                            (greedy_layouts['max_right_landscapes'] >= right_landscapes) &
                            (greedy_layouts['max_right_portraits'] >= right_portraits) &
                            ((left_landscapes + left_portraits) == greedy_layouts['left_total_capacity']) &
                            ((right_landscapes + right_portraits) == greedy_layouts['right_total_capacity']))
                    possible_layouts = greedy_layouts.loc[mask]
                    for layout_idx , layout in possible_layouts.iterrows():
                        greedy_single_spreads.append([layout_idx, set([item[0] for item in grouped_sequences[0]]), set([item[0] for item in grouped_sequences[1]]), len(list(layout['left_square_ids']) + list(layout['right_square_ids']))])

                colors = [photos[photo_id].color for photo_id in spread_photos]
                if len(set(colors)) == 2:
                    photos_color = np.array([photos[photo_id].color for photo_id in spread_photos])
                    color_time = np.mean([photos[photo_id].general_time for photo_id in spread_photos if photos[photo_id].color])
                    gray_time = np.mean([photos[photo_id].general_time for photo_id in spread_photos if not photos[photo_id].color])
                    if gray_time > color_time:
                        left_condition = True
                    else:
                        left_condition = False
                    left_landscapes = np.sum([photos[item].ar > 1 and photos[item].color == left_condition for item in spread_photos])
                    left_portraits = np.sum(photos_color==left_condition) - left_landscapes
                    right_landscapes = np.sum([photos[item].ar > 1 and photos[item].color != left_condition for item in spread_photos])
                    right_portraits = np.sum(photos_color!=left_condition) - right_landscapes

                    mask = ((greedy_layouts['max_left_landscapes'] >= left_landscapes) &
                            (greedy_layouts['max_left_portraits'] >= left_portraits) &
                            (greedy_layouts['max_right_landscapes'] >= right_landscapes) &
                            (greedy_layouts['max_right_portraits'] >= right_portraits) &
                            ((left_landscapes + left_portraits) == greedy_layouts['left_total_capacity']) &
                            ((right_landscapes + right_portraits) == greedy_layouts['right_total_capacity']))
                    possible_layouts = greedy_layouts.loc[mask]

                    for layout_idx, layout in possible_layouts.iterrows():
                        greedy_single_spreads.append([layout_idx, set([photo_id for photo_id in spread_photos if photos[photo_id].color == left_condition]),
                                                      set([photo_id for photo_id in spread_photos if photos[photo_id].color != left_condition]),
                                                      len(list(layout['left_square_ids']) + list(layout['right_square_ids']))])
            else:
                greedy_single_spreads = []
        except Exception as e:
            greedy_single_spreads = []
            print(f"Greedy layout attempt failed with error {e}")

        # greedy_single_spreads=[]

        spreads = []
        for layout in layouts.index:
            left_pages = list()
            right_pages = list()
            left_landscapes = len(layouts.at[layout, 'left_landscape_ids'])
            left_portraits = len(layouts.at[layout, 'left_portrait_ids'])
            landscape_combs = list(combinations(landscape_set, left_landscapes))
            portrait_combs = list(combinations(portrait_set, left_portraits))
            oriented_combs = list(product(landscape_combs, portrait_combs))
            rem_landscapes = []
            rem_portraits = []
            #print(f"CONFIGS['MaxOrientedCombs'] is {CONFIGS['MaxOrientedCombs']}")
            #if len(oriented_combs) > CONFIGS['MaxOrientedCombs']:
            if len(oriented_combs) > params[4]:
                # print('MaxOrientedCombs crossed sampling oriented combinations instead of full listing')
                #sample_idxs = random.sample(range(len(oriented_combs)), CONFIGS['MaxOrientedCombs'])
                sample_idxs = random.sample(range(len(oriented_combs)), params[4])
                oriented_combs = [oriented_combs[i] for i in sample_idxs]

            for comb in oriented_combs:

                # single_left = set()
                #
                # for landscape in comb[0]:
                #     single_left.add(landscape)

                single_left = set(comb[0])

                if len(comb[0]) == 0:
                    rem_landscapes.append(landscape_set)
                else:
                    rem_landscapes.append(landscape_set - set(comb[0]))

                for portrait in comb[1]:
                    single_left.add(portrait)

                if len(comb[1]) == 0:
                    rem_portraits.append(portrait_set)
                else:
                    rem_portraits.append(portrait_set - set(comb[1]))
                left_pages.append(single_left)

            right_landscapes = len(layouts.at[layout, 'right_landscape_ids'])
            right_portraits = len(layouts.at[layout, 'right_portrait_ids'])

            oriented_spreads = []
            rem_right_landscapes = []
            rem_right_portraits = []
            for idx, left_set in enumerate(left_pages):
                landscape_combs = list(combinations(rem_landscapes[idx], right_landscapes))
                portrait_combs = list(combinations(rem_portraits[idx], right_portraits))
                oriented_combs = list(product(landscape_combs, portrait_combs))

                for comb in oriented_combs:
                    # single_right = set()

                    single_right = set(comb[0])

                    # for landscape in comb[0]:
                    #     single_right.add(landscape)

                    if len(comb[0]) == 0:
                        rem_right_landscapes.append(rem_landscapes[idx])
                    else:
                        rem_right_landscapes.append(rem_landscapes[idx] - set(comb[0]))

                    for portrait in comb[1]:
                        single_right.add(portrait)

                    if len(comb[1]) == 0:
                        rem_right_portraits.append(rem_portraits[idx])
                    else:
                        rem_right_portraits.append(rem_portraits[idx] - set(comb[1]))
                    oriented_spreads.append([left_set, single_right])

            left_squares = len(layouts.at[layout, 'left_square_ids'])
            right_squares = len(layouts.at[layout, 'right_square_ids'])

            if len(oriented_spreads) != len(rem_right_landscapes):
                rem_right_landscapes

            single_spreads = []
            single_spreads = greedy_single_spreads.copy()
            for idx, oriented_spread in enumerate(oriented_spreads):
                rem_photos = rem_right_landscapes[idx].union(rem_right_portraits[idx])
                landscape_left_combs = list(combinations(rem_photos, left_squares))
                for comb in landscape_left_combs:
                    single_spreads.append(
                        [layout, oriented_spread[0].union(set(comb)), oriented_spread[1].union(rem_photos) - set(comb),
                         left_squares + right_squares])

            spreads += single_spreads
        if len(spreads) == 0:
            return None
        if len(spreads) > params[2]:
            # print(f"Sampling {params[2]} spreads from {len(spreads)}")
            sample_idxs = random.sample(range(len(spreads)), params[2])
            spreads = [spreads[i] for i in sample_idxs]
        multi_spreads.append(spreads)

    return multi_spreads


def check_page(photo_set, photos):
    bride_centric_classes = ['bride', 'bride party', 'wedding dress', 'getting hair-makeup','bride getting dressed']
    groom_centric_classes = ['groom','groom party','suit']
    if len(photo_set) == 1:
        return [True, True, False, 1]
    else:
        colors = []
        photo_classes = []
        contexts = []
        for photo_id in photo_set:
            photo = photos[photo_id]
            colors.append(photo.color)
            photo_classes.append(photo.photo_class)
            contexts.append(photo.original_context)
        number_of_unique_contexts = len(set(contexts))
        sameColor = all([color == colors[0] for color in colors])
        sameClass = all([photo_class == photo_classes[0] for photo_class in photo_classes])
        if not sameClass:
            bride_centric = any([photo_class in bride_centric_classes for photo_class in photo_classes])
            groom_centric = any([photo_class in groom_centric_classes for photo_class in photo_classes])
            if bride_centric and groom_centric:
                bride_groom_mix = True
            else:
                bride_groom_mix = False
        else:
            bride_groom_mix = False
        return [sameColor, sameClass, bride_groom_mix, number_of_unique_contexts]


def eval_multi_spreads(multi_spreads, layouts_df, photos, comb_weight, crop_penalty=0.5, color_mix=0.000000001,
                       class_mix=0.01,
                       orientation_mix=0.1, score_threshold=0.01, double_mix_color=0.000000000000000001, context_mix_penalty=0.00001,time_order_penalty=0.005):
    #print(f"the CONFIGS['spread_score_threshold'] is {score_threshold}")
    filtered_multi_spreads = []
    for i in range(len(multi_spreads)):
        spread_scores = np.ones(len(multi_spreads[i]))
        for j in range(len(multi_spreads[i])):
            spread = multi_spreads[i][j]
            left_check = check_page(spread[1], photos)
            if not left_check[0]:
                spread_scores[j] = spread_scores[j] * color_mix
            if not left_check[1]:
                spread_scores[j] = spread_scores[j] * class_mix
            if left_check[2]:
                spread_scores[j] = spread_scores[j] * color_mix
            spread_scores[j] = spread_scores[j] * np.power(context_mix_penalty, max(1,left_check[3]) - 1)
            if layouts_df.at[spread[0], 'left_mixed']:
                spread_scores[j] = spread_scores[j] * orientation_mix
            right_check = check_page(spread[2], photos)
            if not right_check[0]:
                spread_scores[j] = spread_scores[j] * color_mix
            if not right_check[1]:
                spread_scores[j] = spread_scores[j] * class_mix
            if right_check[2]:
                spread_scores[j] = spread_scores[j] * color_mix
            if layouts_df.at[spread[0], 'right_mixed']:
                spread_scores[j] = spread_scores[j] * orientation_mix
            spread_scores[j] = spread_scores[j] * np.power(context_mix_penalty, max(1,right_check[3]) - 1)
            if not left_check[0] and not right_check[0]:
                # if two pages has gray colors give it much more worse
                spread_scores[j] = spread_scores[j] * double_mix_color
            spread_scores[j] = spread_scores[j] * np.power(crop_penalty, spread[3])
            photo_order_time = [photos[photo_id].general_time for photo_id in list(spread[1])+list(spread[2])]
            for time_idx1 in range(len(photo_order_time)):
                for time_idx2 in range(time_idx1 + 1, len(photo_order_time)):
                    if photo_order_time[time_idx1] > photo_order_time[time_idx2]:
                        spread_scores[j] = spread_scores[j] * time_order_penalty  # if time order is not correct, give it a penalty
        if len(spread_scores) > 0:
            filtered_idx = np.where(spread_scores / np.max(spread_scores) > score_threshold)[0]
            filtered_multi_spreads.append([multi_spreads[i][j] + [spread_scores[j]] for j in filtered_idx])

    filtered_multi_spreads.append(comb_weight)
    return filtered_multi_spreads


def list_multi_spreads(multi_spread):
    listed_spreads = []
    multi_spread_weight = multi_spread[-1]
    n_spreads = len(multi_spread) - 1
    if n_spreads == 1:
        for spread in multi_spread[0]:
            listed_spreads.append([[spread], spread[-1] * multi_spread_weight])
    else:
        merged = list(product(multi_spread[0], multi_spread[1]))
        merged = [[merged[idx][0], merged[idx][1]] for idx in range(len(merged))]
        for spread in range(2, n_spreads):
            merged = list(product(merged, multi_spread[spread]))
            merged = [merged[idx][0] + [merged[idx][1]] for idx in range(len(merged))]

        for merge in merged:
            merge_score = 1
            for spread in merge:
                merge_score *= spread[-1]
            listed_spreads.append([merge, merge_score * multi_spread_weight])

    return listed_spreads


def eval_single_comb(comb, photo_times, cluster_labels):
    score = 1
    for spread in comb:

        spread_times = [photo_times[id]/60.0 for id in spread]
        spread_labels = [cluster_labels[id] for id in spread]

        time_std = np.std(spread_times)
        if time_std > 0.0001:
            score /= time_std
        if not np.all(np.array(spread_labels) == None):
            score /= (1 + len(spread_labels) - len(set(spread_labels)))
    return score


def generate_filtered_multi_spreads(photos, layouts_df, spread_params,params,logger):
    photos_df = pd.DataFrame([photo.__dict__ for photo in photos])
    photos_df = photos_df.sort_values('general_time')
    layout_parts, weight_parts = selectPartitions(photos_df, spread_params,params,layouts_df=layouts_df)
    # logger.info('Number of photos: {}. Possible partitions: {}'.format(len(photos), layout_parts))

    combs = []
    comb_weights = np.array([])

    photoTimes = [item.general_time for item in photos]
    cluster_labels = [item.cluster_label for item in photos]
    # print("inside the genereatge filtered multi spreads")
    for i in range(len(layout_parts)):
        maxCombsParam = params[2] if len(photos) <= params[5] else params[3]

        maxCombs = int(maxCombsParam / np.power(2, i))
        if len(photos) <= 8 and len(photos) / spread_params[0] <= 2:
            single_combs = listSingleCombinations(photos, layout_parts[i],maxCombs)
        else:
            single_combs = greedy_combination_search(photos, layout_parts[i], layouts_df)
        # print(f"Single Combinations {len(single_combs)} and maxCombs {maxCombs}")

        if len(single_combs) > maxCombs:
            #logger.info('combinations Found {}, sampled {} combinations foe evaluation'.format(len(single_combs), maxCombs))
            sample_idxs = random.sample(range(len(single_combs)), maxCombs)
            single_combs = [single_combs[sample_idx] for sample_idx in sample_idxs]

        single_weights = []
        for single_comb in single_combs:
            single_weights.append(eval_single_comb(single_comb, photoTimes, cluster_labels))
        combs += single_combs
        comb_weights = np.append(comb_weights, np.array(single_weights) * weight_parts[i])
    #print("Getting the filtered multi srpreads")
    filtered_multi_spreads = []
    for idx, comb in enumerate(combs):
        multi_spreads = layoutSingleCombination(comb, layouts_df, photos,params)
        if multi_spreads is not None:
            if len(photos)<13:
                single_filtered_multi_spreads = eval_multi_spreads(multi_spreads, layouts_df, photos, comb_weights[idx],
                                                                   crop_penalty=CONFIGS['crop_penalty'], color_mix=CONFIGS['color_mix'], class_mix=CONFIGS['class_mix'],
                                                                   orientation_mix=CONFIGS['orientation_mix'], score_threshold=params[0], double_mix_color=CONFIGS['double_page_color_mix'])
            else:
                single_filtered_multi_spreads = eval_multi_spreads(multi_spreads, layouts_df, photos, comb_weights[idx],
                                                                   crop_penalty=0.8,
                                                                   color_mix=CONFIGS['color_mix'],
                                                                   class_mix=CONFIGS['class_mix'],
                                                                   orientation_mix=CONFIGS['orientation_mix'],
                                                                   score_threshold=params[0],
                                                                   double_mix_color=CONFIGS['double_page_color_mix'],
                                                                   context_mix_penalty=0.00001,time_order_penalty=0.5)
            filtered_multi_spreads += list_multi_spreads(single_filtered_multi_spreads)

        if len(filtered_multi_spreads) > 10000:
            scores = np.zeros(len(filtered_multi_spreads))
            for multi_spread in range(len(filtered_multi_spreads)):
                scores[multi_spread] = filtered_multi_spreads[multi_spread][1]

            args = np.argsort(scores)[::-1]
            filtered_multi_spreads = [filtered_multi_spreads[args[idx]] for idx in range(1000)]

    if len(filtered_multi_spreads) == 0:
        return None

    scores = np.zeros(len(filtered_multi_spreads))
    for multi_spread in range(len(filtered_multi_spreads)):
        scores[multi_spread] = filtered_multi_spreads[multi_spread][1]

    filtered_scores_idx = np.where(scores / np.max(scores) > 0.01)[0]

    if len(filtered_scores_idx) < 1000:
        filtered_scores = [filtered_multi_spreads[idx] for idx in filtered_scores_idx]
    else:
        args = np.argsort(scores)[::-1]
        filtered_scores = [filtered_multi_spreads[args[idx]] for idx in range(1000)]

    return filtered_scores

#
# if __name__ == '__main__':
#     from utils.load_layouts import load_layouts
#
#     # photos = [[9343321997, 1.499531396438613, True, 0.1180563190791846, 'other', 224, 649.55],
#     #           [9343322004, 1.499531396438613, True, 0.0, 'other', 54, 650.4333333333333],
#     #           [9343322008, 1.499531396438613, True, 0.079553271631674, 'other', 93, 652.2833333333333],
#     #           [9343322026, 1.499531396438613, True, 0.08415017407639093, 'other', 35, 656.6333333333333],
#     #           [9343322033, 1.499531396438613, True, 0.1736544203317863, 'other', 106, 659.05]]
#
#     # photos = [['9343140830.jpg', 1.5023474178403755, True, 0, 'dancing', 1, 0.0], ['9343140831.jpg', 1.5023474178403755, True, 0.3695769766350085, 'dancing', 1, 0.0], ['9343140832.jpg', 1.5023474178403755, True, 0, 'dancing', 1, 0.0], ['9343140836.jpg', 1.5023474178403755, True, 0, 'dancing', 1, 0.0], ['9343140844.jpg', 1.5023474178403755, True, 0, 'dancing', 1, 0.0], ['9343140845.jpg', 1.5023474178403755, True, 0.4967730705598245, 'dancing', 1, 0.0], ['9343140846.jpg', 1.5023474178403755, True, 0.4025748576952515, ''nan''], ['9343140847.jpg', 1.5023474178403755, True, 0, 'dancing', 1, 0.0], ['9343140848.jpg', 1.5023474178403755, True, 0, 'dancing', 1, 0.0], ['9343140850.jpg', 1.5023474178403755, True, 0, 'dancing', 1, 0.0], ['9343140851.jpg', 1.5023474178403755, True, 0.486304249068718, 'dancing', 1, 0.0], ['9343140852.jpg', 1.5023474178403755, True, 0, 'dancing', 1, 0.0], ['9343140865.jpg', 1.5023474178403755, True, 0, 'dancing', 1, 0.0], ['9343140866.jpg', 1.5023474178403755, True, 0, 'dancing', 1, 0.0], ['9343140870.jpg', 1.5023474178403755, True, 0.2742981079633911, ''nan''], ['9343140903.jpg', 1.5023474178403755, True, 0.311074292831681, ''`nan`''], ['9343140911.jpg', 1.5023474178403755, True, 0, 'dancing', 1, 0.0], ['9343140934.jpg', 1.5023474178403755, True, 0, 'dancing', 1, 0.0], ['9343140935.jpg', 1.5023474178403755, True, 0, 'dancing', 1, 0.0], ['9343140936.jpg', 1.5023474178403755, True, 0, 'dancing', 1, 0.0], ['9343140939.jpg', 1.5023474178403755, True, 0, 'dancing', 1, 0.0], ['9343140940.jpg', 1.5023474178403755, True, 0, 'dancing', 1, 0.0], ['9343141066.jpg', 1.5023474178403755, True, 0.5152995470735113, 'dancing', 1, 0.0], ['9343141067.jpg', 1.5023474178403755, True, 0, 'dancing', 1, 0.0]]
#     nana = [[9939826860, 0.6661538481712341, False, 266, 'bride and groom_1_2_1_3_1_4_1_5', 23, 736.0],
#              [9939826889, 1.5011547803878784, False, 264, 'bride and groom_1_2_1_3_1_4_1_5', 21, 674.0],
#              [9939826890, 1.5011547803878784, True, 324, 'bride and groom_1_2_1_3_1_4_1_5', 21, 692.0],
#              [9939827046, 1.5011547803878784, True, 132, 'bride and groom_1_2_1_3_1_4_1_5', 26, 683.0],
#              [9939827112, 1.5011547803878784, True, 326, 'bride and groom_1_2_1_3_1_4_1_5', 24, 696.0]]
#
#     _photos = [Photo.from_array(nana[idx]) for idx in range(len(nana))]
#
#     _layouts_df = load_layouts(r'C:\Users\karmel\Desktop\PicTime\Projects\AlbumDesigner\results\layout_csv\output.csv')
#     _spread_params = [4, 0.5]
#     generate_filtered_multi_spreads(_photos, _layouts_df, _spread_params,None)
