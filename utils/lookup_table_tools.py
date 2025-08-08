# Lookup table with category preferences (mean, std)
import math
from utils.configs import CONFIGS

wedding_lookup_table = {
    'bride and groom': (4, 0.5),
    'bride': (4, 0.5),
    'groom': (4, 0.5),
    'bride party': (6, 0.75),
    'groom party': (6, 0.75),
    'full party': (4, 0.5),
    'large_portrait': (2, 0.5),
    'small_portrait': (5, 0.5),
    'portrait': (5, 0.5),
    'very large group': (2, 0.5),
    'walking the aisle': (4, 0.75),
    'bride getting dressed': (9, 1),
    'first dance': (4, 0.5),
    'cake cutting': (6, 1),
    'ceremony': (5, 1),
    'couple': (6, 1),
    'dancing': (24, 1),
    'entertainment': (1, 1),
    'kiss': (4, 0.5),
    'pet': (4, 0.5),
    'accessories': (10, 0.5),
    'settings': (10, 0.5),
    'speech': (6, 1),
    'detail': (10, 1.5),
    'getting hair-makeup': (10, 1.5),
    'food': (10, 1.5),
    'other': (2, 0.5),
    'invite': (6, 0.5),
    'None':(2,0.5),
    'wedding dress': (6,0.5),
    'vehicle':(6,0.5),
    'inside vehicle':(6,0.5),
    'rings': (3, 0.5),
    'suit': (3, 0.5),
}

non_wedding_lookup_table = {
    '1':(2,0.4),
    '2':(2,0.4),
    '3':(2,0.4),
    '4':(3,0.4),
    '5':(4,0.5),
    '6':(4,0.5),
    '7':(4,0.5),
    '8':(4,0.5),
    '9':(4,0.5),
    '10':(2,0.5),
}


def get_lookup_table(group2images, is_wedding, logger=None,density=3):
    density_factors = CONFIGS['density_factors']

    try:
        if is_wedding:
            lookup_table = wedding_lookup_table
        else:
            lookup_table = non_wedding_lookup_table

        max_per_spread = 24

        for group_name, num_images in group2images.items():
            if is_wedding:
                group_id = group_name[1]  # Extract group ID
            else:
                group_id = group_name[0].split('_')[0]

            # Assign default values if group_id is not in lookup_table
            if group_id not in lookup_table:
                lookup_table[group_id] = (10, 4)

            lookup_table[group_id] = (max(1,min(max_per_spread, lookup_table[group_id][0]* density_factors[density])),lookup_table[group_id][1])



        return lookup_table  # Return the updated lookup table

    except Exception as e:
        logger.error(f"Error: Unexpected error while updating lookup table: {str(e)}")
        return None


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
    # First pass: Calculate initial spreads per group and total spreads
    total_spreads = 0
    spreads_per_group = {}

    for key, number_images in group2images.items():
        # Get spread parameters for the current group
        spread_params = get_current_spread_parameters(key, number_images, is_wedding, lookup_table)
        spreads = math.ceil(number_images / spread_params[0])  # Calculate required spreads for this group

        # Store the calculated spreads and add to the total
        spreads_per_group[key] = spreads
        total_spreads += spreads

    # If the total spreads exceed the limit, start reducing spreads
    if total_spreads > max_total_spreads:
        # Sort groups by the number of spreads in descending order (reduce larger groups first)
        sorted_groups = sorted(spreads_per_group.items(), key=lambda x: x[1], reverse=True)

        excess_spreads = total_spreads - max_total_spreads

        for key, current_spreads in sorted_groups:
            if excess_spreads <= 0:
                break  # Exit if we've reduced enough spreads

            # Try reducing spreads for this group
            new_spreads = max(1, current_spreads - 1)  # Ensure at least one spread per group
            reduction = current_spreads - new_spreads

            spreads_per_group[key] = new_spreads

            # Update the group in the lookup table
            content_key = key[1].split("_")[0] if is_wedding and "_" in key[1] else key[1] if is_wedding else key[0].split("_")[0]
            current_max_images, extra_value = lookup_table[content_key]
            value_to_change = math.ceil(group2images[key] / new_spreads)
            if value_to_change > current_max_images:
                lookup_table[content_key] = (value_to_change, extra_value)
                excess_spreads -= reduction

    return lookup_table
