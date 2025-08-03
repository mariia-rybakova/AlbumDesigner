# Lookup table with category preferences (mean, std)
import math
from utils.parser import CONFIGS

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
    'other': (6, 0.5),
    'invite': (6, 0.5),
    'None':(6,0.5),
    'wedding dress': (6,0.5),
    'vehicle':(6,0.5),
    'inside vehicle':(6,0.5)
}


spreads_wedding_lookup_table = {
    'bride and groom': (3, 0.5),
    'bride': (3, 0.5),
    'groom': (3, 0.5),
    'bride party': (1, 0.75),
    'groom party': (1, 0.75),
    'full party': (1, 0.5),
    'portrait': (2, 0.5),
    'very large group': (1, 0.5),
    'walking the aisle': (3, 0.75),
    'bride getting dressed': (1, 1),
    'first dance': (1, 0.5),
    'cake cutting': (1, 1),
    'ceremony': (2, 1),
    'couple': (0, 0),
    'dancing': (1, 1),
    'entertainment': (1, 1),
    'kiss': (1, 0.5),
    'pet': (0, 0),
    'accessories': (0, 0),
    'settings': (1, 0.5),
    'speech': (1, 1),
    'detail': (1, 1.5),
    'getting hair-makeup': (1, 1.5),
    'food': (1, 1.5),
    'other': (0, 0),
    'invite': (0, 0),
    'None':(0,0),
    'wedding dress': (0,0),
    'vehicle':(0,0),
    'inside vehicle':(0,0)
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


def calculate_flexible_mean(total_images,group_original_mean, max_per_spread=24):
    if (total_images / group_original_mean) <= 4:
        return group_original_mean
    else:
        min_per_spread = total_images / 2
        if total_images <= min_per_spread:
            return total_images

        # Use logarithmic scaling
        log_scale = math.log(total_images, 2)

        # Calculate the mean
        mean = min(max(log_scale, min_per_spread), max_per_spread)
        return int(mean)


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