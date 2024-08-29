# Lookup table with category preferences (mean, std)
import math


lookup_table = {
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

    'entertainment': (8, 1),

    'kiss': (4, 0.5),
    'pet': (4, 0.5),

    'accessories': (10, 0.5),
    'settings': (10, 0.5),
    'speech': (6, 1),

    'detail': (10, 1.5),
    'getting hair-makeup': (10, 1.5),
    'food': (10, 1.5),
    'other': (16, 1.5),
    'invite': (12, 1.5)
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


def genreate_look_up(group2images):
    for group_name, images in group2images.items():
        parts = group_name.split('_')
        group_id = parts[1]
        if group_id in lookup_table:
            # mean = calculate_flexible_mean(images,lookup_table[group_id][0] )
            # lookup_table[group_name] = (mean,lookup_table[group_id][1])
            continue
        else:
            lookup_table[group_name] = (5, 0.2)
    return lookup_table


def get_lookup_table():
    return lookup_table
