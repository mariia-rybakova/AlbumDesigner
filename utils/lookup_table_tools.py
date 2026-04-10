# Lookup table with category preferences (mean, std)
import math
from typing import Dict, Tuple, Optional
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
    'portrait': (2, 0.5),
    'very large group': (2, 0.5),
    'walking the aisle': (4, 0.75),
    'bride getting dressed': (9, 1),
    'first dance': (4, 0.5),
    'cake cutting': (6, 1),
    'ceremony': (5, 1),
    'couple': (6, 1),
    'dancing': (24, 1),
    'entertainment': (2, 0.5),
    'kiss': (4, 0.5),
    'pet': (4, 0.5),
    'accessories': (2, 0.5),
    'settings': (4, 0.5),
    'speech': (6, 1),
    'detail': (6, 1.5),
    'getting hair-makeup': (2, 1.5),
    'food': (4, 0.5),
    'other': (2, 0.5),
    'invite': (2, 0.5),
    'None':(2,0.5),
    'wedding dress': (2,0.5),
    'vehicle':(2,0.5),
    'inside vehicle':(2,0.5),
    'rings': (3, 0.5),
    'suit': (3, 0.5),
    'may kiss bride': (1, 0.9),
    'bride and groom with parents': (3, 0.9),
    'groom with his parents': (2, 0.9),
    'bride with her parents': (2, 0.9),
    'parents portrait': (3, 0.9)
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


class LookUpTable:
    default_table = None

    def __init__(self, table: Dict[str, Tuple[float, float]] = None):
        self._table = {} if not table else table

    @property
    def table(self):
        return self._table.copy()

    @staticmethod
    def _get_group_id(group_name):
        pass

    @staticmethod
    def _get_content_key(group_key):
        pass

    def get_table(self, group2images, logger=None, density=3):
        density_factors = CONFIGS['density_factors']

        try:
            lookup_table = self.default_table

            max_per_spread = 24

            for group_name, num_images in group2images.items():
                # Extract group ID
                group_id = self._get_group_id(group_name)

                # Assign default values if group_id is not in lookup_table
                if group_id not in lookup_table:
                    lookup_table[group_id] = (10, 4)

                lookup_table[group_id] = (
                    max(1, min(max_per_spread, lookup_table[group_id][0] * density_factors[density])),
                    max(0.25, min(3, lookup_table[group_id][1] * density_factors[density]))
                )

            # Updated the lookup table
            self._table = lookup_table

        except Exception as e:
            logger.error(f"Error: Unexpected error while updating lookup table: {str(e)}")

    def get_current_spread_parameters(self, group_key, number_of_images):
        # Extract the correct lookup key
        content_key = self._get_content_key(group_key)

        group_params = self._table.get(content_key, (10, 1.5))
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

    def update_with_layouts_size(self, layouts_df):
        layout_sizes = sorted(list(layouts_df['number of boxes'].unique()))

        for key, value in self._table.items():
            if value[0] >= 12:
                # Find the closest value in layout_sizes
                closest_size = min(layout_sizes, key=lambda x: abs(x - value[0]))
                self._table[key] = (min(closest_size,CONFIGS['max_imges_per_spread']), value[1])


    def _compute_initial_spreads(self, group2images: Dict) -> Dict:
        """Compute spread count per group based on current LUT values.

        For each group, reads spread parameters via get_current_spread_parameters
        (which may apply a max_group_spread cap) and computes ceil(number_images / param).
        Enforces at least 1 spread and at most 24 photos per spread.
        """
        spreads_per_group = {}

        for key, number_images in group2images.items():
            # Get spread parameters for the current group
            spread_params = self.get_current_spread_parameters(key, number_images)
            # Calculate required spreads for this group
            spreads = math.ceil(number_images / spread_params[0])
            # Enforce minimum spreads so photos per spread stays <= 24
            min_spreads = math.ceil(number_images / 24)
            spreads = max(spreads, min_spreads, 1)
            spreads_per_group[key] = spreads
        return spreads_per_group

    @staticmethod
    def _reduce_spreads(spreads_per_group: Dict, group2images: Dict, max_total_spreads: int) -> Dict:
        """Reduce spreads one at a time until total <= max_total_spreads.

        Each iteration picks the group with the most spreads that can still
        be reduced (without exceeding 24 photos per spread). Modifies
        spreads_per_group in place.
        """
        total_spreads = sum(spreads_per_group.values())
        while total_spreads > max_total_spreads:
            # Find the group with the most spreads that can still be reduced
            best_key = None
            best_spreads = 0
            for key, current_spreads in spreads_per_group.items():
                min_spreads = max(1, math.ceil(group2images[key] / 24))
                if current_spreads > min_spreads and current_spreads > best_spreads:
                    best_key = key
                    best_spreads = current_spreads

            if best_key is None:
                break  # Cannot reduce further without exceeding 24 photos per spread

            spreads_per_group[best_key] -= 1
            total_spreads -= 1

        return spreads_per_group

    @staticmethod
    def _expand_with_floor(spreads_per_group: Dict, group2images: Dict,
                           min_total_spreads: int, min_photos_per_spread: int) -> Tuple[Dict, int]:
        """Expand spreads one at a time until total >= min_total_spreads.

        Each iteration picks the group with the highest photos-per-spread ratio
        that can still grow without going below min_photos_per_spread per spread.
        Modifies spreads_per_group in place. Returns (spreads_per_group, total).
        """
        total_spreads = sum(spreads_per_group.values())
        while total_spreads < min_total_spreads:
            # Find the group with the highest photos-per-spread ratio that can still expand
            best_key = None
            best_ratio = 0
            for key, current_spreads in spreads_per_group.items():
                num_images_in_group = group2images[key]
                max_spreads = max(1, num_images_in_group // min_photos_per_spread)
                if current_spreads >= max_spreads:
                    continue
                ratio = num_images_in_group / current_spreads
                if ratio > best_ratio:
                    best_key = key
                    best_ratio = ratio

            if best_key is None:
                break

            spreads_per_group[best_key] += 1
            total_spreads += 1

        return spreads_per_group, total_spreads

    @staticmethod
    def _expand_spreads(spreads_per_group: Dict, group2images: Dict, min_total_spreads: int) -> Dict:
        """Expand spreads in two phases to reach min_total_spreads.

        Phase 1: expand keeping >= 2 photos per spread.
        Phase 2 (if still short): allow 1 photo per spread as last resort.
        Modifies spreads_per_group in place.
        """
        spreads_per_group, total_spreads = LookUpTable._expand_with_floor(
            spreads_per_group, group2images, min_total_spreads, min_photos_per_spread=2)

        if total_spreads < min_total_spreads:
            spreads_per_group, total_spreads = LookUpTable._expand_with_floor(
                spreads_per_group, group2images, min_total_spreads, min_photos_per_spread=1)
        return spreads_per_group

    def _apply_table_reduction(self, spreads_per_group: Dict, group2images: Dict) -> None:
        """Write back LUT after reduction: only increase values (to produce fewer spreads).

        For each group, computes ceil(n_images / target_spreads) and writes it
        to the LUT only if it exceeds the current value.
        """
        for key, target_spreads in spreads_per_group.items():
            content_key = self._get_content_key(key)
            current_value, extra_value = self._table.get(content_key, (10, 1.5))
            new_value = min(24, math.ceil(group2images[key] / target_spreads))
            if new_value > current_value:
                self._table[content_key] = (new_value, extra_value)

    def _apply_table_expansion(self, spreads_per_group: Dict,
                               group2images: Dict) -> None:
        """Write back LUT after expansion: only decrease values (to produce more spreads).

        For each group, computes int(n_images / target_spreads) and writes it
        to the LUT only if it is below the current value.
        """
        for key, target_spreads in spreads_per_group.items():
            content_key = self._get_content_key(key)
            current_value, extra_value = self._table.get(content_key, (10, 1.5))
            new_value = max(1, int(group2images[key] / target_spreads))
            if new_value < current_value:
                self._table[content_key] = (new_value, extra_value)

    def update_with_limit(self, group2images, max_total_spreads, min_total_spreads=None, logger=None):
        """Adjust LUT values so total spreads across all groups stays within [min, max].

        1. Computes initial spread count per group from current LUT.
        2. If total > max: reduces spreads (largest groups first), then increases LUT values.
        3. If total < min: expands spreads (most crowded groups first), then decreases LUT values.
        4. If within range: no changes to LUT.
        """
        spreads_per_group = self._compute_initial_spreads(group2images)
        total_spreads = sum(spreads_per_group.values())

        if total_spreads > max_total_spreads:
            spreads_per_group = self._reduce_spreads(spreads_per_group, group2images, max_total_spreads)
            self._apply_table_reduction(spreads_per_group, group2images)
        elif min_total_spreads is not None and total_spreads < min_total_spreads:
            spreads_per_group = self._expand_spreads(spreads_per_group, group2images, min_total_spreads)
            self._apply_table_expansion(spreads_per_group, group2images)
            if logger:
                logger.debug(f'Expanded LUT: {total_spreads} -> {sum(spreads_per_group.values())} (target {min_total_spreads})')
        # else:
        #     self._apply_table_reduction(spreads_per_group, group2images)


class WeddingLookUpTable(LookUpTable):
    default_table = wedding_lookup_table

    @staticmethod
    def _get_group_id(group_name):
        return group_name[1].split('_')[0]

    @staticmethod
    def _get_content_key(group_key):
        if "_" in group_key[1]:
            return group_key[1].split("_")[0]
        else:
            return group_key[1]


class NonWeddingLookUpTable(LookUpTable):
    default_table = non_wedding_lookup_table

    @staticmethod
    def _get_group_id(group_name):
        return group_name[0].split('_')[0]

    @staticmethod
    def _get_content_key(group_key):
        return group_key[0].split("_")[0]