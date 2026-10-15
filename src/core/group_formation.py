"""Forming the album's groups from the photo table.

Where a group comes from. Weddings group by `(time_cluster, cluster_context)`,
everything else by `people_cluster`, and `get_images_per_groups` counts what
landed in each.

This sits in `core` rather than in `src/groups_operations` because both that
package and its caller need it: `album_processing` forms the groups to size
the lookup table before any splitting or merging happens, and for a
non-wedding album it is the only grouping there is. `groups_operations` is a
tree entered at `process_wedding_illegal_groups`; a module reached from
outside it does not belong inside it.
"""

import pandas as pd


def get_images_per_groups(original_groups):
    group2images_data_list = dict()

    for name_group, group_df in original_groups:
        num_images = len(group_df)
        group2images_data_list[name_group] = num_images

    return group2images_data_list


def get_missing_columns(required_columns: set, df: pd.DataFrame, logger) -> set:
    # Check if required column exists
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        logger.error(f"Missing required columns: {missing}")
        return missing
    return set()


def get_wedding_groups(df, manual_selection, logger):
    # Group by content only. Special ('None'/'other') groups are intentionally
    # NOT tagged here: the unique 'class|idx' tagging now happens in _get_groups
    # (inside process_wedding_illegal_groups), i.e. AFTER the first LUT budget
    # pass. That pass therefore sees raw un-fragmented groups (one 'None'/'other'
    # group per time_cluster) instead of one spread per special sub-group, which
    # used to inflate the initial spread count. manual_selection is unused now but
    # kept for the call sites' signature.
    if get_missing_columns({'time_cluster', 'cluster_context', 'cluster_label'}, df, logger):
        return None

    return df.groupby(['time_cluster', 'cluster_context'])


def get_none_wedding_groups(df, logger):
    if get_missing_columns({'people_cluster'}, df, logger):
        return None

    return df.groupby(['people_cluster'])
