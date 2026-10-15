"""Forming the album's groups from the photo table.

The first step of `src/groups_operations`: a photo frame becomes groups, which
`groups_management` then sizes and `split`/`merge` reshape. Weddings group by
`(time_cluster, cluster_context)`, everything else by `people_cluster`.

Special groups -- the ones whose content class says the classifier had nothing
to say -- are separated here by `split_groups` and given a unique `'class|idx'`
context by `assign_special_group_contexts`, which is what keeps two same-class
groups from colliding once a later split reindexes them.
"""

import pandas as pd

from utils.configs import SPECIAL_CONTENT_CLASSES, SPECIAL_GROUP_SEP


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


def split_groups(df):
    # Split DataFrame based on cluster_context being a special class (as strings)
    mask_special = df['cluster_context'].isin(SPECIAL_CONTENT_CLASSES)
    df_special = df[mask_special].copy()
    df_regular = df[~mask_special].copy()

    # Group df_special and update cluster_context for each group
    groups_special = df_special.groupby(['time_cluster', 'cluster_context', 'cluster_label'])
    return df_special, df_regular, groups_special


def assign_special_group_contexts(df_special, groups_special):
    """Give each special ('None'/'other') group a unique 'class|idx' cluster_context.

    Same-class special groups otherwise share a (time_cluster, cluster_context)
    key and are told apart only by group_sub_index, which a later split reindexes
    from 0 and can collide. A unique context tag removes that overlap; the content
    class is recovered as context.split(SPECIAL_GROUP_SEP)[0]. idx is a global
    enumerate over groups_special, so 'class|idx' is unique on its own (no size
    suffix needed). Modifies df_special in place and returns it.
    """
    for idx, (key, group_df) in enumerate(groups_special):
        df_special.loc[group_df.index, 'cluster_context'] = f"{key[1]}{SPECIAL_GROUP_SEP}{idx}"
    return df_special


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
