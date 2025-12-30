import re
import random
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
try:
    from pyclustering.cluster.kmedoids import kmedoids as PycKMedoids
    _HAS_PYCLUST = True
except Exception:
    _HAS_PYCLUST = False

from experiments.plotting import plot_groups_to_pdf

def build_time_merged_groups(
    df: pd.DataFrame,
    time_col: str = "image_time_date",
    cluster_col: str = "sub_group_time_cluster",
    max_gap_minutes: int = 5,
    logger = None
) -> dict:
    """
    Returns a dict {new_group_key: [row_idx,...]} where:
      - Original groups of size==1 are ignored initially.
      - Remaining groups are ordered by time and merged if the gap between
        last(current) and first(next) <= max_gap_minutes.
      - If total capacity from merged non-singletons < total need (handled later),
        you can append singleton groups back as spill capacity.
    """
    try:
        # Ensure sorted by time
        df_sorted = df.sort_values(time_col)
        # Build original groups
        orig_groups = {g: gdf.index.to_list() for g, gdf in df_sorted.groupby(cluster_col)}

        # Compute group first/last times
        g_first = {}
        g_last = {}
        for g, idxs in orig_groups.items():
            t = df.loc[idxs, time_col].sort_values()
            g_first[g] = t.iloc[0]
            g_last[g]  = t.iloc[-1]

        # Split non-singletons / singletons
        non_single = [(g, idxs) for g, idxs in orig_groups.items() if len(idxs) >= 2]
        singles    = [(g, idxs) for g, idxs in orig_groups.items() if len(idxs) == 1]

        # Nothing to merge => return as-is
        if not non_single:
            # only singletons exist; return them (allocator may use them if needed)
            return {f"S{i}": idxs for i, (_, idxs) in enumerate(singles)}

        # Sort non-singletons by their first time
        non_single.sort(key=lambda x: g_first[x[0]])

        # Merge consecutive close groups
        dt = pd.Timedelta(minutes=max_gap_minutes)
        merged = []
        cur_key, cur_idxs = non_single[0]
        cur_last = g_last[cur_key]

        for j in range(1, len(non_single)):
            nxt_key, nxt_idxs = non_single[j]
            gap = g_first[nxt_key] - cur_last
            if gap <= dt:
                # merge
                cur_idxs = cur_idxs + nxt_idxs
                cur_last = max(cur_last, g_last[nxt_key])
            else:
                merged.append(cur_idxs)
                cur_key, cur_idxs = nxt_key, nxt_idxs
                cur_last = g_last[cur_key]
        merged.append(cur_idxs)

        # Package as dict
        merged_groups = {f"M{i}": idxs for i, idxs in enumerate(merged)}

        # Keep singles separate; caller can decide to append them only if needed.
        # We'll return them alongside for convenience.
        merged_groups["_SINGLES_"] = [idxs for _, idxs in singles]  # list of [idxs]

    except Exception as e:
        logger.error(f"Error in build_time_merged_groups: {e}")
        raise

    return merged_groups

def allocate_prefer_larger_artificial(
    groups: Dict[str, List[int]],
    need: int,
    small_threshold: int = 5,   # kept only for signature compatibility; unused
    ensure_one: bool = True,    # kept only for signature compatibility; unused
    base_per_group: int = 1,
    logger:object =None# kept only for signature compatibility; unused
) -> Dict[str, int]:
    """
    Allocation strategy:

    1) Exclude '_SINGLES_' and empty groups from allocation.
    2) Base pass: give 1 per group (largest-first if need < #groups).
    3) If more needed, add +1 per step to the largest groups (by size), respecting group capacity.
    4) Special small-few rule:
         - If (#groups <= 3) AND (every group size <= 4) AND (need > 4):
             -> allocate exactly 1 per group and STOP (sum can be < need).
       If need <= 4, ignore the special rule and fulfill normally.

    Notes:
    - Capacity per group = its size (we don't cap to 1/2 sizes here).
    - Any group named '_SINGLES_' is always assigned 0.
    - The returned allocation may sum to < need in the special small-few case, by design.
    """
    try:
        # Initialize alloc for all keys
        alloc = {g: 0 for g in groups.keys()}

        # Filter valid groups (ignore _SINGLES_ and empties)
        valid = {g: idxs for g, idxs in groups.items()
                 if g != "_SINGLES_" and len(idxs) > 0}

        # Early exits
        if need <= 0 or not valid:
            if "_SINGLES_" in alloc:
                alloc["_SINGLES_"] = 0
            return alloc

        sizes = {g: len(idxs) for g, idxs in valid.items()}
        order = sorted(valid.keys(), key=lambda k: sizes[k], reverse=True)
        num_groups = len(order)
        max_size = max(sizes.values()) if sizes else 0

        # ---------- Special small-few scenario ----------
        if num_groups <= 3 and max_size <= 4 and need > 4:
            # Take 1 from each (up to capacity), then stop.
            for g in order:
                if sizes[g] >= 1:
                    alloc[g] = 1
            if "_SINGLES_" in alloc:
                alloc["_SINGLES_"] = 0
            return alloc

        # ---------- Normal path ----------
        remaining = need

        # Base: 1 per group
        if remaining >= num_groups:
            # give 1 to all groups with capacity
            for g in order:
                if sizes[g] >= 1:
                    alloc[g] = 1
            remaining -= sum(1 for g in order if sizes[g] >= 1)
        else:
            # need < #groups -> give 1 to the largest groups first
            for g in order:
                if remaining == 0:
                    break
                if sizes[g] >= 1:
                    alloc[g] = 1
                    remaining -= 1

        # Top-up: +1 to largest groups while capacity remains
        while remaining > 0:
            progressed = False
            for g in order:
                if remaining == 0:
                    break
                if alloc[g] < sizes[g]:
                    alloc[g] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                break  # all groups at capacity; cannot satisfy more

        if "_SINGLES_" in alloc:
            alloc["_SINGLES_"] = 0

    except Exception as e:
        logger.error(f"Error in : allocate_prefer_larger_artificial {e}")
        raise
    return alloc

def allocate_prefer_larger(
    groups: dict,
    need: int,
    small_threshold: int = 5,   # kept for compatibility; not used in this variant
    ensure_one: bool = True,    # kept for compatibility; not used in this variant
    base_per_group: int = 1,    # kept for compatibility; not used in this variant
):
    """
    Smart allocation:
      - Exclude '_SINGLES_' and groups with <2 items.
      - Base: give 2 per valid group (largest-first) until need is close.
      - Cap per group: size<=2 -> size; else -> max(2, size//2).
      - Round-robin +1 across groups (largest-first) without exceeding caps.
      - If caps prevent reaching `need`, return early (by design).
    """
    # 1) filter valid groups
    valid = {g: idxs for g, idxs in groups.items()
             if g != "_SINGLES_" and len(idxs) >= 2}
    keys_all = list(groups.keys())
    alloc = {g: 0 for g in keys_all}

    if need <= 0 or not valid:
        if "_SINGLES_" in groups:
            alloc["_SINGLES_"] = 0
        return alloc

    sizes = {g: len(idxs) for g, idxs in valid.items()}
    order = sorted(valid.keys(), key=lambda k: sizes[k], reverse=True)

    # 2) per-group caps: don't take more than ~half (except tiny groups can be fully used)
    caps = {}
    for g, sz in sizes.items():
        caps[g] = sz if sz <= 2 else max(2, sz // 2)

    # 3) base pass: try to give 2 per group (largest-first), respecting caps & need
    remaining = need
    base = 2
    base_demand = sum(min(base, caps[g]) for g in order)

    if remaining <= base_demand:
        # not enough to give base to all; give up to 2 (and cap) to largest groups first
        for g in order:
            if remaining == 0:
                break
            give = min(base, caps[g], remaining)
            alloc[g] += give
            remaining -= give
    else:
        # give base to all valid groups
        for g in order:
            give = min(base, caps[g])
            alloc[g] += give
            remaining -= give

        # 4) round-robin +1 until need met or caps reached
        while remaining > 0:
            progressed = False
            for g in order:
                if remaining == 0:
                    break
                if alloc[g] < caps[g]:
                    alloc[g] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                break  # all groups at cap; stop early (acceptable)

    # ensure singles get zero
    if "_SINGLES_" in groups:
        alloc["_SINGLES_"] = 0

    return alloc


def _sanitize_key(text: str) -> str:
    """Safe key: letters/digits/_ only (handy for filenames/alloc keys)."""
    if text is None:
        return "NA"
    text = str(text)
    text = text.strip()
    if text == "" or text.lower() == "nan":
        return "NA"
    # replace spaces and non-word with underscores
    text = re.sub(r"\W+", "_", text)
    return text

def _chunk_by_window(df, order_cols=None, window_size=10, label="W"):
    if order_cols is None:
        order_cols = [c for c in ["scene_order", "image_time_date"] if c in df.columns] or None
    df_sorted = df.sort_values(order_cols) if order_cols else df
    idxs = df_sorted.index.to_list()
    groups = {}
    for i in range(0, len(idxs), window_size):
        groups[f"{label}{i//window_size}"] = idxs[i:i+window_size]
    return groups

def _chunk_by_scene_cluster(
    df: pd.DataFrame,
    *,
    scene_col: str = "scene_order",
    cluster_col: str = "cluster_label",
    order_cols: Optional[List[str]] = None,
    window_size: int = 10,
    label_prefix: str = "WIN",
) -> Dict[str, List[int]]:
    """
    Split a single-scene dataframe into windows of size `window_size`,
    but first partition by `cluster_label` so each cluster forms its own
    contiguous windows.

    Returns:
        { f"{label_prefix}_{cluster}_{w}": [row_idx,...], ... }
    """
    if order_cols is None:
        order_cols = [c for c in (scene_col, "image_time_date") if c in df.columns]

    # Sort once for deterministic order within each cluster
    df_sorted = df.sort_values(order_cols) if order_cols else df

    # If no cluster column (or all missing), just chunk the whole scene
    if cluster_col not in df_sorted.columns or df_sorted[cluster_col].isna().all():
        return _chunk_by_window(df_sorted, order_cols=order_cols, window_size=window_size, label=label_prefix)

    groups: Dict[str, List[int]] = {}
    # Group by cluster within the scene
    for cl, gdf in df_sorted.groupby(cluster_col, dropna=False, sort=True):
        cl_key = _sanitize_key(cl)
        idxs = gdf.index.to_list()
        # Window within this cluster
        for w, start in enumerate(range(0, len(idxs), window_size)):
            win = idxs[start:start + window_size]
            groups[f"{label_prefix}_{cl_key}_{w}"] = win

    return groups


def select_remove_similar(
    is_artificial_time:bool,
    need: int,
    df: pd.DataFrame,
    cluster_name,
    logger,
    target_group_size: int = 10,  # kept for compatibility; no longer used for k-medoids
) -> list[str]:
    try:
        #print(f"remove similar images for {cluster_name}")
        small_threshold = 7
        is_not_scored = all(x == 1 for x in df['total_score'].tolist()) if 'total_score' in df.columns else True
        if not is_not_scored:
            score  = dict(zip(df['image_id'], df['total_score']))
            higher  = True
        else:
            # Fallback
            if 'image_order' not in df.columns:
                # If neither present, assign a neutral 0 to keep stable ordering
                score  = dict(zip(df['image_id'], [0] * len(df)))
                higher  = True
            else:
                score  = dict(zip(df['image_id'], df['image_order']))
                higher  = False  # lower image_order is better

        if is_artificial_time:
            scene_col = "scene_order"
            groups = {}

            if scene_col in df.columns:
                # how many scenes?
                n_scenes = df[scene_col].nunique(dropna=True)

                # group by scene and then by cluster label
                if scene_col not in df.columns:
                    raise KeyError(f"'{scene_col}' not in df")

                df2 = df.copy()
                df2["_orig_idx_"] = df2.index

                # Sorting for deterministic order inside groups
                sort_cols = [scene_col]
                if "image_time_date" and "image_time_date" in df2.columns:
                    sort_cols.append("image_time_date")
                sort_cols.append("_orig_idx_")
                df2 = df2.sort_values(sort_cols)

                use_cluster = ("cluster_label" in df2.columns) and (~df2["cluster_label"].isna()).any()

                groups: Dict[str, List[int]] = {}

                if use_cluster:
                    # scene -> cluster
                    for s_val, sgdf in df2.groupby(scene_col, sort=True, dropna=False):
                        for c_val, cgdf in sgdf.groupby("cluster_label", sort=True, dropna=False):
                            scene_key = _sanitize_key(s_val)
                            cl_key = _sanitize_key(c_val)
                            key = f"SCN_{scene_key}__CL_{cl_key}"
                            groups[key] = cgdf["_orig_idx_"].tolist()
                else:
                    # fallback: scene only
                    for s_val, sgdf in df2.groupby(scene_col, sort=True, dropna=False):
                        scene_key = _sanitize_key(s_val)
                        key = f"SCN_{scene_key}"
                        groups[key] = sgdf["_orig_idx_"].tolist()


        else:
            groups = build_time_merged_groups(df, logger=logger)


        alloc = allocate_prefer_larger_artificial(groups, need, small_threshold=small_threshold, ensure_one=True, logger=logger)

        # Selection Process
        final_selected = []

        id_to_unit = dict(zip(df["image_id"], df["embedding"]))

        selected_mat = None  # np.ndarray of shape (k, d)
        cos_thresh = 0.90

        def is_diverse(iid: str) -> bool:
            """Check cosine(candidate, ANY already selected) < cos_thresh."""
            nonlocal selected_mat
            if selected_mat is None or selected_mat.size == 0:
                return False
            cand = id_to_unit[iid]
            # dot with all selected (both unit), so equals cosine
            sims = selected_mat @ cand
            too_similar = (sims >= cos_thresh).any()
            return too_similar

        for g, idxs in groups.items():
            if len(idxs) == 0 or g == '_SINGLES_':
                continue

            m = alloc.get(g, 0)
            if m <= 0 or not idxs:
                continue

            gdf = df.loc[idxs]

            # your way: ids -> sort by score map
            ids = gdf['image_id'].tolist()
            ids_sorted = sorted(ids, key=lambda x: score[x], reverse=higher)

            picked = 0
            for iid in ids_sorted:
                if picked >= m:
                    break
                if not is_diverse(iid):
                    final_selected.append(iid)
                    picked += 1
                    # append to selected matrix
                    v = np.array(id_to_unit[iid]).reshape(1, -1)
                    selected_mat = v if selected_mat is None else np.vstack([selected_mat, v])

        # pdf_path = plot_groups_to_pdf(
        #     groups=groups,  # your {group_key: [row_idx,...]}
        #     alloc=alloc,  # your {group_key: k_to_select}
        #     df=df,  # must contain 'image_id'
        #     images_dir=r"C:\Users\user\Desktop\PicTime\AlbumDesigner\dataset/47981912",
        #     # folder with files named like <image_id>.jpg/png/...
        #     cluster_name=cluster_name,
        #     cluster_label='cluster_label',
        #     output_dir=r"C:\Users\user\Desktop\PicTime\AlbumDesigner\output\47981912",  # optional; defaults to images_dir
        #     cols=5, rows=6,
        #     selected_images=final_selected# 30 thumbs per page
        # )
        # print("PDF written to:", pdf_path)

    except Exception as e:
        logger.error(f"Error in : select_remove_similar {e}")
        raise

    return final_selected

