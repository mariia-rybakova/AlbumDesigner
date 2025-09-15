import numpy as np
import pandas as pd
from collections import defaultdict
from testing_code.plotting import plot_selected_to_pdf


import random
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
try:
    from pyclustering.cluster.kmedoids import kmedoids as PycKMedoids
    _HAS_PYCLUST = True
except Exception:
    _HAS_PYCLUST = False



def _normalize_rows(E: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(E, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return E / n

def _score_lookup_from_df(df: pd.DataFrame):
    # If not scored (all ones), fall back to image_order (lower is better? You used ascending unless reversed)
    is_not_scored = all(x == 1 for x in df['total_score'].tolist())
    if is_not_scored:
        lookup = dict(zip(df['image_id'], df['image_order']))
        higher_is_better = False  # smaller order is better
    else:
        lookup = dict(zip(df['image_id'], df['total_score']))
        higher_is_better = True
    return lookup, higher_is_better

def _sort_ids_by_score(ids, score_lookup, higher_is_better):
    return sorted(ids, key=lambda x: score_lookup[x], reverse=higher_is_better)

def _fair_time_allocation(df: pd.DataFrame, need: int) -> dict:
    """
    Allocate counts per time bin (sub_group_time_cluster) with:
      - at least 1 per bin when possible,
      - proportional remainder,
      - largest remainder rounding,
      - capacity caps.
    Returns: {time_bin: m}
    """
    groups = {tb: g.index.to_list() for tb, g in df.groupby('sub_group_time_cluster')}
    bins = list(sorted(groups.keys()))
    sizes = {tb: len(groups[tb]) for tb in bins}
    T = len(bins)
    total = sum(sizes.values())

    alloc = {tb: 0 for tb in bins}
    if need <= 0 or total == 0:
        return alloc

    # Case 1: need < T → pick evenly spaced bins, 1 each.
    if need < T:
        # space bins across timeline (bins are sorted)
        step = T / need
        picks = set()
        acc = 0.0
        for _ in range(need):
            idx = int(acc)
            picks.add(bins[idx])
            acc += step
        for tb in picks:
            alloc[tb] = 1
        return alloc

    # Case 2: need ≥ T → give 1 to each bin for coverage
    for tb in bins:
        alloc[tb] = 1
    remaining = need - T
    if remaining <= 0:
        return alloc

    # Distribute remainder proportionally by (capacity = size-1)
    caps = {tb: max(0, sizes[tb] - alloc[tb]) for tb in bins}
    cap_total = sum(caps.values())
    if cap_total == 0:
        return alloc  # all bins exhausted

    # Proportional quotas
    raw = {tb: remaining * (caps[tb] / cap_total) for tb in bins}
    floors = {tb: int(np.floor(raw[tb])) for tb in bins}
    used = sum(floors.values())
    remainders = {tb: raw[tb] - floors[tb] for tb in bins}

    # Start with floors, cap by capacity
    for tb in bins:
        alloc[tb] += min(floors[tb], caps[tb])

    leftover = remaining - sum(min(floors[tb], caps[tb]) for tb in bins)
    if leftover <= 0:
        return alloc

    # Largest remainders with capacity caps
    for tb in sorted(bins, key=lambda b: remainders[b], reverse=True):
        if leftover <= 0:
            break
        if alloc[tb] < sizes[tb]:
            alloc[tb] += 1
            leftover -= 1

    return alloc

def _maxmin_select(indices, E_norm, k, already_idx, id_at, score_lookup, higher_is_better):
    """
    Greedy max-min on normalized embeddings.
    - indices: candidate dataframe indices (integers into df/E_norm)
    - E_norm: (n, d) normalized embeddings
    - k: number to pick
    - already_idx: set of df indices already selected
    - id_at: function idx->image_id
    """
    selected = []

    # Seed: best score among candidates (stable & fast)
    seed = max(indices, key=lambda i: score_lookup[id_at(i)]) if higher_is_better \
            else min(indices, key=lambda i: score_lookup[id_at(i)])
    if seed not in already_idx:
        selected.append(seed)
        already_idx.add(seed)

    # Greedy max-min
    while len(selected) < k and len(already_idx) < len(indices) + len(already_idx):
        best_i, best_min_d = None, -1.0
        # Prestack selected for fast dot
        S = np.stack([E_norm[j] for j in already_idx]) if already_idx else None
        for i in indices:
            if i in already_idx:
                continue
            # distance to selected set = 1 - max cosine similarity
            if S is None:
                min_d = 1.0
            else:
                # cos sims with selected
                sims = S @ E_norm[i]
                min_d = 1.0 - float(np.max(sims))
            # tie-break by score
            key = (min_d, score_lookup[id_at(i)])
            if best_i is None:
                best_i, best_min_d = i, min_d
                best_key = key
            else:
                # for tie-break correctness when lower-is-better scores, adjust second key
                second = score_lookup[id_at(i)] if higher_is_better else -score_lookup[id_at(i)]
                best_second = best_key[1] if higher_is_better else -best_key[1]
                if (min_d > best_min_d) or (min_d == best_min_d and second > best_second):
                    best_i, best_min_d = i, min_d
                    best_key = key
        if best_i is None:
            break
        already_idx.add(best_i)
        selected.append(best_i)

    return selected

def filter_similarity_diverse_old(
    need: int,
    df: pd.DataFrame,
    cluster_name,
    logger,
    target_group_size: int = 10,  # kept for compatibility; no longer used for k-medoids
) -> list[str]:
    """
    Fast, fair, timeline-aware selection:
      1) early exits,
      2) fair allocation across time bins,
      3) greedy max-min diversity on L2-normalized embeddings,
      4) tie-break by score.
    """
    try:
        n = len(df)
        if n == 0 or need <= 0:
            return []

        # Canonicalize required columns
        assert 'image_id' in df.columns
        assert 'embedding' in df.columns
        assert 'sub_group_time_cluster' in df.columns
        # Orientation is optional; only used for minor tie-breaks, ignore if missing

        # Score lookup
        score_lookup, higher_is_better = _score_lookup_from_df(df)

        # Early exit 1: overall
        if n <= need:
            # Return all, sorted by score (no heavy work)
            ids = df['image_id'].tolist()
            ids = _sort_ids_by_score(ids, score_lookup, higher_is_better)
            logger.info(f"[{cluster_name}] Early exit: n={n} <= need={need}, returning all.")
            return ids

        # Build normalized embedding matrix in the existing row order
        E = np.vstack(df['embedding'].values).astype('float32')
        E = _normalize_rows(E)

        # Time-aware fair allocation
        time_alloc = _fair_time_allocation(df, need)  # {time_bin: m}
        # Sanity: ensure total equals need (may happen if caps hit)
        total_alloc = sum(time_alloc.values())
        if total_alloc < need:
            # distribute the deficit by score across all bins with capacity left
            capacity = {tb: sum(1 for _ in df.index[df['sub_group_time_cluster'] == tb]) - time_alloc[tb]
                        for tb in time_alloc.keys()}
            deficit = need - total_alloc
            # order time bins by (remaining capacity, then size)
            order = sorted(capacity.keys(), key=lambda tb: (capacity[tb],
                                                            len(df.index[df['sub_group_time_cluster'] == tb])),
                           reverse=True)
            for tb in order:
                if deficit == 0: break
                if capacity[tb] > 0:
                    time_alloc[tb] += 1
                    deficit -= 1

        # Selection (global set of selected row indices)
        selected_idx = set()
        id_at = lambda i: df.iloc[i]['image_id']

        # For each time bin, pick its quota using greedy max-min *against the global selected set*
        for tb, m in sorted(time_alloc.items(), key=lambda kv: kv[0]):  # process in time order
            if m <= 0:
                continue
            bin_indices = df.index[df['sub_group_time_cluster'] == tb].tolist()
            if m >= len(bin_indices):
                # Early exit 2: quota >= bin size → just take all by score
                ids = [df.iloc[i]['image_id'] for i in bin_indices]
                ids_sorted = _sort_ids_by_score(ids, score_lookup, higher_is_better)
                take = ids_sorted[:m]
                selected_idx.update(df.index[df['image_id'].isin(take)])
                continue

            # Prefer higher-scored candidates first inside the bin to seed better
            # (the _maxmin_select will still maximize diversity against global selected set)
            bin_indices_sorted = sorted(
                bin_indices,
                key=lambda i: score_lookup[id_at(i)],
                reverse=higher_is_better
            )
            # Run greedy max-min on this bin
            picked = _maxmin_select(
                indices=bin_indices_sorted,
                E_norm=E,
                k=m,
                already_idx=selected_idx,
                id_at=id_at,
                score_lookup=score_lookup,
                higher_is_better=higher_is_better
            )
            selected_idx.update(picked)

        # If rounding left us short/overfull, adjust:
        if len(selected_idx) > need:
            # Trim by lowest score among those with smallest marginal diversity (approx: lowest score)
            sel_ids = [id_at(i) for i in selected_idx]
            trimmed = _sort_ids_by_score(sel_ids, score_lookup, higher_is_better)[:need]
            selected_idx = set(df.index[df['image_id'].isin(trimmed)])
        elif len(selected_idx) < need:
            # Fill remaining globally via max-min over all remaining
            remaining_k = need - len(selected_idx)
            all_indices_sorted = sorted(
                [i for i in range(n) if i not in selected_idx],
                key=lambda i: score_lookup[id_at(i)],
                reverse=higher_is_better
            )
            picked = _maxmin_select(
                indices=all_indices_sorted,
                E_norm=E,
                k=remaining_k,
                already_idx=selected_idx,
                id_at=id_at,
                score_lookup=score_lookup,
                higher_is_better=higher_is_better
            )
            selected_idx.update(picked)

        selected_ids = [id_at(i) for i in selected_idx]

        # Final deterministic ordering: by time bin (timeline), then by score
        tb_of = dict(zip(df['image_id'], df['sub_group_time_cluster']))
        selected_ids = sorted(
            selected_ids,
            key=lambda x: (tb_of[x], -score_lookup[x] if higher_is_better else score_lookup[x])
        )

        plot_selected_to_pdf(df, selected_ids, image_dir=r'C:\Users\karmel\Desktop\AlbumDesigner\dataset\newest_wedding_galleries/46229128', output_pdf=fr"C:\Users\karmel\Desktop\AlbumDesigner\output\46229128\{cluster_name}.pdf")

        return selected_ids

    except Exception as e:
        logger.error(f"Error filter_similarity_diverse cluster name {cluster_name}: {e}")
        return []



def allocate_simple_by_group_size(groups: dict, need: int, small_threshold: int = 5) -> dict:
    """
    groups: {group_key: [row_idx, ...]}
    need: total items to select
    small_threshold: groups with size <= this stay at 1 during the first pass

    Returns: {group_key: m_to_select}
    """
    group_sizes = {g: len(idx_list) for g, idx_list in groups.items()}
    keys = list(group_sizes.keys())
    G = len(keys)

    # Edge cases
    if need <= 0 or G == 0:
        return {g: 0 for g in keys}

    # Start with 1 each (coverage), capped by capacity
    alloc = {g: min(1, group_sizes[g]) for g in keys}
    already = sum(alloc.values())

    # If fewer slots than groups: keep the biggest `need` groups at 1, others 0
    if need < G:
        # zero-out all, then set 1 for top-need by size
        alloc = {g: 0 for g in keys}
        for g in sorted(keys, key=lambda k: group_sizes[k], reverse=True)[:need]:
            alloc[g] = 1
        return alloc

    remaining = need - already
    if remaining <= 0:
        return alloc

    # ---- Pass 1: add extras to large groups only, but DO NOT take a group fully yet ----
    # Soft cap = size-1 (leave at least one unselected), and skip small groups
    large_groups = [g for g in sorted(keys, key=lambda k: group_sizes[k], reverse=True)
                    if group_sizes[g] > small_threshold and group_sizes[g] > 1]

    # Distribute one-by-one in round-robins over large groups until remaining is 0
    # or all large groups hit soft caps
    progressed = True
    while remaining > 0 and progressed:
        progressed = False
        for g in large_groups:
            soft_cap = max(0, group_sizes[g] - 1)   # don't take the full group in pass 1
            if alloc[g] < soft_cap:
                alloc[g] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break

    if remaining <= 0:
        return alloc

    # ---- Pass 2: if still short, allow filling to full capacity (largest groups first) ----
    for g in large_groups:
        cap = group_sizes[g]
        if alloc[g] < cap and remaining > 0:
            take = min(cap - alloc[g], remaining)
            alloc[g] += take
            remaining -= take
            if remaining == 0:
                break

    if remaining <= 0:
        return alloc

    # ---- Pass 3: finally, if still short, try medium/small groups (while respecting capacity) ----
    others = [g for g in sorted(keys, key=lambda k: group_sizes[k], reverse=True)
              if g not in large_groups and group_sizes[g] > 1]
    for g in others:
        cap = group_sizes[g]
        if alloc[g] < cap and remaining > 0:
            # still try not to fully consume tiny groups if we have alternatives
            soft_cap = max(1, min(cap - 1, cap)) if len(others) > 1 else cap
            target = min(soft_cap, cap)
            if alloc[g] < target:
                take = min(target - alloc[g], remaining)
                alloc[g] += take
                remaining -= take
                if remaining == 0:
                    break

    # If there's still remaining (rare), fill any capacity left anywhere
    if remaining > 0:
        for g in sorted(keys, key=lambda k: group_sizes[k], reverse=True):
            cap = group_sizes[g]
            if alloc[g] < cap and remaining > 0:
                take = min(cap - alloc[g], remaining)
                alloc[g] += take
                remaining -= take
                if remaining == 0:
                    break

    return alloc


def _allocate_within_group_by_time(gdf: pd.DataFrame, m: int,need:int) -> dict:
    """
    Allocate m picks across time bins inside one subquery group.
    Returns: {time_bin: count}
    """

    vals = pd.to_numeric(gdf['image_time_date'], errors='coerce')
    unique_count = len(np.unique(vals))
    q = max(1, min(int(m), unique_count))
    time_bins = pd.qcut(vals, q=q, labels=False, duplicates='drop')
    sizes = time_bins.value_counts().sort_index().to_dict()  # {bin: size}
    # Reuse the largest remainder allocator with ensure-one coverage
    keys = list(sizes.keys())
    G = len(keys)
    alloc = {k: 0 for k in keys}
    total = sum(sizes.values())
    if need <= 0 or total == 0:
        return alloc

    # If we have fewer slots than groups, give 1 to the biggest groups first
    if need < G:
        # Choose the top-need groups by size
        for k in sorted(keys, key=lambda k: sizes[k], reverse=True)[:need]:
            alloc[k] = 1
        return alloc

    # Otherwise, start with coverage: 1 per group (when possible)

    for k in keys:
        alloc[k] = min(1, sizes[k])  # cap by capacity
    remaining = need - sum(alloc.values())


    if remaining <= 0:
        return alloc

    # Capacity left per group
    caps = {k: max(0, sizes[k] - alloc[k]) for k in keys}
    cap_total = sum(caps.values())
    if cap_total == 0:
        return alloc

    # Proportional remainder (largest remainder method)
    raw = {k: remaining * (caps[k] / cap_total) for k in keys}
    floors = {k: int(np.floor(raw[k])) for k in keys}
    # Assign floors (respect capacity)
    used = 0
    for k in keys:
        take = min(floors[k], caps[k])
        alloc[k] += take
        used += take

    leftover = remaining - used
    if leftover <= 0:
        return alloc

    remainders = {k: raw[k] - floors[k] for k in keys}
    for k in sorted(keys, key=lambda kk: remainders[kk], reverse=True):
        if leftover == 0:
            break
        if alloc[k] < sizes[k]:
            alloc[k] += 1
            leftover -= 1
    return alloc


def _time_bins_for_group(gdf: pd.DataFrame, m: int):
    """
    Determine time bins within a group.
    Priority:
      1) Use 'sub_group_time_cluster' if present (already integer bins).
      2) Else use 'image_time_date' (numeric timestamp or parseable) and create q bins.
      3) Else return a single bin (no time info).
    Returns: pd.Series of bin labels (aligned to gdf)
    """
    if 'sub_group_time_cluster' in gdf.columns:
        return gdf['sub_group_time_cluster'].astype('int64')

    if 'image_time_date' in gdf.columns:
        # Try to ensure numeric timestamps
        s = gdf['image_time_date']
        if not np.issubdtype(s.dtype, np.number):
            # try parse to datetime then to int
            s = pd.to_datetime(s, errors='coerce')
            # If still NaT, fill with group median later
        # Build q bins = min(unique_times, m) (at least 1)
        # If still datetime, convert to view int (ns)
        if np.issubdtype(s.dtype, 'datetime64[ns]'):
            vals = s.view('int64')
        else:
            vals = pd.to_numeric(s, errors='coerce')
        # Handle NaNs by filling with median
        med = np.nanmedian(vals) if np.isnan(vals).any() else None
        if np.isnan(vals).any():
            vals = np.where(np.isnan(vals), med, vals)
        unique_count = len(np.unique(vals))
        q = max(1, min(int(m), unique_count))
        if q <= 1:
            return pd.Series(0, index=gdf.index)
        try:
            bins = pd.qcut(vals, q=q, labels=False, duplicates='drop')
            return pd.Series(bins, index=gdf.index)
        except ValueError:
            # qcut can fail if not enough unique edges
            return pd.Series(0, index=gdf.index)

    # No time columns available
    return pd.Series(0, index=gdf.index)



def allocate_prefer_larger(groups, need, small_threshold=5, ensure_one=True):
    """
    groups: {group_key: [row_idx, ...]}
    Rule:
      - Start with 1 per group (coverage) if ensure_one and capacity allows.
      - Distribute the remainder proportionally to group size (largest remainder),
        but in the first pass **only among groups with size > small_threshold**.
      - Respect capacity (can’t exceed group size).
      - If still short, allow filling remaining capacity anywhere (largest-first).
    Returns: {group_key: m}
    """
    sizes = {g: len(idx) for g, idx in groups.items()}
    keys = list(sizes.keys())
    G = len(keys)
    if need <= 0 or G == 0:
        return {g: 0 for g in keys}

    # If need < #groups: give 1 to the largest `need` groups
    if need < G:
        alloc = {g: 0 for g in keys}
        for g in sorted(keys, key=lambda k: sizes[k], reverse=True)[:need]:
            alloc[g] = 1
        return alloc

    # Coverage
    alloc = {g: min(1, sizes[g]) for g in keys} if ensure_one else {g: 0 for g in keys}
    remaining = need - sum(alloc.values())
    if remaining <= 0:
        return alloc

    # Eligible big groups for proportional pass
    big = [g for g in keys if sizes[g] > small_threshold and sizes[g] > alloc[g]]
    cap_big = {g: sizes[g] - alloc[g] for g in big}
    total_cap_big = sum(cap_big.values())

    if total_cap_big > 0:
        # Proportional by SIZE (not capacity ratio) gives more to very large groups
        raw = {g: remaining * (sizes[g] / sum(sizes[h] for h in big)) for g in big}
        floors = {g: int(np.floor(raw[g])) for g in big}
        used = 0
        for g in big:
            take = min(floors[g], cap_big[g])
            alloc[g] += take
            used += take
        leftover = remaining - used

        if leftover > 0:
            rema = {g: (raw[g] - floors[g]) for g in big}
            for g in sorted(big, key=lambda k: rema[k], reverse=True):
                if leftover == 0:
                    break
                if alloc[g] < sizes[g]:
                    alloc[g] += 1
                    leftover -= 1
        remaining = need - sum(alloc.values())

    if remaining <= 0:
        return alloc

    # Final fill anywhere by remaining capacity, largest-first
    for g in sorted(keys, key=lambda k: sizes[k], reverse=True):
        if remaining == 0:
            break
        cap = sizes[g] - alloc[g]
        if cap <= 0:
            continue
        take = min(cap, remaining)
        alloc[g] += take
        remaining -= take

    return alloc


def prune_near_duplicates(ids, embeddings, scores, cos_thresh=0.97):
    """
    Keep diverse items by skipping any whose cosine similarity to an already-kept
    item is >= cos_thresh. Greedy order by 'scores' (desc).
    - ids: list[str]
    - embeddings: np.ndarray shape (n, d)
    - scores: dict image_id -> score (higher is better). If lower-is-better, pass -score.
    """
    if len(ids) <= 1:
        return ids[:]

    # L2-normalize rows
    E = embeddings.astype(np.float32)
    N = np.linalg.norm(E, axis=1, keepdims=True)
    N[N == 0.0] = 1.0
    E = E / N

    # sort ids by score desc
    order = sorted(ids, key=lambda x: scores[x], reverse=True)
    kept = []
    kept_vecs = None

    for iid in order:
        i = ids.index(iid)
        v = E[i]
        if kept_vecs is None or kept_vecs.size == 0:
            kept.append(iid)
            kept_vecs = v.reshape(1, -1)
            continue

        # cosine with all kept
        sims = kept_vecs @ v
        if float(np.max(sims)) >= cos_thresh:
            continue  # too similar, skip
        kept.append(iid)
        kept_vecs = np.vstack([kept_vecs, v])

    return kept

def filter_similarity_diverse(
    need: int,
    df: pd.DataFrame,
    cluster_name,
    logger,
    target_group_size: int = 10,  # kept for compatibility; no longer used for k-medoids
) -> list[str]:
    cos_thresh = 0.97
    small_threshold = 7

    if len(df) <= need:
        return  df['image_id'].tolist()

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

    groups = {g: gdf.index.to_list() for g, gdf in df.groupby('image_subquery_content')}
    pruned_ids_per_group = {}
    for g, idxs in groups.items():
        gdf = df.loc[idxs]
        ids = gdf['image_id'].tolist()
        E = np.vstack(gdf['embedding'].values).astype(np.float32)

        # For lower-is-better scores, pass negative so "higher is better" internally
        adj_scores = {iid: (score[iid] if higher else -score[iid]) for iid in ids}
        keep_ids = prune_near_duplicates(ids, E, adj_scores, cos_thresh=cos_thresh)
        pruned_ids_per_group[g] = keep_ids

    # 3) allocate (use pruned capacities)
    pruned_groups = {g: [df.index[df['image_id'] == iid][0] for iid in ids]
                     for g, ids in pruned_ids_per_group.items()}
    alloc = allocate_prefer_larger(pruned_groups, need, small_threshold=small_threshold, ensure_one=True)

    # 4) pick by score within each group
    selected = []
    for g, ids in pruned_ids_per_group.items():
        m = alloc.get(g, 0)
        if m <= 0 or not ids:
            continue
        # sort by score preference
        ids_sorted = sorted(ids, key=lambda x: score[x], reverse=higher)
        selected.extend(ids_sorted[:min(m, len(ids_sorted))])

    # Trim/fill to exactly `need`
    # Dedup preserve order
    seen, dedup = set(), []
    for x in selected:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    selected = dedup

    if len(selected) > need:
        selected.sort(key=lambda x: score[x], reverse=higher)
        selected = selected[:need]
    elif len(selected) < need:
        # Fill from remaining (global best by score, skipping already selected)
        remaining_ids = [iid for iid in sorted(df['image_id'].tolist(),
                                               key=lambda x: score[x], reverse=higher)
                         if iid not in seen]
        selected += remaining_ids[:(need - len(selected))]





    plot_selected_to_pdf(df, selected,
                         image_dir=r'C:\Users\karmel\Desktop\AlbumDesigner\dataset\newest_wedding_galleries/46229128',
                         output_pdf=fr"C:\Users\karmel\Desktop\AlbumDesigner\output\46229128\{cluster_name}.pdf")


    return selected



def filter_similarity_diverse_new(
    need: int,
    df: pd.DataFrame,
    cluster_name,
    logger,
    target_group_size: int = 10,  # kept for compatibility; no longer used for k-medoids
) -> list[str]:
    cos_thresh = 0.97
    small_threshold = 7

    if len(df) <= need:
        return  df['image_id'].tolist()

    is_not_scored = all(x == 1 for x in df['total_score'].tolist()) if 'total_score' in df.columns else True
    if not is_not_scored:
        score_map  = dict(zip(df['image_id'], df['total_score']))
        higher_is_better  = True
    else:
        # Fallback
        if 'image_order' not in df.columns:
            # If neither present, assign a neutral 0 to keep stable ordering
            score_map  = dict(zip(df['image_id'], [0] * len(df)))
            higher_is_better  = True
        else:
            score_map  = dict(zip(df['image_id'], df['image_order']))
            higher_is_better  = False  # lower image_order is better


    groups_idx = {g: gdf.index.to_list() for g, gdf in df.groupby('image_subquery_content')}

    # Allocation across groups
    groups_ids = {g: df.loc[idxs, 'image_id'].tolist() for g, idxs in groups_idx.items()}
    alloc = allocate_prefer_larger(groups_ids, need, small_threshold=small_threshold, ensure_one=True)

    # Selection bucket
    chosen: List[str] = []

    # Helper to sort ids by score (best first)
    def sort_by_score(ids: List[str]) -> List[str]:
        return sorted(ids, key=lambda iid: score_map[iid], reverse=higher_is_better)

    # Minimal cosine-distance to be considered "different enough"
    min_cos_dist = 1.0 - float(cos_thresh)  # e.g., 0.03 when thresh=0.97

    rng = random.Random(42)

    for gname, idxs in groups_idx.items():
        m = alloc.get(gname, 0)
        if m <= 0:
            continue

        gdf = df.loc[idxs].copy().reset_index(drop=True)
        if len(gdf) == 0:
            continue

        if len(gdf) <= 2:
            top_ids = sorted(
                gdf['image_id'].tolist(),
                key=lambda iid: score_map[iid],
                reverse=higher_is_better
            )
            chosen.extend(top_ids[:m])
            continue

        # Embeddings -> numpy (float32) and normalize for cosine distance
        E = []
        for v in gdf['embedding'].values:
            arr = np.array(v, dtype=np.float32).reshape(-1)
            E.append(arr)
        E = np.vstack(E)

        # Orientation bin
        ori_ser = gdf['image_orientation'].astype(str).str.lower() if 'image_orientation' in gdf.columns else pd.Series(
            ['unknown'] * len(gdf))
        orientation_bin = ori_ser.map({'portrait': 0, 'landscape': 1}).fillna(1).astype('int8').to_numpy().reshape(-1,
                                                                                                                   1)

        # Time cluster numeric
        if 'sub_group_time_cluster' in gdf.columns:
            time_cluster = pd.to_numeric(gdf['sub_group_time_cluster'], errors='coerce').fillna(-1).astype(
                int).to_numpy().reshape(-1, 1)
        else:
            time_cluster = np.full((len(gdf), 1), -1, dtype=int)

        # Combined features (for clustering only)
        X = np.hstack([E, orientation_bin, time_cluster])
        X = StandardScaler().fit_transform(X)

        # Decide clusters per GROUP
        approx_k = max(1, int(np.ceil(len(gdf) / max(1, target_group_size))))
        k = min(approx_k, len(gdf))

        # Prepare k-medoids (or fallback)
        if _HAS_PYCLUST and k > 0:
            init_medoids = rng.sample(range(len(gdf)), k)
            try:
                km = PycKMedoids(X.tolist(), init_medoids)
                km.process()
                clusters_idx_local = km.get_clusters()  # list of lists (local indices)
                medoids_local = set(km.get_medoids())
            except Exception:
                # Fallback to trivial 1 cluster if k-medoids fails
                clusters_idx_local = [list(range(len(gdf)))]
                medoids_local = {rng.randrange(len(gdf))}
        else:
            # Fallback to 1 cluster
            clusters_idx_local = [list(range(len(gdf)))]
            medoids_local = {rng.randrange(len(gdf))}

        # Map clusters -> image_ids (sorted by score best-first)
        clusters_ids: Dict[int, List[str]] = {}
        for cid, locs in enumerate(clusters_idx_local):
            ids = [gdf.loc[i, 'image_id'] for i in locs]
            clusters_ids[cid] = sort_by_score(ids)

        # Local lookup helpers
        id2loc = {gdf.loc[i, 'image_id']: i for i in range(len(gdf))}
        # Cosine distance matrix (since E is normalized)
        cosD = 1.0 - np.clip(E @ E.T, -1.0, 1.0)

        # Pick at most m per group
        selected_g: List[str] = []

        # 1) Seed: one from each cluster (its medoid if available, else best-by-score)
        # Order clusters by medoid score (best first)
        def medoid_score(cid: int) -> float:
            locs = clusters_idx_local[cid]
            # choose medoid if known, else best by score
            med = next((i for i in locs if i in medoids_local), locs[0])
            iid = gdf.loc[med, 'image_id']
            return score_map[iid]

        ordered_cids = sorted(range(len(clusters_idx_local)),
                              key=lambda cid: medoid_score(cid),
                              reverse=higher_is_better)

        for cid in ordered_cids:
            if len(selected_g) >= m:
                break
            locs = clusters_idx_local[cid]
            # prefer medoid as first pick
            med = next((i for i in locs if i in medoids_local), locs[0])
            iid = gdf.loc[med, 'image_id']
            if iid not in selected_g:
                selected_g.append(iid)

        # 2) Extra picks in the group: farthest-first, prefer unseen time clusters
        def time_of(iid: str) -> int:
            return int(time_cluster[id2loc[iid], 0])

        while len(selected_g) < m:
            # Candidates are remaining in this group
            group_all_ids = [gdf.loc[i, 'image_id'] for i in range(len(gdf))]
            cands = [iid for iid in sort_by_score(group_all_ids) if iid not in selected_g]
            if not cands:
                break

            used_times = {time_of(i) for i in selected_g}
            best_id, best_key = None, (-1.0, -np.inf)  # (min_dist, score_tiebreak)

            for iid in cands:
                i = id2loc[iid]
                if selected_g:
                    dmin = min(cosD[i, id2loc[j]] for j in selected_g)
                else:
                    dmin = 1.0  # if none selected, treat as max distance

                # hard filter if too similar and we still have alternatives
                # We'll rank by (dmin, score), with a small bonus if new time_cluster
                time_bonus = 0.05 if time_of(iid) not in used_times else 0.0
                rank_score = score_map[iid] if higher_is_better else (-score_map[iid])

                key = (dmin + time_bonus, rank_score)

                # keep the best key
                if key > best_key:
                    best_id, best_key = iid, key

            if best_id is None:
                break

            # Enforce similarity ceiling if possible
            if selected_g:
                i = id2loc[best_id]
                dmin = min(cosD[i, id2loc[j]] for j in selected_g)
                if dmin < min_cos_dist:
                    # Try to find a candidate that passes the ceiling
                    feasible = None
                    for iid in cands:
                        i2 = id2loc[iid]
                        dmin2 = min(cosD[i2, id2loc[j]] for j in selected_g)
                        if dmin2 >= min_cos_dist:
                            feasible = iid
                            break
                    if feasible is not None:
                        best_id = feasible

            selected_g.append(best_id)

        chosen.extend(selected_g)

    # If we didn't reach need, backfill across all remaining by score with diversity check
    if len(chosen) < need:
        remaining_ids = [iid for iid in df['image_id'].tolist() if iid not in chosen]
        remaining_ids = sorted(remaining_ids, key=lambda iid: score_map[iid], reverse=higher_is_better)

        # Global cosine distance for backfill
        # Prepare once
        id2row = {iid: i for i, iid in enumerate(df['image_id'].tolist())}
        # normalize global embeddings
        Eglobal = []
        for v in df['embedding'].values:
            Eglobal.append(np.array(v, dtype=np.float32).reshape(-1))
        Eglobal = np.vstack(Eglobal)
        Dglob = 1.0 - np.clip(Eglobal @ Eglobal.T, -1.0, 1.0)

        selected_set = set(chosen)
        while len(chosen) < need and remaining_ids:
            candidate = None
            best_key = (-1.0, -np.inf)
            for iid in remaining_ids:
                if iid in selected_set:
                    continue
                i = id2row[iid]
                if chosen:
                    dmin = min(Dglob[i, id2row[j]] for j in chosen)
                else:
                    dmin = 1.0
                rank_score = score_map[iid] if higher_is_better else (-score_map[iid])
                key = (dmin, rank_score)
                if key > best_key:
                    candidate, best_key = iid, key

            if candidate is None:
                break

            # similarity ceiling try
            if chosen:
                i = id2row[candidate]
                dmin = min(Dglob[i, id2row[j]] for j in chosen)
                if dmin < min_cos_dist:
                    # try next feasible
                    feas = None
                    for iid in remaining_ids:
                        if iid in selected_set:
                            continue
                        i2 = id2row[iid]
                        dmin2 = min(Dglob[i2, id2row[j]] for j in chosen)
                        if dmin2 >= min_cos_dist:
                            feas = iid
                            break
                    if feas is not None:
                        candidate = feas

            chosen.append(candidate)
            selected_set.add(candidate)
            remaining_ids = [iid for iid in remaining_ids if iid != candidate]


    plot_selected_to_pdf(df, chosen,
                         image_dir=r'C:\Users\karmel\Desktop\AlbumDesigner\dataset\newest_wedding_galleries/46229128',
                         output_pdf=fr"C:\Users\karmel\Desktop\AlbumDesigner\output\46229128\{cluster_name}.pdf")



    # Final trim (just in case)
    return chosen[:need]



