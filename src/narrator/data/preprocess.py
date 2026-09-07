"""Per-gallery precomputed caches (computed once at episode reset).

Modest gallery size (hundreds of photos) makes O(N^2) pairwise matrices cheap and
lets the reward/masking read them in O(1) per step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import RewardConfig
from .schema import PackedGallery


@dataclass
class GalleryCache:
    sim: np.ndarray            # (N, N) CLIP cosine similarity
    dt: np.ndarray             # (N, N) |time_norm_i - time_norm_j| in [0,1]
    cast_jaccard: np.ndarray   # (N, N) Jaccard of identity sets (0 if either empty)
    face_sim: np.ndarray       # (N, N) pooled face-embedding cosine (0 if either missing)
    style_conflict: np.ndarray  # (N, N) bool: candid vs formal mismatch (hard)
    bw_conflict: np.ndarray    # (N, N) bool: grayscale vs color mismatch (hard)
    is_dup: np.ndarray         # (N, N) bool: near-duplicate pair
    # Where each half of `cast` is DEFINED, i.e. where its 0.0 means "no cast in common" rather
    # than "no evidence". Both use the same convention -- the signal must exist in BOTH photos --
    # which is what `cast_mode: "defined"` needs and what the two halves used to disagree about
    # (face_sim already required both; cast_jaccard only required one). See RewardConfig.cast_mode.
    cast_defined: np.ndarray   # (N, N) bool: both photos have >= 1 identified person
    face_defined: np.ndarray   # (N, N) bool: both photos have a valid face embedding
    # Reference value for `photocov`: the mean nearest-selected similarity a GOOD selection of
    # `budget` photos achieves on this gallery, found greedily. 0.0 when the term is unweighted (it
    # is not computed then). Gallery-only and budget-fixed, so the policy cannot raise it.
    cov_ref: float = 0.0


def _greedy_cover(sim: np.ndarray, budget: int) -> float:
    """Mean nearest-selected similarity achieved by a greedy `budget`-photo selection.

    Facility location is submodular, so greedy is within (1 - 1/e) of optimal and is the standard
    solution. Used as `photocov`'s denominator: it depends only on the gallery and a FIXED budget, so
    unlike a policy-dependent normaliser it cannot be raised by building a smaller album.
    """
    s = np.clip(sim, 0.0, 1.0).astype(np.float64)
    n = s.shape[0]
    best = np.zeros(n)
    for _ in range(min(budget, n)):
        j = int(np.maximum(best[:, None], s).sum(0).argmax())
        new = np.maximum(best, s[:, j])
        if new.sum() <= best.sum() + 1e-12:
            break
        best = new
    return float(best.mean())


def timestamps_separate_clusters(pg: PackedGallery, window_s: float) -> bool:
    """Do this gallery's timestamps resolve distinct content clusters at ``window_s``?

    Judged ONLY on pairs of clusters that are time-consecutive -- whose ``[min, max]`` time
    intervals do not overlap. Two clusters that interleave in time (the photographer alternating
    between subjects) are necessarily ~0 s apart no matter how good the timestamps are, so they say
    nothing about timestamp quality. Measured over 250 real galleries, interleaving is the norm:
    the median gallery has 25.3% of its cluster pairs overlapping in time, p75 48.7%.

    Including those pairs makes the test measure interleaving instead of clock quality -- it fires
    on **16.4%** of galleries and correlates more with cluster overlap (+0.565) than with tied
    timestamps (+0.441), and the galleries it condemns but this version absolves average 83.3%
    cluster overlap vs 32.4% elsewhere. Restricted to time-consecutive pairs it fires on **1.2%**,
    which is the honest rate of "``dateTaken`` cannot resolve moments here".

    Uses the MEDIAN of those gaps, not the minimum: 58.4% of galleries contain some cross-cluster
    pair at a 0 s gap, so a minimum-based test would declare 95.2% of galleries broken. Returns
    True (trust time) when there is no time-consecutive pair to judge by -- 10.4% of galleries.
    """
    valid = pg.time_valid & (pg.cluster_id >= 0)
    if valid.sum() < 4:
        return True
    t = pg.time_norm[valid].astype(np.float64) * pg.time_span
    cl = pg.cluster_id[valid]
    ids = np.unique(cl)
    if ids.size < 2:
        return True
    d = np.abs(t.reshape(-1, 1) - t.reshape(1, -1))
    masks = {c: cl == c for c in ids}
    span = {c: (t[masks[c]].min(), t[masks[c]].max()) for c in ids}
    gaps = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            ca, cb = ids[a], ids[b]
            (lo_a, hi_a), (lo_b, hi_b) = span[ca], span[cb]
            if lo_a <= hi_b and lo_b <= hi_a:
                continue                       # interleaved: uninformative about the clock
            gaps.append(d[np.ix_(masks[ca], masks[cb])].min())
    if not gaps:
        return True
    return bool(np.median(gaps) >= window_s)


def build_cache(pg: PackedGallery, cfg: RewardConfig) -> GalleryCache:
    n = pg.n
    sim = (pg.clip @ pg.clip.T).astype(np.float32)
    np.clip(sim, -1.0, 1.0, out=sim)

    t = pg.time_norm.reshape(-1, 1)
    dt = np.abs(t - t.T).astype(np.float32)

    # Jaccard over identity multi-hot: |A&B| / |A|B||. Handle empty sets as 0.
    mh = pg.cast_multihot.astype(np.float32)
    if mh.shape[1] > 0:
        inter = mh @ mh.T
        sizes = mh.sum(1, keepdims=True)
        union = sizes + sizes.T - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            cast_jaccard = np.where(union > 0, inter / union, 0.0).astype(np.float32)
    else:
        cast_jaccard = np.zeros((n, n), dtype=np.float32)

    face_sim = (pg.face_emb @ pg.face_emb.T).astype(np.float32)
    face_defined = pg.face_valid.reshape(-1, 1) & pg.face_valid.reshape(1, -1)
    face_sim = np.where(face_defined, face_sim, 0.0).astype(np.float32)

    has_cast = pg.cast_multihot.sum(1) > 0 if pg.cast_multihot.shape[1] > 0         else np.zeros(n, dtype=bool)
    cast_defined = has_cast.reshape(-1, 1) & has_cast.reshape(1, -1)

    # Facility-location reference for `photocov` (see terms.photo_coverage). Skipped entirely when
    # the term carries no weight -- it is O(budget * N^2) and every reset pays for it otherwise.
    cov_ref = 0.0
    if cfg.weights.get("photocov", 0.0) > 0.0:
        budget = max(int(round(pg.shape.target_pages * pg.shape.target_photos_per_page)), 1)
        cov_ref = _greedy_cover(sim, budget)

    style_conflict = (pg.style_class.reshape(-1, 1) != pg.style_class.reshape(1, -1))
    bw_conflict = (pg.bw.reshape(-1, 1) != pg.bw.reshape(1, -1))

    # near-duplicate: visually near-identical, temporally close, same people.
    both_valid_t = pg.time_valid.reshape(-1, 1) & pg.time_valid.reshape(1, -1)
    if cfg.dup_dt_seconds > 0 and cfg.dup_time_cluster_gate and \
            not timestamps_separate_clusters(pg, cfg.dup_dt_seconds):
        # This gallery's timestamps do not resolve its clusters, so they cannot testify to "same
        # moment" -- judge on appearance and cast alone rather than on a coin flip.
        close_time = np.ones((n, n), dtype=bool)
    elif cfg.dup_dt_seconds > 0:
        # Absolute window. `dt` is normalized by the gallery's own span, so scale it back into
        # seconds -- a fraction-of-span window means 12 s in a 20-minute gallery and 3.6 days in a
        # year-long one (see RewardConfig.dup_dt_seconds). Both timestamps must exist: without a
        # time there is no evidence two similar photos are the SAME MOMENT rather than a repeated
        # pose, and the fraction path's bypass silently asserted it.
        close_time = (dt * pg.time_span <= cfg.dup_dt_seconds) & both_valid_t
    else:
        close_time = (dt <= cfg.dup_dt_frac) | (~both_valid_t)
    same_cast = cast_jaccard >= 0.5
    is_dup = (sim >= cfg.dup_sim) & close_time & same_cast
    np.fill_diagonal(is_dup, False)

    return GalleryCache(
        sim=sim, dt=dt, cast_jaccard=cast_jaccard, face_sim=face_sim,
        style_conflict=style_conflict, bw_conflict=bw_conflict, is_dup=is_dup,
        cast_defined=cast_defined, face_defined=face_defined, cov_ref=cov_ref,
    )
