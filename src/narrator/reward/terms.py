"""Pure reward-term functions.

All terms are normalized to roughly [0, 1] (higher = better) so a weighted sum keeps
comparable magnitudes. They read the packed gallery and its precomputed caches; no
policy or env state leaks in here, which keeps them trivially unit-testable.

Two groups:
  * page-local terms (visual / temporal / cast / softgroup) -- combined into the
    potential Phi(s) that drives potential-based shaping.
  * global terms (coverage / narrative / diversity / shape / selection / dedup) --
    computed on the finished, ordered album.
"""

from __future__ import annotations

import numpy as np

from ..data.preprocess import GalleryCache
from ..data.schema import PackedGallery, ShapeSpec


def _pair_mean(mat: np.ndarray, idx: list[int]) -> float:
    """Mean of the strict upper triangle of mat restricted to idx x idx."""
    if len(idx) < 2:
        return 0.0
    sub = mat[np.ix_(idx, idx)]
    iu = np.triu_indices(len(idx), k=1)
    return float(sub[iu].mean())


def _attr_coherence(a: np.ndarray, idx: list[int]) -> float:
    """1 - mean pairwise |Δ| of a scalar attribute over the page (in [0,1])."""
    if len(idx) < 2:
        return 0.0
    vals = a[idx]
    d = np.abs(vals.reshape(-1, 1) - vals.reshape(1, -1))
    iu = np.triu_indices(len(idx), k=1)
    return float(1.0 - d[iu].mean())


# ----------------------------- page-local terms ----------------------------- #

def visual_coherence(idx: list[int], cache: GalleryCache) -> float:
    return float(np.clip(_pair_mean(cache.sim, idx), 0.0, 1.0))


def cluster_purity(idx: list[int], pg: PackedGallery) -> float:
    """1 when every photo on the page comes from ONE content cluster, decaying linearly to 0
    when they all come from different ones.

    ``content_cluster.pb`` already groups photos by similarity *and* time, so this reuses a
    better grouping signal than raw pairwise cosine -- and it decouples the page-coherence term
    from ``dedup``, which reads the same cosine matrix (measured visual<->dedup = -0.67; see
    docs/reward-collinearity.md).

        purity(P) = 1 - (U - 1) / (|P| - 1)

    where U counts distinct clusters on the page. Unclustered photos (``cluster_id < 0``, 5.8%
    of the real corpus) each count as their own cluster, so a page cannot collect them for free.
    """
    if len(idx) < 2:
        return 0.0
    ids = pg.cluster_id[idx]
    n_unclustered = int((ids < 0).sum())
    n_distinct = len(set(ids[ids >= 0].tolist())) + n_unclustered
    return float(np.clip(1.0 - (n_distinct - 1) / (len(idx) - 1), 0.0, 1.0))


def temporal_coherence(idx: list[int], cache: GalleryCache) -> float:
    # dt in [0,1]; closeness = 1 - dt.
    return float(np.clip(_pair_mean(1.0 - cache.dt, idx), 0.0, 1.0))


# What `cast` returns for a page carrying no cast evidence at all under ``"defined"`` (no pair has
# either half defined): mid-range, so such a page is neither maximally punished as it is today (0.0,
# which is the artifact being fixed) nor rewarded. It sits BELOW the measured means of pages that do
# have evidence (formal 0.815, candid 0.741), so people-free pages cannot become free `cast` credit.
CAST_NO_EVIDENCE = 0.5


def dominant_identity_share(idx: list[int], pg: PackedGallery) -> float | None:
    """Share of the page's PEOPLED photos that contain its most frequent identity.

    The candid reading of "cast consistency": documentary coverage of a moment has a shared
    protagonist, not an identical cast list in every frame. Photos with no identified cast carry no
    evidence and leave the denominator (the same principle as ``cast_defined``). Returns None -- no
    evidence -- when fewer than two photos are peopled, since a lone portrait would otherwise score
    a free 1.0.
    """
    mh = pg.cast_multihot[idx]
    if mh.shape[1] == 0:
        return None
    peopled = mh.sum(1) > 0
    n = int(peopled.sum())
    if n < 2:
        return None
    return float(mh[peopled].sum(0).max() / n)


def cast_consistency(idx: list[int], cache: GalleryCache, mode: str = "zero",
                     pg: PackedGallery | None = None) -> float:
    """Do the same people recur across the page? Mean over pairs of the identity-set Jaccard and
    the face-embedding cosine. ``mode`` decides how missing evidence and page STYLE are handled:

        "zero"    - historical. An undefined half contributes 0.0, i.e. the maximum penalty.
        "defined" - average only the halves that ARE defined for each pair, drop pairs with
                    neither, and fall back to ``CAST_NO_EVIDENCE`` if the page has no evidence.
        "style"   - "defined" for FORMAL pages; for CANDID pages replace the pairwise Jaccard with
                    ``dominant_identity_share`` (requires ``pg``), keeping the defined face half.

    See ``RewardConfig.cast_mode`` for the measurements behind each.
    """
    face = np.clip(cache.face_sim, 0.0, 1.0)
    if mode not in ("defined", "style"):
        combined = 0.5 * cache.cast_jaccard + 0.5 * face
        return float(np.clip(_pair_mean(combined, idx), 0.0, 1.0))
    if len(idx) < 2:
        return 0.0                      # as every other page-local term does for a 1-photo page
    ii = np.ix_(idx, idx)
    iu = np.triu_indices(len(idx), k=1)
    dj = cache.cast_defined[ii][iu]
    df = cache.face_defined[ii][iu]

    if mode == "style":
        if pg is None:
            raise ValueError('cast_consistency(mode="style") needs the PackedGallery to read '
                             'page style and identities')
        # Style is uniform within a page (the don't-mix mask forbids mixing), so read it once.
        if int(pg.style_class[idx[0]]) == 0:
            dom = dominant_identity_share(idx, pg)
            if dom is None:
                # No identity evidence at all; the face half alone still speaks if it is defined.
                if not df.any():
                    return CAST_NO_EVIDENCE
                return float(np.clip(face[ii][iu][df].mean(), 0.0, 1.0))
            if not df.any():
                return float(np.clip(dom, 0.0, 1.0))
            return float(np.clip(0.5 * dom + 0.5 * face[ii][iu][df].mean(), 0.0, 1.0))

    num = np.where(dj, cache.cast_jaccard[ii][iu], 0.0) + np.where(df, face[ii][iu], 0.0)
    den = dj.astype(np.float64) + df.astype(np.float64)
    ok = den > 0
    if not ok.any():
        return CAST_NO_EVIDENCE
    return float(np.clip((num[ok] / den[ok]).mean(), 0.0, 1.0))


def softgroup_coherence(idx: list[int], pg: PackedGallery) -> float:
    if len(idx) < 2:
        return 0.0
    parts = [
        _attr_coherence(pg.indoor, idx),
        _attr_coherence(pg.lighting, idx),
        _attr_coherence(pg.bgcolor, idx),
    ]
    return float(np.clip(np.mean(parts), 0.0, 1.0))


def page_size_fit(idx: list[int], pg: PackedGallery, mode: str = "band") -> float:
    """Does the page hold the right number of photos for its STYLE?

    Formal pages want ``formal_page_min..max`` (2-3), candid pages ``candid_page_min..max``
    (3-4). Style is uniform within a page (the don't-mix mask forbids mixing), so it is read
    from the first photo. ``mode`` selects how the range is scored (see
    ``RewardConfig.pagesize_mode``):

        "band" - historical. Full credit anywhere INSIDE the range, -0.5 per photo outside:
                 fit(P) = max(0, 1 - 0.5 * distance_outside_range)
        "peak" - full credit at the TOP of the style's range, -0.5 per photo away from it in
                 either direction: fit(P) = max(0, 1 - 0.5 * |k - hi|)

    NOTE: when ``cfg.page_size_enforced`` is True the same range is a hard mask, so under "band"
    this term is 1.0 by construction and contributes no gradient. It only earns its weight with
    the mask relaxed to the global bounds. See docs/reward-terms.md.
    """
    if not idx:
        return 0.0
    s = pg.shape
    formal = int(pg.style_class[idx[0]]) == 1
    lo, hi = ((s.formal_page_min, s.formal_page_max) if formal
              else (s.candid_page_min, s.candid_page_max))
    k = len(idx)
    dist = abs(k - hi) if mode == "peak" else max(lo - k, k - hi, 0)
    return float(np.clip(1.0 - 0.5 * dist, 0.0, 1.0))


PAGE_LOCAL_TERMS = ("visual", "temporal", "cast", "softgroup", "pagesize")


def page_local(idx: list[int], pg: PackedGallery, cache: GalleryCache,
               visual_mode: str = "clip", pagesize_mode: str = "band",
               cast_mode: str = "zero") -> dict[str, float]:
    """Page-local terms. ``visual_mode`` selects how page coherence is measured:
    ``"clip"`` = mean pairwise CLIP cosine (original), ``"cluster"`` = content-cluster purity.
    ``pagesize_mode`` selects how the style range is scored (see ``page_size_fit``), and
    ``cast_mode`` how undefined cast evidence is handled (see ``cast_consistency``)."""
    return {
        "visual": (cluster_purity(idx, pg) if visual_mode == "cluster"
                   else visual_coherence(idx, cache)),
        "temporal": temporal_coherence(idx, cache),
        "cast": cast_consistency(idx, cache, cast_mode, pg),
        "softgroup": softgroup_coherence(idx, pg),
        "pagesize": page_size_fit(idx, pg, pagesize_mode),
    }


# ------------------------------- global terms ------------------------------- #

def people_coverage(included: list[int], pg: PackedGallery) -> float:
    """Fraction of distinct identities present + balance of appearance counts."""
    if pg.n_persons == 0 or not included:
        return 0.0
    counts = pg.cast_multihot[included].sum(0).astype(np.float64)  # (P,)
    present = (counts > 0).mean()
    total = counts.sum()
    if total > 0:
        probs = counts[counts > 0] / total
        entropy = -(probs * np.log(probs)).sum()
        balance = entropy / np.log(len(counts)) if len(counts) > 1 else 1.0
    else:
        balance = 0.0
    return float(0.7 * present + 0.3 * balance)


def _page_median_time(page: list[int], pg: PackedGallery) -> float:
    valid = [i for i in page if pg.time_valid[i]]
    if not valid:
        return float("nan")
    return float(np.median(pg.time_norm[valid]))


def narrative_order(ordered_pages: list[list[int]], pg: PackedGallery) -> float:
    """Spearman correlation of page position vs page median time, mapped to [0,1]."""
    meds = np.array([_page_median_time(p, pg) for p in ordered_pages], dtype=np.float64)
    valid = np.isfinite(meds)
    if valid.sum() < 2:
        return 0.5
    from scipy.stats import spearmanr
    pos = np.arange(len(ordered_pages))[valid]
    rho, _ = spearmanr(pos, meds[valid])
    if not np.isfinite(rho):
        return 0.5
    return float((rho + 1.0) / 2.0)


def _page_mean_clip(page: list[int], pg: PackedGallery) -> np.ndarray:
    v = pg.clip[page].mean(0)
    return v / max(np.linalg.norm(v), 1e-8)


def page_diversity(ordered_pages: list[list[int]], pg: PackedGallery) -> float:
    """Mean dissimilarity of adjacent pages (1 - cosine of page-mean CLIP)."""
    pages = [p for p in ordered_pages if p]
    if len(pages) < 2:
        return 0.0
    means = np.stack([_page_mean_clip(p, pg) for p in pages])
    diss = [1.0 - float(means[i] @ means[i + 1]) for i in range(len(pages) - 1)]
    return float(np.clip(np.mean(diss), 0.0, 1.0))


def _gauss(x: float, mu: float, sigma: float) -> float:
    return float(np.exp(-0.5 * ((x - mu) / max(sigma, 1e-6)) ** 2))


def shape_preference(ordered_pages: list[list[int]], shape: ShapeSpec,
                     use_avg_pp: bool = True) -> float:
    """Is the album the right SHAPE? A Gaussian on the page count, optionally averaged with a
    second Gaussian on the mean photos-per-page (``use_avg_pp``; see
    ``RewardConfig.shape_use_avg_pp`` for why that half is retired)."""
    pages = [p for p in ordered_pages if p]
    if not pages:
        return 0.0
    n_pages = len(pages)
    g_pages = _gauss(n_pages, shape.target_pages,
                     max(shape.target_pages * shape.pages_sigma_frac, 1e-3))
    if not use_avg_pp:
        return float(g_pages)
    avg_pp = float(np.mean([len(p) for p in pages]))
    g_pp = _gauss(avg_pp, shape.target_photos_per_page,
                  max(shape.target_photos_per_page * shape.pp_sigma_frac, 1e-3))
    return float(0.5 * g_pages + 0.5 * g_pp)


def scene_coverage(included: list[int], pg: PackedGallery) -> float:
    """Are all the EVENTS of the session in the album? A weighted set-cover over scenes.

        scene_coverage = sum(sqrt(n_s) for covered scenes) / sum(sqrt(n_s) for the top-K scenes)

    where n_s is the scene's photo count, a scene is *covered* when at least one of its photos is in
    the album, and K = the album's target page count.

    Three decisions, each load-bearing:

    * **sqrt weighting, not photo-weighted.** A photo-weighted mean distance to page centres was
      measured first and is the wrong shape: it saturated at 0.854 and correlated **-0.339** with
      album_score, because it rewards spending the album on the biggest scene. sqrt makes a big scene
      matter more than a small one but sub-linearly, so a 1-cluster scene is still worth covering --
      with scene sizes 10/8/5/3/1 the smallest is 3.4% of the photos but **9.1% of the term**.

    * **Presence, not proportion.** One photo covers a scene. The requirement is that every place of
      the session appears, not that each gets its pro-rata share.

    * **Normalised by the top-K scenes, not by all of them.** An album of K pages can realistically
      draw from ~K scenes, so dividing by every scene would hand any gallery with more scenes than
      pages a permanent deficit -- exactly the "big galleries must not suffer more than small" trap.
      K comes from ``shape.target_pages`` and NOT from the album's own page count, which the policy
      chooses: a policy-dependent denominator could be raised by building fewer pages.

    Returns 1.0 when the gallery has no scenes to cover (all photos unclustered) -- abstention, not
    a free win, since the policy cannot make photos unclustered.
    """
    if not len(pg.scene_id):
        return 1.0
    sid = pg.scene_id
    scenes, sizes = np.unique(sid[sid >= 0], return_counts=True)
    if not len(scenes):
        return 1.0
    w = np.sqrt(sizes.astype(np.float64))
    k = max(int(round(pg.shape.target_pages)), 1)
    denom = float(np.sort(w)[::-1][:k].sum())
    if denom <= 0:
        return 1.0
    covered = np.isin(scenes, np.unique(sid[included][sid[included] >= 0])) if included         else np.zeros(len(scenes), dtype=bool)
    return float(np.clip(w[covered].sum() / denom, 0.0, 1.0))


def photo_coverage(included: list[int], cache: GalleryCache) -> float:
    """How well do the SELECTED PHOTOS cover the gallery's content? Facility location.

        photocov = mean_i max_{j in album} cos(i, j)  /  cache.cov_ref

    i.e. every gallery photo is represented by the album photo most like it, averaged, and normalised
    by what a good album of `target_pages * target_photos_per_page` photos achieves on this gallery
    (`_greedy_cover`, computed at cache time; gallery-only and budget-fixed, so the policy cannot
    raise the denominator by building a smaller album).

    WHY THE COVERING SET IS PHOTOS, NOT PAGE CENTRES. The first version of this idea measured distance
    to page MEAN vectors and was rejected on measurement -- it saturated at 0.854 and correlated
    -0.339 with album_score. A page mean of several diverse photos is a centroid close to everything
    and far from nothing, which destroys exactly the signal wanted. Using individual selected photos
    makes the objective a proper nearest-neighbour cover.

    WHY THIS IS THE TERM THAT SIZES A PAGE. It is submodular, so its marginal value depends on the
    LOCAL DIVERSITY of what is already selected:
      * homogeneous cluster -- 2 photos already cover their near-identical neighbours, so a 3rd adds
        ~nothing (and `dedup` charges for it). Two photos is correct and the reward says so.
      * diverse cluster -- the 3rd photo reaches content the first two do not, so it pays.
    No other term can express that: `scene` fires on scene PRESENCE (one photo covers a scene) and is
    blind to within-cluster depth, while the page-local terms are pairwise means that only ever FALL
    as a page grows. This is what makes 2-vs-3 a property of the cluster rather than a global rule --
    which is why enforcing the page-size range instead would be wrong: a mask treats a homogeneous
    cluster and a diverse one identically.

    Returns 0.0 for an empty album, and 1.0 when the reference could not be computed (the term is
    unweighted, so the value is never used).
    """
    if cache.cov_ref <= 0.0:
        return 1.0
    if not included:
        return 0.0
    s = np.clip(cache.sim[:, included], 0.0, 1.0)
    return float(np.clip(s.max(1).mean() / cache.cov_ref, 0.0, 1.0))


def selection_quality(included: list[int], pg: PackedGallery) -> float:
    if not included:
        return 0.0
    return float(np.clip(pg.selection[included].mean(), 0.0, 1.0))


def dedup(included: list[int], cache: GalleryCache, mode: str = "any") -> float:
    """1 - fraction of included photos that have a near-duplicate also included.

    NOTE: when ``cfg.dedup_enforced`` is True the masking makes this 1.0 by construction, so the
    term contributes no gradient -- the same situation ``page_size_fit`` has under
    ``page_size_enforced``. That is the point: it reads the same cosine matrix as ``visual`` and
    loses to it 12:1 at a 1/6 global scale, so it is enforced rather than scored. See
    RewardConfig.dedup_enforced.
    """
    if len(included) < 2:
        return 1.0
    sub = cache.is_dup[np.ix_(included, included)]
    if mode == "excess":
        # Count each duplicate GROUP's surplus once: a connected component of k near-duplicates
        # leaves k-1 redundant photos. `any` instead flags every member, so a pair costs 2/n
        # instead of 1/n -- double-counting that is the binding tax on album size (see
        # RewardConfig.dedup_mode).
        n = len(included)
        seen = np.zeros(n, dtype=bool)
        redundant = 0
        for i in range(n):
            if seen[i]:
                continue
            stack, size = [i], 0
            seen[i] = True
            while stack:
                j = stack.pop()
                size += 1
                for k in np.nonzero(sub[j] & ~seen)[0]:
                    seen[k] = True
                    stack.append(int(k))
            redundant += size - 1
    else:
        redundant = int(sub.any(1).sum())
    return float(1.0 - redundant / len(included))


GLOBAL_TERMS = ("coverage", "narrative", "diversity", "shape", "selection", "dedup", "scene",
                "photocov")
