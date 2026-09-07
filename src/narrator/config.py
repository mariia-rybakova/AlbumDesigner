"""Central configuration: dataclasses that are the single source of truth for the
env shape, reward weights, model dims, and PPO hyperparameters.

Configs are plain dataclasses with sensible defaults. `load_config(path)` overlays a
YAML file (nested dict) onto the defaults so experiments are config diffs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import yaml


@dataclass
class EnvConfig:
    # Feature dims (must match ingest / mock generator output).
    clip_dim: int = 768          # fine-tuned V2 CLIP image embedding
    face_dim: int = 512          # raw face embedding
    body_dim: int = 2048         # raw body ReID embedding
    n_attr: int = 4              # candid, indoor, lighting, bgcolor CLIP-axis scores

    # Album shape (masking enforces the hard edges).
    # Per-page photo count is STYLE-dependent: formal pages 2-3, candid pages 4-5.
    # min/max_photos_per_page are the overall floor/ceiling and clamp the style bounds
    # (so tighter test/experiment configs stay valid).
    min_photos_per_page: int = 2
    max_photos_per_page: int = 5
    formal_page_min: int = 2      # style_class == 1 (formal)
    formal_page_max: int = 3
    candid_page_min: int = 3      # style_class == 0 (candid)
    candid_page_max: int = 4
    min_pages: int = 12
    max_pages: int = 17
    target_photos_per_page: float = 3.5
    target_pages: float = 14.0
    # Width of the shape_preference Gaussians as a fraction of the target (see ShapeSpec).
    shape_pages_sigma_frac: float = 0.4
    shape_pp_sigma_frac: float = 0.4

    # Scale handling.
    cap_photos: int = 512        # stratified-cap galleries larger than this
    step_budget_factor: float = 3.0  # truncate after factor * n_photos steps

    # Hard-constraint thresholds.
    candid_threshold: float = 0.5    # candid_score >= threshold => candid, else formal

    # Cosine-distance cut that groups content clusters into SCENES (see schema._derive_scenes).
    # A cluster is one moment; a scene is one place/event and usually spans several clusters -- the
    # median real gallery has 30.5 clusters. 0.35 was chosen by inspection of the resulting groups.
    scene_cut: float = 0.35


@dataclass
class RewardConfig:
    # Weights per named term. Page-local terms are shaped (potential-based);
    # global terms are awarded at/after album completion.
    weights: dict[str, float] = field(default_factory=lambda: {
        # page-local (shaped)
        "visual": 1.0,
        "temporal": 0.7,
        "cast": 0.6,
        "softgroup": 0.4,
        "pagesize": 0.5,
        "dedup": 0.5,
        # global (terminal)
        "coverage": 0.8,
        "narrative": 0.7,
        "diversity": 0.5,
        "shape": 0.3,
        "selection": 0.3,
        # Scene set-cover: are all the events/locations of the session represented at all? Defaults
        # to 0.0 so it is inert until a config opts in, leaving every v1-v27 score untouched.
        # Measured headroom is large: with a 10-page budget the reachable weighted scene coverage is
        # 87.1% and v27 attains 76.4%, a 10.6-point gap. See terms.scene_coverage.
        "scene": 0.0,
        # Photo-level coverage (facility location over the SELECTED PHOTOS -- see
        # terms.photo_coverage). This is the term that decides how many photos a page deserves, and it
        # decides it per cluster: a homogeneous cluster is fully covered by 2 photos so a 3rd earns
        # nothing, while a diverse cluster pays for the 3rd. Defaults to 0.0 -- and `build_cache`
        # skips its O(budget*N^2) reference computation entirely while it is 0.
        "photocov": 0.0,
    })
    reward_scale: float = 1.0

    # How page coherence (the "visual" term) is measured:
    #   "clip"    - mean pairwise CLIP cosine over the page (original).
    #   "cluster" - purity w.r.t. content_cluster.pb, which already groups photos by similarity
    #               AND time. Reuses that grouping instead of re-deriving it from raw cosine,
    #               and decouples the term from `dedup`, which reads the same cosine matrix
    #               (visual<->dedup measured -0.67; see docs/reward-collinearity.md).
    # Default stays "clip" so existing checkpoints keep their semantics when reloaded.
    visual_mode: str = "clip"

    # Whether the style-dependent page-size range is a HARD mask (True, historical) or only a
    # scored preference via the `pagesize` term (False). With the mask on, `pagesize` is 1.0 by
    # construction and carries no gradient -- the same failure `shape` was measured to have
    # (corr +0.115). Set False to let the reward express the preference instead.
    page_size_enforced: bool = True

    # How the `pagesize` term scores the style range:
    #   "band" - historical. Full credit ANYWHERE inside the range, -0.5 per photo outside.
    #   "peak" - full credit at the TOP of the range, -0.5 per photo away from it either way.
    #
    # Why "band" is broken, measured on v27: it prices STYLE, not size. Two defects compound.
    #   (i) FLAT TOP. Inside a style, k = lo scores exactly as well as k = hi, so there is no
    #       reason ever to exceed the floor. v27's formal pages sit at 2.00 photos although 3 is
    #       equally legal and equally paid.
    #  (ii) STYLE-CONDITIONAL FLOOR. `formal_page_min` (2) equals the global minimum while
    #       `candid_page_min` is 3, so a 2-photo formal page earns 1.00 where a 2-photo candid page
    #       earns 0.50. The cheapest legal scoring unit in the whole MDP is a formal pair.
    # Together these explain BOTH halves of the observation exactly: 81.9% of v27's body pages are
    # formal (the cheaper style) and every one holds exactly 2.00 photos (the floor of the flat top),
    # stranding candid material at 18.1% of the album against 36-50% of supply.
    #
    # "peak" removes both: each style's floor now costs the same 0.50, so neither style is cheaper
    # at minimum size, and the credit rises with k up to the style ceiling, so a marginal photo is
    # paid rather than merely tolerated. The functional form and the 0.5/photo slope are unchanged --
    # only the reference point moves from "the range" to "the top of the range".
    #
    # NOTE this also re-prices a 1-photo page (0.50 -> 0.00). That only reaches the score at all
    # when `hero_pages` is on WITHOUT `hero_exempt_local`; v16+ set both.
    pagesize_mode: str = "band"

    # What `cast` does when a pair carries NO evidence:
    #   "zero"    - historical. `build_cache` sets cast_jaccard = 0 when the identity union is empty
    #               and face_sim = 0 when EITHER photo lacks a face embedding, so "no evidence"
    #               scores as the MAXIMUM penalty and is indistinguishable from "no people shared".
    #   "defined" - average only the halves that are defined for each pair, drop pairs with neither.
    #   "style"   - "defined" for FORMAL pages, but ask CANDID pages a style-appropriate question:
    #               the share of the page's PEOPLED photos containing its most frequent identity
    #               (`dominant_identity_share`), blended 50/50 with the defined face half.
    #
    # Why: the two halves disagreed about the same situation. `face_sim` already required the signal
    # in BOTH photos; `cast_jaccard` required it in only one, so a portrait paired with a people-free
    # detail shot scored a hard 0.0 on cast rather than abstaining. Candid photos hit this constantly
    # -- measured over the 24 withheld galleries, 13.3% have an empty cast and 12.5% no valid face
    # against formal's 4.9% / 3.9%, so on cluster-pure candidate pages 23.1% of candid pairs have an
    # undefined Jaccard half (formal 6.0%) and 12.5% of candid candidate pages score EXACTLY 0.0
    # today (formal 4.3%).
    #
    # The result was a structural formal-photo subsidy, and it is 60% artifact: the formal-candid
    # `cast` gap is -0.1825 as scored today and -0.0735 once undefined halves abstain. The residual
    # -0.0735 is real -- candid pages do vary more in who appears -- and is left alone.
    #
    # This matters because `cast` (weight 0.6) is PAGE-LOCAL and so carries full weight, while the
    # terms that favour candid pages (`coverage`, `shape`) are globals held at 1/3 in training: at
    # gs=1/3 a formal pair paid +0.0202/photo against a candid triple's +0.0129 (1.57x), and neither
    # `page_local_reduce: "photo"` nor `pagesize_mode: "peak"` moved that ratio.
    # Why "style": the pairwise form asks whether every frame shows the SAME SET of people. That is
    # the right question for a posed formal page and the wrong one for a candid page, which documents
    # a moment and has a shared protagonist rather than an identical cast list. Asking it of both
    # styles leaves a -0.0939 formal-candid gap even after the definedness fix.
    #
    # Measured on 483 formal / 224 candid cluster-pure candidate pages (levels include the
    # CAST_NO_EVIDENCE abstention, which is why they sit below the pair-level figures quoted above):
    #
    #   form         formal  candid     gap   candid sd   >=0.95   ==1.0
    #   zero         0.7819  0.5994  -0.1825      0.3467    10.7%    0.0%
    #   defined      0.8048  0.7109  -0.0939      0.2395    13.4%    0.9%
    #   style        0.8479  0.7709  -0.0770      0.1953    13.4%    0.9%   <- shipped
    #   no face half 0.9299  0.8808  -0.0491      0.2225    75.9%   75.9%   <- rejected: exploit
    #   relax x0.7   0.8634  0.7977  -0.0657      0.1677    21.4%    0.9%   <- rejected: rescale
    #
    # Dropping the face half for candid pages closes more of the gap but hands 75.9% of candid pages
    # exactly 1.0, which buys the style balance by making `cast` (weight 0.6) meaningless on 36-50%
    # of the supply. A tuned deficit relaxation correlates +1.000 with "defined" -- a monotone
    # rescale carrying no new information. "style" correlates +0.906, so it reads real structure, and
    # its >=0.95 share is unchanged from "defined": it does not flatten.
    cast_mode: str = "zero"

    # Whether `shape` keeps its second half, a Gaussian on the album's MEAN PHOTOS PER PAGE.
    # True is historical: shape = 0.5 * g(n_pages) + 0.5 * g(avg_photos_per_page).
    #
    # Retire it (False) because it double-counts `pagesize`, which already prices per-page size and
    # does it better -- per STYLE (formal 2-3, candid 3-4) rather than against one global 3.5 target,
    # so the avg-pp half penalises any album whose style mix is not centred on 3.5 even when every
    # page is the right size for its own style. Two terms pricing the same quantity from different
    # premises is the collinearity trap that docs/reward-collinearity.md documents elsewhere.
    #
    # It also halves the one thing `shape` uniquely measures. With it on, moving the album from 10
    # pages to the 14-page target can raise `shape` by at most 0.5 * (1 - g(10)); with it off, the
    # page count gets the term's whole dynamic range.
    #
    # ⚠️ `shape`'s long-standing failure (0.701 in v27, its lowest across all four scan arms) was
    # attributed in the log to the actor not observing page count. That was true before v25 and is
    # NOT why v27 failed: v27 ran `model.album_state_features: true`, whose feature 1 is
    # `pages_vs_target`, and its actor MLPs are 778/774 wide accordingly. The real cause was the
    # sigma bug (see data/schema.py): v27 asked for shape_pages_sigma_frac 0.15 (sigma 2.1 pages) and
    # actually ran at the 0.4 default (sigma 5.6), where a 10-page album still scores 0.775 on the
    # page half instead of 0.163. Weighted through 0.3 x global_scale 1/3, shrinking from 14 pages to
    # 10 cost 0.011 -- the eval noise floor. `shape` was not unobservable, it was FLAT.
    shape_use_avg_pp: bool = True

    # Force single-photo opening and closing pages ("hero" pages): the opening photo is drawn
    # from the earliest 25% of the gallery by time, the closing one from the latest 25%, both
    # chosen to feature the most frequently appearing group of people. Selected deterministically
    # at reset and withheld from the assignable pool, so the policy composes only the body.
    hero_pages: bool = False
    # Order the finished pages CHRONOLOGICALLY by construction (page median capture time) instead
    # of leaving the order to the policy. Phase B then emits no decisions at all: the env finalizes
    # phase A and immediately places every page in time order.
    #
    # Why: measured on v19a's own best-of-8 albums, sorting the pages it produced -- changing nothing
    # else -- is worth **+0.126 album_score** (narrative 0.727 -> 0.951), better in 22 of 24
    # galleries, with page composition untouched. For scale, v19a's entire advantage over v16 was
    # +0.042, so a trivial sort is worth 3x every gain to date.
    #
    # Why the policy never learned it: `narrative` has weight 0.7, but training holds global_scale at
    # 1/6, so its effective weight is 0.117 and a 0.224 gain is worth 0.026 in the training reward --
    # at the eval noise floor -- while EVAL (scale 1.0) prices the same gain at 0.157. The eval
    # rewards ordering 6x more than the training signal does, and with the globals off it is zero.
    # Rather than pay more for it and re-break page quality, construct it: this is the project's
    # "structure via construction, not via penalty" principle, already used for the hero bookends.
    #
    # ⚠️ `diversity` rewards ADJACENT pages being dissimilar, which directly fights chronological
    # order; it drops ~0.062. Net is still +0.126. That term looks mis-specified.
    order_by_time: bool = False
    # Exempt the two hero bookends from the PAGE-LOCAL terms (no effect unless `hero_pages`).
    # Every page-local term is a pairwise mean and so returns 0.0 for a 1-photo page, which
    # scores a structural choice the policy did not make as two failed pages. Measured over 24
    # real galleries under configs/v16.yaml: the weighted page-local potential reads 2.176
    # instead of 2.525 (x0.858 = (K-2)/K on a 14-page album) and every coherence term is
    # understated by ~1.17x. Worse, a hero page's contribution (~0.18 weighted, all of it from
    # `pagesize`) sits far below the body-page average (~2.53), so under the `mean` reduction
    # ADDING any body page raises Phi by ~0.02 through dilution alone -- a page-count incentive
    # unrelated to page quality, twice the eval noise floor (0.010).
    # Only the page-local terms are exempted. The heroes stay in `included` for
    # coverage/selection/dedup and stay real pages for narrative/diversity/shape, because the
    # shipped album genuinely has them. Default False so v14-v16 keep their scoring.
    hero_exempt_local: bool = False

    # How page-local terms combine across pages in Phi(s) (and therefore in album_score).
    #   "sum"  - historical behaviour. Multiplies every page-local term by the page count
    #            (~14.6), so page-local terms are ~94% of album_score and the six global
    #            terms ~6% -- nothing like the nominal weights. See docs/reward-audit.md.
    #   "mean" - average over non-empty pages, so the weights above mean what they appear to
    #            and album_score is independent of album length.
    # Default stays "sum" so checkpoints trained under it keep their semantics when reloaded.
    #   "photo" - mean over pages WEIGHTED BY PAGE SIZE, i.e. a mean over photos. Fixes the
    #             small-page arbitrage that "mean" creates: under "mean" a 2-photo pure page and a
    #             4-photo pure page both contribute 1.0, so minimum-size pages are free purity and
    #             free `pagesize`. v27 exploited exactly that -- 81.9% formal pages, ALL of them
    #             exactly 2.00 photos (formal's range starts at 2, candid's at 3), stranding candid
    #             material at 18.1% of the album against 36-50% of supply, and settling on ~21
    #             photos where the heuristic uses ~46. Weighting by page size makes a page's
    #             contribution proportional to the photos it commits, so purity on a BIG page is
    #             worth more than purity on a small one, and the style asymmetry in `pagesize`
    #             stops being an arbitrage (credit per photo is equal either way).
    #             Still a bounded mean, unlike "sum" -- it re-weights rather than scaling with
    #             album length, so it does not reintroduce the v9-v14 failure.
    page_local_reduce: str = "sum"
    # Move `narrative` from the terminal reward into the shaping potential Phi. album_score is
    # unchanged (the term is still counted once, at the same weight) -- only the credit timing
    # changes: each PLACE in phase B gets dense credit for extending the temporal order instead
    # of the whole ordering being paid for once at the end. narrative carries 31.5% of score
    # variability and the policy scores ~0.57 against a heuristic's 1.000, so its credit path
    # is the bottleneck. See docs/reward-audit.md.
    shape_narrative: bool = False
    # Move these GLOBAL terms out of the terminal reward and into the shaping potential Phi, so each
    # step earns attributable credit for them instead of one lump at termination. Same mechanism as
    # `shape_narrative`, generalized; `album_score` is unchanged because a shaped term is removed
    # from `terminal()` and counted once inside Phi at the same weight and global_scale.
    #
    # WHY (docs/research-log.md §1g). The globals have never moved: v19a's best `coverage` gain was
    # +0.016, v24/v25 are flat to three decimals, and rewarding `coverage` at 1/3 from update 0 did
    # not even slow its decay (-0.069 over the first 60 updates, vs -0.030 where it was worth
    # nothing). Neither observability (v25) nor timing (§1e) explains that. The credit structure
    # does: measured on real galleries, one maximally-coverage-improving ASSIGN is worth **0.0455**
    # of album_score against a **0.1113** rollout-to-rollout std -- a signal/noise of **0.41** that
    # arrives only in `terminal()`, so the policy must infer it through a whole-episode return
    # spanning ~35 steps and 8 galleries. The page-local terms meanwhile get dense per-step
    # potential shaping. That asymmetry is the remaining explanation.
    #
    # Only include terms that are meaningful on a PARTIAL album. `coverage`, `selection` and `dedup`
    # are pure functions of the included set, so they are exact at every step. `shape` is a function
    # of page count and mean page size -- well defined but low early and rising, which is fine for
    # Phi (only differences matter). ⚠️ `diversity` is ORDER-dependent and, under `order_by_time`,
    # the order is not decided until finalize, so shaping it rewards a proxy that need not match the
    # final value; it is left out of the recommended set for that reason. `narrative` is already
    # handled by `shape_narrative` -- listing it here too is deduplicated, not double-counted.
    shape_globals: list[str] = field(default_factory=list)
    # --- phased training ---------------------------------------------------------------
    # On real data the page-local and global terms actively cancel: dedup<->visual -0.67,
    # coverage<->visual -0.35 (docs/reward-collinearity.md). Training the page-local terms
    # ALONE first gives the policy an unconflicted objective to learn page composition against,
    # then the globals ramp in. `global_scale` multiplies every global term (including
    # `narrative` when it lives in Phi) and is set per update by the training loop.
    #   updates < phase_local_updates                     -> scale 0   (page-local only)
    #   ... + phase_ramp_updates                          -> linear 0 -> 1
    #   after                                             -> scale 1   (full reward)
    # EVAL always uses the full reward (scale 1) so scores stay comparable across phases and
    # against earlier runs; only the training env is scaled.
    phase_local_updates: int = 0     # 0 disables phasing
    phase_ramp_updates: int = 0
    # Ceiling the ramp climbs to and then HOLDS at, instead of always finishing at 1.0. This is
    # the fine-tuning knob: the globals become a small persistent nudge the policy can satisfy
    # while keeping the page behaviour it already learned, rather than an objective that
    # eventually outweighs it. Motivated by v18, which ramped 0 -> 1.0 from v16's best and whose
    # only improvement over the 4.036 starting point (4.056 at u2199) occurred while the scale was
    # still ~1/3; every eval after that declined as it approached 1.0. 1.0 = historical behaviour.
    global_scale_max: float = 1.0
    global_scale: float = 1.0        # set at runtime, not meant to be configured directly
    # Enforce near-duplicate exclusion by MASKING instead of scoring it. A photo cannot be
    # assigned while a near-duplicate of it is already in the album, so `dedup` is 1.0 by
    # construction (like `pagesize` under page_size_enforced) and carries no gradient.
    #
    # Why: `dedup` and `visual` read the SAME cosine matrix with overlapping thresholds -- visual
    # rewards high cosine, dup fires at >= dup_sim -- so they cancel by construction (measured
    # correlation -0.67, docs/reward-collinearity.md). The v19 dose-response made the cost
    # concrete: with the globals held at 1/6, five of six global terms improve but `dedup` does
    # not move at all (-0.002), because its effective weight is 0.5/6 = 0.083 against `visual`'s
    # 1.0 on the same matrix -- 12:1 against. It only moves when the global scale reaches 1.0
    # (+0.049), and precisely there page composition breaks (cast -0.027, pagesize -0.021).
    # Masking removes the contest instead of trying to win it.
    #
    # ⚠️ Turning this on raises album_score by a constant ~w_dedup * (1 - dedup_before) *
    # global_scale, so scores are NOT comparable to runs with it off.
    # How `dedup` counts redundancy:
    #   "any"    - historical. A photo counts as redundant if ANY near-duplicate of it is also in
    #              the album, so a single duplicate PAIR marks BOTH photos and costs 2/n. That
    #              double-counts, and it is the binding tax on album size: measured at v27's
    #              operating point, the marginal photo (the best remaining same-cluster one, which
    #              is usually a near-duplicate) costs -0.0405 of `dedup`, i.e. -0.0202 weighted --
    #              1.4x the entire -0.0147 deficit that keeps pages at 2 photos.
    #   "excess" - count each duplicate GROUP's surplus once: a connected group of k near-duplicates
    #              contributes k-1 redundant photos, so a pair costs 1/n rather than 2/n. This is
    #              the natural reading of "how many redundant photos are in this album" and halves
    #              the marginal tax without weakening what the term is for.
    dedup_mode: str = "any"
    dedup_enforced: bool = False
    gamma_shaping: float = 0.99  # gamma used in potential-based shaping F = g*Phi(s') - Phi(s)
    dup_sim: float = 0.92        # cosine >= this AND close in time AND same cast => near-duplicate
    dup_dt_frac: float = 0.01    # |dt| fraction of gallery span below which two photos are "same moment"
    # ABSOLUTE "same moment" window in seconds. When > 0 this replaces `dup_dt_frac` and also
    # requires both timestamps to be present.
    #
    # `dup_dt_frac` is a fraction of each gallery's OWN span, so one config means different things
    # per gallery: measured p10/p50/p90 span is 20 min / 1 h / 357 days, i.e. a window of 12 s /
    # 37 s / 3.6 days. Audited over 250 real galleries, the current rule flags 90,478 pairs of
    # which **30.2% are more than 10 s apart** and 9.0% more than a minute -- those are photos of
    # the same scene, not duplicates, and masking them out removes legitimate page material.
    # (The missing-timestamp bypass in the fraction path turned out to be harmless: 0.1% of pairs.)
    #
    # `dup_sim` itself is defensible: 0.92 sits at ~the 99th percentile of pairwise CLIP cosine
    # (per-gallery p99 = 0.931), i.e. 1.57% of all pairs. The time test is the loose half.
    #
    # Sensitivity of "share of clustered photos that are near-duplicates" / median dup-free
    # perfect-page ceiling, which is what decides whether dedup masking is affordable:
    #     current (0.92, 1% of span)     44.8%  /  21 pages
    #     0.92 + 10 s                    38.5%  /  25
    #     0.95 + 10 s                    30.4%  /  30      <- recommended
    #     0.95 +  3 s                    24.5%  /  35
    #     0.97 +  3 s                    17.1%  /  41
    # 0 keeps the legacy fraction behaviour so existing runs stay bit-identical.
    dup_dt_seconds: float = 0.0
    # Distrust timestamps PER GALLERY when they fail to separate content clusters.
    #
    # The time test exists to tell "same moment" from "same subject at a different time", and
    # content_cluster.pb already groups by similarity AND time -- so distinct clusters ought to be
    # separated in time. If they are not, `dateTaken` is unusable at this resolution for that
    # gallery (batch/upload dates, or second-granularity truncation), and the time test contributes
    # noise: it drops true duplicates whose stamps happen to differ and admits pairs whose stamps
    # happen to collide. When that is detected, drop the time test for that gallery and fall back
    # to cosine + cast alone.
    #
    # Statistic: median, over pairs of clusters that are TIME-CONSECUTIVE (their [min,max] intervals
    # do not overlap), of the minimum gap between them, compared against `dup_dt_seconds`.
    # Two refinements, both measured over 250 real galleries, both needed:
    #   * MEDIAN not minimum -- 58.4% of galleries contain some cross-cluster pair at a 0 s gap, so a
    #     minimum-based test declares 95.2% of galleries broken.
    #   * TIME-CONSECUTIVE pairs only -- interleaved clusters (a photographer alternating between
    #     subjects) are ~0 s apart however good the clock is. Interleaving is the norm: the median
    #     gallery has 25.3% of its cluster pairs overlapping in time, p75 48.7%. Including them makes
    #     the test measure interleaving rather than clock quality -- it then fires on 16.4% and
    #     correlates more with cluster overlap (+0.565) than with tied timestamps (+0.441), and the
    #     galleries it condemns but this version absolves average 83.3% overlap vs 32.4% elsewhere.
    # Restricted properly the gate fires on **1.2%** of galleries, so it is nearly inert -- the
    # load-bearing fix is `dup_dt_seconds` itself, not this. Kept because those 1.2% are real
    # (they average 96.5% tied timestamps) and because it is the correct thing to do.
    dup_time_cluster_gate: bool = False


@dataclass
class ModelConfig:
    d_model: int = 256
    n_isab: int = 3
    n_heads: int = 4
    n_induced: int = 16          # induced points in ISAB
    head_hidden: int = 256
    # Stop the critic gradient at the shared encoder. The critic contributes ~99.7% of the
    # encoder's gradient, and the trained trunk preserves cluster identity WORSE than a random
    # init (1-NN precision 0.299 vs 0.563) -- see docs/representation-probe.md.
    detach_critic: bool = False
    # Feed the assign head two raw per-candidate scalars (cosine to the open page's mean CLIP,
    # and same-cluster fraction) that bypass the encoder entirely.
    pairwise_features: bool = False
    # Residual skip from the per-photo input projection to h, so the Set-Transformer's
    # contextualization does not erase per-photo identity (docs/representation-probe.md).
    input_skip: bool = False
    # LayerNorm each projected modality before concatenation, so no block dominates by raw
    # magnitude. Measured at init: 1-NN cluster precision 0.598 -> ~0.8 (raw CLIP is 0.835).
    input_block_norm: bool = False
    # Give the ACTOR the album-level state the global terms are functions of. Without this the
    # assign/close/end heads see only [per-photo h + status, mean of the OPEN page, a cached
    # state-INDEPENDENT gallery embedding] (+2 optional pairwise scalars), and `progress`
    # (inc_frac / pages_frac / open_frac) reaches the VALUE head only -- so the actor cannot see
    # the page count (`shape`), the open page's size (`pagesize`), which identities are already
    # covered (`coverage`), anything about CLOSED pages (`diversity`), or whether a near-duplicate
    # is already in the album (`dedup`). Those are exactly the terms that fail: v24 has `shape`
    # -0.051 and `dedup` -0.048 while every page-local term rises, and v19a's best-ever `coverage`
    # gain was +0.016 against a measured **+0.226** available to a state-dependent picker
    # (docs/research-log.md §1f). This is not a capacity limit -- the information is absent, so more
    # parameters cannot help.
    #
    # Adds 6 album-global scalars to assign/close/end and 2 per-candidate scalars to assign; see
    # AlbumNarratorEnv._album_features for the definitions.
    # ⚠️ Changes `assign_mlp`'s input width, so a checkpoint trained without it CANNOT be resumed
    # into it (`load_state_dict` size mismatch). Fresh runs only -- which §1e says is required
    # anyway, since new inputs are new information and a converged policy has no budget to use them.
    album_state_features: bool = False


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 4
    minibatch: int = 1024
    ent_coef: float = 0.03       # initial entropy bonus (annealed to ent_coef_final)
    ent_coef_final: float = 0.003

    # --- adaptive entropy (SAC-style automatic temperature, targeting a sharpness level) ---
    # Both a fixed (v6) and an annealed (v5, v8) coefficient failed the same way: the policy
    # sharpens past the point where it is still improving and held-out score falls. The mock
    # control reproduced it in both reward variants, so it is an optimization phenomenon.
    #
    # Instead of scheduling the coefficient, target the QUANTITY that goes wrong: the sharpness
    # gap ln(n_available) - H. Raw entropy is useless here (it correlates 0.996 with mask size),
    # which is why the target is expressed as a gap. Multiplicative control, clamped:
    #   gap above target -> raise ent_coef (push back toward uniform)
    #   gap below target -> lower ent_coef (allow the policy to commit)
    ent_adaptive: bool = False
    ent_target_sharpness: float = 0.40   # runs peaked around 0.25-0.40 before declining
    ent_adapt_rate: float = 0.05         # multiplicative step per update
    ent_coef_min: float = 0.001
    ent_coef_max: float = 0.30
    # --- auxiliary cluster-discrimination loss --------------------------------------
    # Measured: the encoder does not fail to ACQUIRE cluster structure, training REMOVES it.
    # With per-modality input normalization h starts at 1-NN cluster precision 0.783 (raw CLIP
    # 0.804) and 500 PPO updates drag it to 0.394 -- the same place it lands from a much worse
    # start, with or without the critic gradient attached. A supervised-contrastive term over
    # content clusters holds the representation in place while the policy trains on top.
    aux_cluster_coef: float = 0.0     # 0 disables
    aux_temperature: float = 0.1
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # --- group-relative advantages (H4) ----------------------------------------------
    # `_normalize_adv` pools every timestep of 8 DIFFERENT galleries into one mean/std, so
    # between-gallery difficulty dominates the advantage signal and within-gallery action
    # credit is squashed. With group_k > 1 the batch becomes (rollout_galleries / group_k)
    # distinct galleries rolled out group_k times each, and advantages are normalized WITHIN
    # each gallery's group -- so a rollout is judged only against other attempts at the same
    # gallery. Episode count is unchanged, so compute per update is unchanged; the cost is
    # fewer distinct galleries seen per update.
    group_k: int = 1             # 1 = historical batch-wide normalization
    rollout_galleries: int = 8   # galleries (episodes) collected per update
    updates: int = 500


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    seed: int = 0
    device: str = "cuda"         # falls back to cpu at use-site if unavailable


def _overlay(obj: Any, data: dict[str, Any]) -> None:
    """Recursively overlay a nested dict onto a dataclass instance in place."""
    valid = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in valid:
            raise KeyError(f"Unknown config key '{key}' for {type(obj).__name__}")
        cur = getattr(obj, key)
        if is_dataclass(cur) and isinstance(value, dict):
            _overlay(cur, value)
        elif key == "weights" and isinstance(value, dict):
            # merge reward weights rather than replace
            cur.update(value)
        else:
            setattr(obj, key, value)


def load_config(path: str | None = None) -> Config:
    cfg = Config()
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        _overlay(cfg, data)
    return cfg
