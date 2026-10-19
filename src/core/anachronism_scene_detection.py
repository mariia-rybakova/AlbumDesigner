"""Finding the scene that breaks the album's clock.

A gallery sometimes carries a "highlight" scene -- a folder of shots gathered
from across the whole day. Its timestamps are real but its place in the story
is not, so left alone it drags the artificial timeline out of shape.
`detect_anachronism_scene` names it, and `time_processing` sets it aside.

The scene is found from six lightweight signals over the photo table, no
embeddings needed:

  class_entropy        Shannon entropy of the image_class distribution
  dominant_ratio       share held by the modal class (scored as 1 - ratio,
                       so a scene of many classes scores higher)
  class_coverage       unique classes here / unique classes in the gallery
  ranking_spread       IQR of ranking (a wide spread means mixed content)
  ranking_mean         mean ranking (a highlight is made of the good shots)
  ctx_entropy          Shannon entropy of cluster_context, when present

Each scene gets a composite score and the highest wins; ties are all returned.

Written as an offline experiment ("Experiment 2") whose detector was promoted
into the pipeline. The evaluation half -- ground-truth parsing and a TP/FP/FN
report -- was removed once its author left and nothing called it; the git
history has it if the experiment is ever re-run.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# Default signal weights  (order matches SIGNAL_NAMES below)
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_WEIGHTS = [
    1.5,   # class_entropy_norm   — balance signal, size-corrected
    1.0,   # 1 − dominant_ratio   — no one class dominates
    2.5,   # class_coverage       — strongest: borrows classes from all scenes
    1.0,   # class_density        — unique classes per image
    1.0,   # ranking_spread       — wide ranking range = curated from many moments
    0.5,   # ranking_mean         — supporting signal
    1.5,   # ctx_entropy_norm     — context diversity, size-corrected
]

SIGNAL_NAMES = [
    "class_entropy_norm",  # H / log2(k)                — balance, size-corrected
    "1-dominant_ratio",    # size-invariant              — no single class dominates
    "class_coverage",      # n_unique / gallery_total    — breadth across all gallery classes
    "class_density",       # n_unique / n_images         — class diversity per image
    "ranking_spread",      # IQR of ranking              — size-invariant
    "ranking_mean",        # mean ranking                — size-invariant
    "ctx_entropy_norm",    # H / log2(k) on ctx labels   — size-corrected
]


def _is_highlight_scene(scene_name: str) -> bool:
    """Check if scene name looks like a Highlights scene."""
    keywords = ['highlight', 'sneak', 'preview', 'teaser', 'favorite', 'best', 'slideshow', 'blog']
    return any(kw in scene_name.lower() for kw in keywords)

# ══════════════════════════════════════════════════════════════════════════════
# Signal computation  (all signals are inherently size-normalised)
# ══════════════════════════════════════════════════════════════════════════════
def _safe_entropy(series: pd.Series) -> float:
    counts = series.dropna().value_counts()
    if counts.sum() == 0:
        return 0.0
    return float(scipy_entropy(counts / counts.sum(), base=2))


def _normalised_entropy(series: pd.Series) -> tuple[float, int]:
    """
    Returns (H_norm, n_unique) where H_norm = H / log2(n_unique).

    Why: raw Shannon entropy grows with sample size — a 400-image scene
    accumulates more rare classes by volume alone, inflating H even when
    the proportional diversity is the same as a 40-image scene.
    Dividing by log2(n_unique) gives the fraction of the theoretical maximum
    entropy for that many categories: 0 = one class dominates, 1 = perfectly
    uniform.  This is fully size-independent.
    """
    counts   = series.dropna().value_counts()
    n_unique = len(counts)
    total    = counts.sum()
    if total == 0 or n_unique <= 1:
        return 0.0, n_unique
    raw_H  = float(scipy_entropy(counts / total, base=2))
    h_norm = raw_H / np.log2(n_unique)   # always in [0, 1]
    return float(h_norm), n_unique


def compute_scene_signals(df: pd.DataFrame,logger,
                          use_cluster_ctx: bool = True) -> pd.DataFrame:
    """
    Compute size-normalised signals for every scene.

    Size-bias corrections:
      class_entropy_norm  = H(image_class) / log2(n_unique_classes)
          fraction of max possible entropy; independent of scene size.
      class_coverage      = n_unique_in_scene / n_unique_in_gallery
          what fraction of the gallery's full class vocabulary appears here.
          The anachronism scene borrows images from every event segment, so it
          should cover nearly all classes seen anywhere in the gallery.
          Fully size-independent: a 50-image scene can cover 100% of classes
          just as well as a 400-image one.
      class_density       = n_unique / n_images
          unique classes per image; penalises large scenes that accumulate
          rare classes purely by volume.
      dominant_ratio, ranking_spread, ranking_mean — all already size-invariant.
      ctx_entropy_norm    = H(cluster_context) / log2(n_unique_ctx)
    """
    has_class       = "image_class"     in df.columns
    has_ranking     = "ranking"         in df.columns
    has_cluster_ctx = "cluster_class" in df.columns and use_cluster_ctx

    # Gallery-level totals used for coverage normalisation
    gallery_total_classes = df["image_class"].dropna().nunique() if has_class else 1
    gallery_total_classes = max(gallery_total_classes, 1)
    gallery_total_ctx     = df["cluster_class"].dropna().nunique() if has_cluster_ctx else 1
    gallery_total_ctx     = max(gallery_total_ctx, 1)

    rows = []
    for scene, sdf in df.groupby("scene_name"):
        n = len(sdf)

        # ── image_class signals ───────────────────────────────────────────────
        if has_class:
            ent_norm, n_unique = _normalised_entropy(sdf["image_class"])
            raw_H              = _safe_entropy(sdf["image_class"])
            counts             = sdf["image_class"].dropna().value_counts()
            dom_ratio          = (float(counts.iloc[0] / counts.sum())
                                  if len(counts) > 0 else 1.0)
            # coverage: what fraction of ALL gallery classes appear in this scene
            class_coverage = n_unique / gallery_total_classes
            # density: unique classes per image (size-penalised breadth)
            class_density  = n_unique / n
        else:
            raw_H = ent_norm = dom_ratio = class_coverage = class_density = 0.0
            n_unique = 0

        # ── ranking signals (already size-invariant) ──────────────────────────
        if has_ranking:
            r = sdf["ranking"].dropna()
            if len(r) >= 4:
                q1, q3         = np.percentile(r, [25, 75])
                ranking_spread = float(q3 - q1)
            elif len(r) > 1:
                ranking_spread = float(r.max() - r.min())
            else:
                ranking_spread = 0.0
            ranking_mean = float(r.mean()) if len(r) > 0 else 0.0
        else:
            ranking_spread = ranking_mean = 0.0

        # ── cluster_class entropy (size-corrected) ──────────────────────────
        if has_cluster_ctx:
            ctx_norm, n_ctx = _normalised_entropy(sdf["cluster_class"])
            ctx_raw         = _safe_entropy(sdf["cluster_class"])
            ctx_coverage    = n_ctx / gallery_total_ctx
        else:
            ctx_norm = ctx_raw = ctx_coverage = 0.0
            n_ctx = 0

        rows.append({
            # ── identifiers ──────────────────────────────────────────────────
            "scene_name":           scene,
            "n_images":             n,
            # ── size-corrected signals (used for scoring) ─────────────────────
            "class_entropy_norm":   ent_norm,        # H / log2(k)  ∈ [0,1]
            "dominant_ratio":       dom_ratio,        # size-invariant
            "class_coverage":       class_coverage,   # n_unique_scene / n_unique_gallery
            "class_density":        class_density,    # n_unique / n_images
            "ranking_spread":       ranking_spread,
            "ranking_mean":         ranking_mean,
            "ctx_entropy_norm":     ctx_norm,         # H / log2(k)  ∈ [0,1]
            # ── raw / debug values ────────────────────────────────────────────
            "class_entropy_raw":    raw_H,
            "n_unique_classes":     n_unique,
            "ctx_entropy_raw":      ctx_raw,
            "n_unique_ctx":         n_ctx,
            "ctx_coverage":         ctx_coverage,
        })

    return pd.DataFrame(rows)


def score_scenes(signals_df: pd.DataFrame,
                 weights: list[float],
                 use_cluster_ctx: bool = True) -> pd.DataFrame:
    """
    Min-max normalise each size-corrected signal across scenes within the
    gallery (so scores are relative to the gallery's own range), then sum
    the weighted contributions into a composite score.

    The min-max step here is about making the six signals comparable to each
    other in magnitude (they have different natural scales even after size
    correction).  The size correction in compute_scene_signals already
    removed the n_images bias before this step.
    """
    df = signals_df.copy()

    # Signal column → invert? (True means lower raw value = more candidate-like)
    signal_spec = [
        ("class_entropy_norm",  False),  # higher → more balanced diversity
        ("dominant_ratio",      True),   # lower  → no dominant class
        ("class_coverage",      False),  # higher → covers more of gallery's classes
        ("class_density",       False),  # higher → more unique classes per image
        ("ranking_spread",      False),  # higher → wider ranking range
        ("ranking_mean",        False),  # higher → better ranked images
        ("ctx_entropy_norm",    False),  # higher → more context diversity
    ]

    scored_cols = []
    for (col, inv), w in zip(signal_spec, weights):
        if col not in df.columns:
            continue
        if not use_cluster_ctx and col == "ctx_entropy_norm":
            continue

        vals = df[col].values.astype(float)
        if inv:
            vals = 1.0 - vals

        vmin, vmax = vals.min(), vals.max()
        norm = (vals - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(vals)

        col_s = f"_s_{col}"
        df[col_s] = norm * w
        scored_cols.append(col_s)

    df["composite_score"] = df[scored_cols].sum(axis=1)
    df = df.drop(columns=scored_cols)

    # ── Name-based boost for highlight-like scenes ───────────────────────────
    # Scenes with highlight-like names get a boost to help identify them when
    # signal scores are close. This improves accuracy without affecting the
    # relative ranking of non-highlight scenes.
    NAME_BOOST = 2.0
    df["name_boost"] = df["scene_name"].apply(
        lambda x: NAME_BOOST if _is_highlight_scene(x) else 0.0
    )
    df["composite_score"] = df["composite_score"] + df["name_boost"]

    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Detection logic
# ══════════════════════════════════════════════════════════════════════════════
def detect_anachronism(scored_df: pd.DataFrame) -> list[str]:
    """
    Return the scene(s) with the highest composite score.
    If two scenes are tied at the top score they are both returned —
    either matching GT counts as TP.
    """
    if scored_df.empty:
        return []
    top_score = scored_df["composite_score"].iloc[0]
    # Ties: keep all scenes within floating-point epsilon of the top
    tied = scored_df[np.isclose(scored_df["composite_score"], top_score, atol=1e-6)]
    return tied["scene_name"].tolist()


def detect_anachronism_scene(df,logger):
    weights = DEFAULT_WEIGHTS
    use_cluster_ctx=True
    # Compute signals
    signals  = compute_scene_signals(df,logger, use_cluster_ctx=use_cluster_ctx)
    if signals.empty:
      logger.info(f"[SKIP] No scenes found")
    else:
        # Score
        scored = score_scenes(signals, weights, use_cluster_ctx=use_cluster_ctx)
        detected = detect_anachronism(scored)
        if detected:
           logger.info(f"Detected the highlight or anachronismScene that disrupt our artificial time {detected[0]}")


