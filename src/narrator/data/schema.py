"""Per-photo and per-gallery data model, plus packing to numpy tensors.

The field set mirrors the real pic-time protobufs (joined by ``photo_id``):
  * ``clip``        -- fine-tuned V2 CLIP image embedding (768-d, L2-normed)
  * ``date_taken``  -- bg_segmentation.Photo.dateTaken (epoch seconds; NaN if missing)
  * ``is_bw``       -- bg_segmentation.Photo.colorEnum == grayscale
  * ``cast``        -- set of personInfo identityNumeralId present in the photo
  * ``face_emb`` / ``body_emb`` -- per-photo pooled raw detection embeddings
  * attribute scores (candid/indoor/lighting/bgcolor) from CLIP text-concept axes
  * precomputed aesthetics/scene signals (selection/composition/flatness, cluster/scene)

``Gallery`` holds a list of ``Photo`` plus a ``ShapeSpec`` (album soft ranges).
``pack_gallery`` converts a Gallery into a ``PackedGallery`` of dense numpy arrays,
which is what the environment and reward operate on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import EnvConfig


@dataclass
class Photo:
    photo_id: int
    clip: np.ndarray                    # (clip_dim,) float32, L2-normed
    date_taken: float = float("nan")    # epoch seconds, UTC; NaN if missing
    is_bw: bool = False                 # colorEnum == grayscale
    cast: frozenset[int] = frozenset()  # personInfo identity ids present

    # Attribute axis scores in [0, 1] (candid: 1=candid; indoor: 1=indoor).
    candid_score: float = 0.5
    indoor_score: float = 0.5
    lighting: float = 0.5               # e.g. bright(1) .. dark(0)
    bgcolor: float = 0.5                # colorful(1) .. neutral(0)

    # Pooled raw detection embeddings (mean over detections); zeros if none.
    face_emb: np.ndarray | None = None  # (face_dim,)
    body_emb: np.ndarray | None = None  # (body_dim,)

    # Precomputed aesthetics / scene / selection signals.
    selection_score: float = 0.0
    composition_score: float = 0.0
    flatness_score: float = 0.0
    aspect_ratio: float = 1.0
    blob_centroid: tuple[float, float] = (0.5, 0.5)
    blob_diameter: float = 0.0
    cluster_id: int = -1                # content_cluster.clusterId (-1 if unknown)
    scene_order: float = float("nan")   # bg.sceneOrder (NaN if unknown)


@dataclass
class ShapeSpec:
    """Album shape soft ranges for a gallery (falls back to EnvConfig defaults)."""
    min_photos_per_page: int
    max_photos_per_page: int
    min_pages: int
    max_pages: int
    target_photos_per_page: float
    target_pages: float
    # Gaussian widths for shape_preference, as a fraction of each target. The default 0.4
    # gives sigma = 5.6 pages against a legal range of only +/-3, so the term is nearly flat
    # across every reachable album (measured corr(shape, album_score) = +0.06). Lower it to
    # make album shape an actual preference. See docs/reward-audit.md.
    pages_sigma_frac: float = 0.4
    pp_sigma_frac: float = 0.4
    formal_page_min: int = 2
    formal_page_max: int = 3
    candid_page_min: int = 3
    candid_page_max: int = 4

    @classmethod
    def from_env(cls, cfg: EnvConfig) -> "ShapeSpec":
        return cls(
            min_photos_per_page=cfg.min_photos_per_page,
            max_photos_per_page=cfg.max_photos_per_page,
            min_pages=cfg.min_pages,
            max_pages=cfg.max_pages,
            target_photos_per_page=cfg.target_photos_per_page,
            target_pages=cfg.target_pages,
            pages_sigma_frac=cfg.shape_pages_sigma_frac,
            pp_sigma_frac=cfg.shape_pp_sigma_frac,
            formal_page_min=cfg.formal_page_min, formal_page_max=cfg.formal_page_max,
            candid_page_min=cfg.candid_page_min, candid_page_max=cfg.candid_page_max,
        )


@dataclass
class Gallery:
    gallery_id: str
    photos: list[Photo]
    shape: ShapeSpec | None = None

    def __len__(self) -> int:
        return len(self.photos)


@dataclass
class PackedGallery:
    """Dense numpy view of a gallery. Row i corresponds to photos[i]."""
    gallery_id: str
    photo_ids: np.ndarray               # (N,) int64
    clip: np.ndarray                    # (N, clip_dim) float32 (L2-normed)
    time_norm: np.ndarray               # (N,) float32 in [0,1]; 0 where date missing
    time_valid: np.ndarray              # (N,) bool
    bw: np.ndarray                      # (N,) uint8 (1 = grayscale)
    style_class: np.ndarray             # (N,) int8 (0 = candid, 1 = formal)
    candid: np.ndarray                  # (N,) float32
    indoor: np.ndarray                  # (N,) float32
    lighting: np.ndarray                # (N,) float32
    bgcolor: np.ndarray                 # (N,) float32
    face_emb: np.ndarray                # (N, face_dim) float32 (L2-normed; 0 if none)
    body_emb: np.ndarray                # (N, body_dim) float32 (L2-normed; 0 if none)
    face_valid: np.ndarray              # (N,) bool
    body_valid: np.ndarray              # (N,) bool
    cast: list[frozenset[int]]          # per-photo identity id sets
    cast_multihot: np.ndarray           # (N, P) uint8
    person_ids: np.ndarray              # (P,) int64  (column j -> identity id)
    selection: np.ndarray               # (N,) float32
    composition: np.ndarray             # (N,) float32
    flatness: np.ndarray                # (N,) float32
    aspect: np.ndarray                  # (N,) float32
    blob_centroid: np.ndarray           # (N, 2) float32
    blob_diameter: np.ndarray           # (N,) float32
    cluster_id: np.ndarray              # (N,) int64
    scene_order: np.ndarray             # (N,) float32 (NaN where unknown)
    shape: ShapeSpec
    # SCENES: content clusters grouped by appearance, so one scene = one place/event of the session.
    # -1 for photos with no content cluster. `scene_order` looks like it should provide this and does
    # not -- measured over 40 real galleries it holds exactly ONE distinct value in every one of them,
    # so it carries no information at all. Derived here instead (see `_derive_scenes`).
    scene_id: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    # Seconds spanned by the gallery's valid timestamps (>=1.0), i.e. the divisor `time_norm` was
    # built with. Kept so a criterion needing an ABSOLUTE time window can recover one:
    # dt_seconds = dt_norm * time_span. Measured p10/p50/p90 span = 20 min / 1 h / 357 days, so
    # any threshold expressed as a FRACTION of it means wildly different things per gallery.
    time_span: float = 1.0

    @property
    def n(self) -> int:
        return self.photo_ids.shape[0]

    @property
    def n_persons(self) -> int:
        return self.person_ids.shape[0]


def _l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-8, None)


def _derive_scenes(clip: np.ndarray, cluster_id: np.ndarray, cut: float) -> np.ndarray:
    """Group content clusters into SCENES by the appearance of their centroids.

    A content cluster is one moment (similar *and* temporally close photos); a scene is one
    place/event, which usually spans several clusters -- the median real gallery has 30.5 clusters.
    Average-linkage agglomerative clustering over the L2-normalised cluster centroids, cut at
    ``cut`` cosine distance. Average linkage rather than single (which chains distinct locations
    through one ambiguous cluster) or complete (which splits a location whenever one cluster in it
    is framed differently).

    Returns (N,) int64 scene labels, -1 where ``cluster_id`` is negative (5.8% of the real corpus):
    an unclustered photo must not become a phantom scene that the album is then required to cover.
    """
    n = len(cluster_id)
    scene = np.full(n, -1, dtype=np.int64)
    uniq = np.unique(cluster_id[cluster_id >= 0])
    if len(uniq) == 0:
        return scene
    if len(uniq) == 1:
        scene[cluster_id == uniq[0]] = 0
        return scene
    cent = np.stack([clip[cluster_id == c].mean(0) for c in uniq])
    cent /= np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-8)
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    lab = fcluster(linkage(pdist(cent, metric="cosine"), method="average"), t=cut,
                   criterion="distance")
    for c, sid in zip(uniq, lab):
        scene[cluster_id == c] = int(sid) - 1        # fcluster labels are 1-based
    return scene


def pack_gallery(gallery: Gallery, cfg: EnvConfig) -> PackedGallery:
    photos = gallery.photos
    n = len(photos)
    if n == 0:
        raise ValueError(f"gallery {gallery.gallery_id} has no photos")

    clip = _l2norm(np.stack([p.clip.astype(np.float32) for p in photos]))

    times = np.array([p.date_taken for p in photos], dtype=np.float64)
    time_valid = np.isfinite(times)
    if time_valid.any():
        lo = np.nanmin(times[time_valid])
        hi = np.nanmax(times[time_valid])
        span = max(hi - lo, 1.0)
        time_norm = np.where(time_valid, (times - lo) / span, 0.0).astype(np.float32)
    else:
        span = 1.0
        time_norm = np.zeros(n, dtype=np.float32)

    bw = np.array([1 if p.is_bw else 0 for p in photos], dtype=np.uint8)
    candid = np.array([p.candid_score for p in photos], dtype=np.float32)
    style_class = (candid < cfg.candid_threshold).astype(np.int8)  # 1 = formal
    indoor = np.array([p.indoor_score for p in photos], dtype=np.float32)
    lighting = np.array([p.lighting for p in photos], dtype=np.float32)
    bgcolor = np.array([p.bgcolor for p in photos], dtype=np.float32)

    def pack_emb(attr: str, dim: int):
        embs = np.zeros((n, dim), dtype=np.float32)
        valid = np.zeros(n, dtype=bool)
        for i, p in enumerate(photos):
            e = getattr(p, attr)
            if e is not None:
                embs[i] = e.astype(np.float32)
                valid[i] = True
        return _l2norm(embs), valid

    face_emb, face_valid = pack_emb("face_emb", cfg.face_dim)
    body_emb, body_valid = pack_emb("body_emb", cfg.body_dim)

    cast = [frozenset(p.cast) for p in photos]
    all_ids = sorted({pid for c in cast for pid in c})
    id_to_col = {pid: j for j, pid in enumerate(all_ids)}
    person_ids = np.array(all_ids, dtype=np.int64)
    cast_multihot = np.zeros((n, max(len(all_ids), 1)), dtype=np.uint8)
    for i, c in enumerate(cast):
        for pid in c:
            cast_multihot[i, id_to_col[pid]] = 1
    if not all_ids:
        cast_multihot = np.zeros((n, 0), dtype=np.uint8)

    selection = np.array([p.selection_score for p in photos], dtype=np.float32)
    composition = np.array([p.composition_score for p in photos], dtype=np.float32)
    flatness = np.array([p.flatness_score for p in photos], dtype=np.float32)
    aspect = np.array([p.aspect_ratio for p in photos], dtype=np.float32)
    blob_centroid = np.array([p.blob_centroid for p in photos], dtype=np.float32)
    blob_diameter = np.array([p.blob_diameter for p in photos], dtype=np.float32)
    cluster_id = np.array([p.cluster_id for p in photos], dtype=np.int64)
    scene_order = np.array([p.scene_order for p in photos], dtype=np.float32)

    # ALWAYS rebuild from the LIVE EnvConfig; never trust `gallery.shape`.
    #
    # ShapeSpec mixes two different kinds of thing: album-shape bounds (min/max pages, targets) AND
    # the reward's Gaussian widths `pages_sigma_frac` / `pp_sigma_frac`, which are reward *tuning
    # knobs*, not properties of a gallery. RealIngest stamps a ShapeSpec onto every Gallery at INGEST
    # time and pack_dataset pickles it, so the widths inside the shards are whatever the config said
    # when the corpus was packed. Reading them back (the old `gallery.shape or ...`) silently
    # overrode every later config: v9-v27 all set `shape_pages_sigma_frac: 0.15` (sigma 2.1 pages)
    # and all actually ran at the 0.4 default (sigma 5.6) -- wide enough that `shape` was nearly flat
    # across every legal album, which is precisely the failure ShapeSpec's own docstring predicts.
    # Measured cost: it cut the page-count incentive by ~4x (11.7 -> 14 pages was worth +0.014 of
    # album_score instead of +0.05), which is part of why the policy settled at the minimum album.
    #
    # Invisible on mock data -- mock_generator leaves `shape=None`, so every test always saw the live
    # config and passed. It only ever manifested on real packed galleries.
    shape = ShapeSpec.from_env(cfg)

    return PackedGallery(
        gallery_id=gallery.gallery_id,
        photo_ids=np.array([p.photo_id for p in photos], dtype=np.int64),
        clip=clip, time_norm=time_norm, time_valid=time_valid,
        bw=bw, style_class=style_class, candid=candid, indoor=indoor,
        lighting=lighting, bgcolor=bgcolor,
        face_emb=face_emb, body_emb=body_emb, face_valid=face_valid, body_valid=body_valid,
        cast=cast, cast_multihot=cast_multihot, person_ids=person_ids,
        selection=selection, composition=composition, flatness=flatness,
        aspect=aspect, blob_centroid=blob_centroid, blob_diameter=blob_diameter,
        cluster_id=cluster_id, scene_order=scene_order, shape=shape,
        scene_id=_derive_scenes(clip, cluster_id, cfg.scene_cut),
        time_span=float(span),
    )
