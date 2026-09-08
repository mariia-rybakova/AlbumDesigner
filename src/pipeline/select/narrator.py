"""Selection and page grouping by the albumNarrator policy.

Non-wedding galleries the user has not assembled themselves. The wedding path
budgets per category and picks with CP-SAT; a non-wedding gallery has no
category vocabulary to budget against, and `smart_non_wedding_selection` --
what runs today -- selects photos but says nothing about how they group.

The policy does both in one pass. It was trained as a sequential-construction
MDP over the gallery: interleaved ASSIGN / CLOSE_PAGE / END_ALBUM decisions
compose pages, then PLACE orders them. So a page comes out of the same
decision that put photos in it, which is the part `smart_non_wedding_selection`
cannot express.

Where the grouping goes
-----------------------
Into the predefined-spreads path, which already exists: one narrator page
becomes one designer spread. `sync_to_message` writes
``content['predefined_layout']`` and ProcessStage's existing hooks
(`build_first_last_pages`, `predefined_layout_processing`) lay the fixed
spreads out. No new layout code, and the same code path an external narrator
service would drive through the ``predefinedLayout`` request key.

Where the input comes from
--------------------------
Entirely from the frame that read + ingest + enrich already produced -- the two
projects parse the same protobufs, so every field the policy wants is a column
we already have. The vendored `data/ingest.py` (which reads per-gallery blobs
off disk) is deliberately *not* used; `_gallery_from_frame` below is the
adapter, and it is the only place the two schemas meet.

Two fields are not one-to-one:

``composition_score``
    ``bg_segmentation.compositionScore``, which `get_background_info` now reads.
``flatness_score``
    ``flatnessScore`` is ``1e6`` for every photo in the real corpus -- a "not
    computed" sentinel. The narrator's own ingest maps out-of-range values to
    neutral, so defaulting it here is the same value, not a loss.

Requires the 768-d V2 embedding: the policy's input width is fixed at training
time, and a ``model_version`` 1 gallery carries 512-d vectors. Those galleries
fall through to the existing selection rather than being served badly.
"""

from __future__ import annotations

import os
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.pipeline.contracts import AlbumContext, Col, photo
from src.pipeline.registry import register
from src.pipeline.substage import SubStage
from utils.configs import CONFIGS

#: The embedding width the policy was trained on. V1 galleries are 512-d.
V2_MODEL_VERSION = 2

#: colorEnum: 0 = grayscale, 1 = colour.
GRAYSCALE = 0

#: Loaded once per process -- 46 MB of weights and a torch import are far too
#: expensive per request. Guarded because the service reads messages on more
#: than one thread.
_POLICY: Optional[Tuple[object, object]] = None
_POLICY_LOCK = threading.Lock()


def settings() -> dict:
    return CONFIGS.get('narrator', {})


# -- the model -------------------------------------------------------------


def _load_policy(logger=None):
    """The policy and its config, loaded once and reused.

    ``weights_only=False`` is what the narrator's own loader uses -- the
    checkpoint carries the training `cfg` alongside the tensors, and the config
    is needed to rebuild the network before the weights can be loaded into it.
    It also means the checkpoint is executable content, so it must come from a
    location only we can write.
    """
    global _POLICY
    if _POLICY is not None:
        return _POLICY

    with _POLICY_LOCK:
        if _POLICY is not None:
            return _POLICY

        import torch  # deferred: only non-wedding requests pay the import

        from src.narrator.config import Config, _overlay
        from src.narrator.models.policy import ActorCritic

        path = settings().get('checkpoint')
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"narrator checkpoint not found: {path!r}")

        blob = torch.load(path, map_location='cpu', weights_only=False)
        cfg = Config()
        _overlay(cfg, blob['cfg'])

        # CPU, whatever the checkpoint says. `cfg` is the *training* config, so
        # `cfg.device` is 'cuda' -- the narrator's own `build_policy` (in
        # train/loop.py, deliberately not vendored) honours it and only falls
        # back when CUDA is absent. Here the fallback is the target: this is a
        # CPU inference path, and on a box that happens to have a GPU we still
        # do not want a 46 MB model migrating onto it per process.
        cfg.device = 'cpu'
        net = ActorCritic(cfg).to('cpu')
        net.load_state_dict(blob['state_dict'])
        net.eval()

        # Inference only, and on CPU: no autograd tape, and one thread per
        # request rather than torch's default of every core, because the
        # service already runs galleries concurrently and oversubscribing the
        # box makes every one of them slower.
        torch.set_grad_enabled(False)
        threads = int(settings().get('torch_threads', 1))
        if threads > 0:
            torch.set_num_threads(threads)

        if logger:
            logger.info(f"narrator: loaded {os.path.basename(path)} "
                        f"(clip_dim={cfg.env.clip_dim}, {threads} torch thread(s))")
        _POLICY = (net, cfg)
    return _POLICY


# -- the adapter -----------------------------------------------------------


def _pooled(detections, width: int) -> Optional[np.ndarray]:
    """Mean of a photo's detection embeddings, or None.

    Mirrors the narrator's own pooling: each detection carries its vector as
    raw bytes, wrong-width ones are dropped rather than reshaped, and a photo
    with no usable detection gets None (the schema's "no detections" value).
    """
    if detections is None:
        return None
    vectors = []
    for det in detections:
        raw = getattr(det, 'embedding', None)
        if not raw:
            continue
        vector = np.frombuffer(raw, dtype=np.float32)
        if vector.size == width:
            vectors.append(vector)
    if not vectors:
        return None
    return np.mean(vectors, axis=0).astype(np.float32)


def _axis_scores(embeddings: np.ndarray, axes_path: Optional[str]) -> Dict[str, np.ndarray]:
    """candid / indoor / lighting / bgcolor, projected from the embeddings.

    The narrator derives these from CLIP text-concept axes rather than any
    stored field, and min-max normalises each within the gallery. Without the
    axes file every score stays at the schema's neutral 0.5, which is what the
    narrator does when the file is absent.
    """
    if not axes_path or not os.path.exists(axes_path):
        return {}
    bundle = np.load(axes_path, allow_pickle=False)
    names = [str(n) for n in bundle['names'].tolist()]
    axes = bundle['axes'].astype(np.float32)

    unit = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8, None)
    projected = unit @ axes.T

    scores: Dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        column = projected[:, i]
        low, high = float(column.min()), float(column.max())
        scores[name] = (((column - low) / (high - low)).astype(np.float32)
                        if high > low else np.full(len(column), 0.5, dtype=np.float32))
    return scores


def _unit(value, default: float) -> float:
    """A score that is meant to be in [0, 1], or the neutral default.

    Out-of-range values are sentinels rather than measurements -- flatnessScore
    is 1e6 across the whole corpus -- and passing one through unscaled would
    dominate every other feature before the encoder's LayerNorm sees it.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        return default
    return number


def _centroid(value) -> Tuple[float, float]:
    """`background_centroid` as a plain (x, y); the frame holds a proto or None."""
    if value is None:
        return (0.5, 0.5)
    x, y = getattr(value, 'x', None), getattr(value, 'y', None)
    if x is None or y is None:
        try:
            x, y = value[0], value[1]
        except (TypeError, IndexError, KeyError):
            return (0.5, 0.5)
    return (float(x), float(y))


def _shape_spec(context: AlbumContext, cfg):
    """Album shape for this request: the trained ranges, capped by the design.

    The ranges are hard masking bounds -- the policy cannot end an album below
    ``min_pages`` or past ``max_pages`` -- so they are the shape it learned to
    compose for, not a preference to be overwritten with the product's own
    limits. v29 composes 10-13 spreads where a typical album is 15-26, and
    forcing it up to the product's floor would put it somewhere it has never
    been. So the trained range stands, and only a design that is *tighter* than
    it applies.
    """
    from src.narrator.data.schema import ShapeSpec

    spec = ShapeSpec.from_env(cfg.env)
    pages = context.designs.pages if context.designs else None
    design_max = (pages or {}).get('maxPages') if isinstance(pages, dict) else None
    if isinstance(design_max, int) and 0 < design_max < spec.max_pages:
        spec.max_pages = design_max
        spec.min_pages = min(spec.min_pages, design_max)
    return spec


def _gallery_from_frame(photos: pd.DataFrame, context: AlbumContext, cfg):
    """Build the narrator's `Gallery` out of the enriched frame.

    The only place the two schemas meet. Every field comes from a column that
    read + ingest + enrich already produced; nothing is fetched here.
    """
    from src.narrator.data.schema import Gallery, Photo

    frame = photos.reset_index(drop=True)
    embeddings = np.vstack(frame[Col.EMBEDDING].values).astype(np.float32)
    axes = _axis_scores(embeddings, settings().get('attribute_axes'))

    def axis(name: str, i: int) -> float:
        column = axes.get(name)
        return float(column[i]) if column is not None else 0.5

    has = frame.columns.__contains__
    entries: List[Photo] = []
    for i, row in frame.iterrows():
        cast = row[Col.PERSONS_IDS] if has(Col.PERSONS_IDS) else None
        entries.append(Photo(
            photo_id=int(row[Col.IMAGE_ID]),
            clip=embeddings[i],
            date_taken=_epoch(row, has),
            is_bw=bool(has(Col.IMAGE_COLOR) and row[Col.IMAGE_COLOR] == GRAYSCALE),
            cast=frozenset(int(p) for p in cast) if isinstance(cast, (list, tuple, set)) else frozenset(),
            candid_score=axis('candid', i), indoor_score=axis('indoor', i),
            lighting=axis('lighting', i), bgcolor=axis('bgcolor', i),
            face_emb=_pooled(row[Col.FACES_INFO] if has(Col.FACES_INFO) else None,
                             cfg.env.face_dim),
            body_emb=_pooled(row[Col.BODIES_INFO] if has(Col.BODIES_INFO) else None,
                             cfg.env.body_dim),
            # `ranking` IS selectionScore -- read_protos_files appends
            # photo.selectionScore into it. `image_order` is selectionOrder, a
            # rank, and is a different quantity.
            selection_score=_unit(row[Col.RANKING] if has(Col.RANKING) else None, 0.0),
            composition_score=_unit(row['composition_score'] if has('composition_score') else None, 0.0),
            flatness_score=0.5,
            aspect_ratio=float(row[Col.IMAGE_AS]) if has(Col.IMAGE_AS) and row[Col.IMAGE_AS] else 1.0,
            blob_centroid=_centroid(row[Col.BACKGROUND_CENTROID] if has(Col.BACKGROUND_CENTROID) else None),
            blob_diameter=float(row[Col.DIAMETER] or 0.0) if has(Col.DIAMETER) else 0.0,
            cluster_id=int(row[Col.CLUSTER_LABEL]) if has(Col.CLUSTER_LABEL)
                       and pd.notna(row[Col.CLUSTER_LABEL]) else -1,
            scene_order=float(row[Col.SCENE_ORDER]) if has(Col.SCENE_ORDER)
                        and pd.notna(row[Col.SCENE_ORDER]) else float('nan'),
        ))

    # `GalleryFacts` carries no project id; the request does.
    gallery_id = str((context.request or {}).get('projectId', 'gallery'))
    return Gallery(gallery_id=gallery_id, photos=entries, shape=_shape_spec(context, cfg))


def _epoch(row, has) -> float:
    """Seconds since the epoch, preferring the axis the rest of the pipeline trusts.

    `general_time` is the rebuilt monotonic day where the EXIF is unusable, and
    `image_time` the original. The narrator's temporal terms measure *gaps*, so
    an unusable axis is worse than a synthetic one.
    """
    for column in (Col.GENERAL_TIME, Col.IMAGE_TIME):
        if has(column) and pd.notna(row[column]):
            try:
                return float(row[column])
            except (TypeError, ValueError):
                continue
    return float('nan')


# -- the substage ----------------------------------------------------------


@register
class NarratorSubStage(SubStage):
    """Compose a non-wedding album with the policy, selection and grouping together.

    Optional, and gated hard. A gallery it cannot serve -- wrong embedding
    width, no checkpoint, a failed solve -- must fall through to the selection
    that exists rather than lose the album.
    """

    name = "select.narrator"
    requires = frozenset({
        photo(Col.IMAGE_ID),
        photo(Col.EMBEDDING),
        photo(Col.MODEL_VERSION),
    })
    provides = frozenset()
    optional = True

    def applies_to(self, context: AlbumContext) -> bool:
        if not settings().get('enabled', False):
            return False
        # Weddings have their own budgeted, category-aware path.
        if context.facts.is_wedding:
            return False
        # A manual request is already the user's album; nothing to compose.
        #
        # This is the boundary of "no user selection", and it is drawn at the
        # request shape rather than at intent, because nothing distinguishes
        # the two intents:
        #
        #   photoIds [1,2,3] -> AI, the user steered      -> served
        #   photoIds []      -> AI, the user chose nothing -> served
        #   photoIds null    -> manual                     -> NOT served
        #   no aiMetadata    -> manual                     -> NOT served
        #
        # A null `photoIds` means the user assembled the album by hand and
        # `content['photos']` holds their picks, so composing over it would
        # discard their work -- 52755795 is a real 42-page album of exactly
        # that kind. The cost of the rule is that a non-wedding request whose
        # producer sends null rather than [] gets no composition at all and its
        # whole gallery reaches layout unnarrowed. Decided deliberately: not
        # overriding a real album is worth more than serving a request shape
        # nothing produces yet, and Album Designer is reachable only from a
        # wedding gallery today, so no such traffic exists to measure.
        if context.selection is not None and context.selection.manual:
            return False
        photos = context.photos
        if photos is None or photos.empty:
            return False
        if Col.EMBEDDING not in photos.columns or Col.MODEL_VERSION not in photos.columns:
            return False
        if int(photos.iloc[0][Col.MODEL_VERSION]) != V2_MODEL_VERSION:
            return False
        minimum = int(settings().get('min_photos', 1))
        return len(photos) >= minimum

    def execute(self, context: AlbumContext) -> AlbumContext:
        logger = context.logger
        try:
            net, cfg = _load_policy(logger)
            gallery = _gallery_from_frame(context.photos, context, cfg)

            from src.narrator import export
            from src.narrator.env import hero

            sample_k = int(settings().get('sample_k', 1))
            packed, pages, score = export.compose(
                net, cfg, gallery, sample_k=sample_k,
                seed=int(settings().get('seed', 0)))
            first, last = (hero.pick_hero_photos(packed)
                           if cfg.reward.hero_pages else (-1, -1))
            block = export.album_to_predefined_block(packed, pages, first, last)
        except Exception as exc:  # noqa: BLE001 - fall through, never lose the album
            if logger:
                logger.warning(f"narrator: declined ({type(exc).__name__}: {exc}); "
                               "leaving the gallery to the existing selection")
            return context

        problems = export.validate_block(block, set(context.photos[Col.IMAGE_ID]))
        if problems:
            if logger:
                logger.warning(f"narrator: block rejected {problems[:3]}; "
                               "leaving the gallery to the existing selection")
            return context

        self._commit(context, block, score)
        return context

    @staticmethod
    def _commit(context: AlbumContext, block: dict, score: float) -> None:
        """Record the composition as both a selection and a fixed grouping."""
        from src.predefined.models import PredefinedLayoutInput

        predefined = PredefinedLayoutInput.from_request({'predefinedLayout': block})
        context.predefined = predefined
        context.selection.photo_ids = list(predefined.all_photo_ids())

        if context.logger:
            spreads = [len(s.photo_ids) for s in predefined.spreads]
            context.logger.info(
                f"narrator: {len(spreads)} spreads, {sum(spreads)} photos "
                f"(+{len(predefined.first_page_photo_ids or [])} first, "
                f"{len(predefined.last_page_photo_ids or [])} last cover), "
                f"per spread {spreads}, album_score {score:.4f}")
