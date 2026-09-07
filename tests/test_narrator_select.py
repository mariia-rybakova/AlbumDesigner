"""Tests for ``select.narrator``.

Split by what they need. The adapter and the gating are pure -- no torch, no
checkpoint -- and are the parts most likely to break silently, because a
mis-mapped column does not raise, it just feeds the policy a worse gallery.
The one test that runs the real policy is skipped when the weights are not on
disk, since they are gitignored (46 MB, retrained often).

    python -m pytest tests/test_narrator_select.py -v
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import AlbumContext, Col, build_select  # noqa: E402
from src.pipeline.contracts import AiHints, DesignSpec, GalleryFacts  # noqa: E402
from src.pipeline.registry import get  # noqa: E402
from src.pipeline.select import narrator as narrator_stage  # noqa: E402
from utils.configs import CONFIGS  # noqa: E402

CKPT = CONFIGS['narrator'].get('checkpoint', '')
needs_weights = pytest.mark.skipif(
    not os.path.exists(CKPT), reason=f"narrator checkpoint not present at {CKPT}")


@pytest.fixture(autouse=True)
def enabled():
    """The substage is off by default; every test here wants it on."""
    original = CONFIGS['narrator']
    CONFIGS['narrator'] = {**original, 'enabled': True}
    yield
    CONFIGS['narrator'] = original


def gallery_frame(n=60, dim=768, model_version=2, **overrides):
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    frame = pd.DataFrame({
        Col.IMAGE_ID: [700000 + i for i in range(n)],
        Col.EMBEDDING: list(vectors),
        Col.MODEL_VERSION: model_version,
        Col.GENERAL_TIME: [i * 60 for i in range(n)],
        Col.IMAGE_TIME: [1_700_000_000 + i * 60 for i in range(n)],
        Col.IMAGE_COLOR: 1,
        Col.PERSONS_IDS: [[1, 2] for _ in range(n)],
        Col.RANKING: rng.uniform(0, 1, size=n),
        'composition_score': rng.uniform(0, 1, size=n),
        Col.IMAGE_AS: 1.5,
        Col.CLUSTER_LABEL: rng.integers(0, 5, size=n),
        Col.SCENE_ORDER: [float(i // 8) for i in range(n)],
        Col.DIAMETER: 0.3,
        Col.BACKGROUND_CENTROID: [None] * n,
        Col.FACES_INFO: [[] for _ in range(n)],
        Col.BODIES_INFO: [[] for _ in range(n)],
        'number_bodies': 1,
    })
    for column, value in overrides.items():
        frame[column] = value
    return frame


def context(photos=None, is_wedding=False, present=True, pages=None):
    return AlbumContext(
        photos=gallery_frame() if photos is None else photos,
        request={'projectId': 4242},
        facts=GalleryFacts(is_wedding=is_wedding, model_version=2),
        designs=DesignSpec(pages={'firstPage': True} if pages is None else pages),
        hints=AiHints(present=present),
    )


def applies(ctx) -> bool:
    """`applies_to` needs a `selection`, which `select.route` creates."""
    get('select.route')()(ctx)
    return get('select.narrator')().applies_to(ctx)


# -- composition -----------------------------------------------------------


def test_registered_and_placed_before_pick():
    """It can settle a gallery outright, so `pick` must see the result."""
    from src.pipeline import SELECT

    order = list(SELECT)
    assert 'select.narrator' in order
    assert order.index('select.narrator') < order.index('select.pick')


def test_the_select_pipeline_still_satisfies_its_contracts():
    assert build_select().unsatisfied() == []


# -- gating ----------------------------------------------------------------


def test_serves_a_non_wedding_v2_gallery():
    assert applies(context()) is True


def test_declines_a_wedding():
    """Weddings have a budgeted, category-aware path of their own."""
    assert applies(context(is_wedding=True)) is False


def test_declines_a_manual_request():
    """A null `photoIds` means the user assembled the album; nothing to compose."""
    assert applies(context(present=False)) is False


def test_declines_a_v1_gallery():
    """The policy's input width is fixed at training time: 768-d. A v1 gallery
    carries 512-d vectors, and serving it badly is worse than not serving it."""
    assert applies(context(gallery_frame(dim=512, model_version=1))) is False


def test_declines_when_disabled():
    CONFIGS['narrator'] = {**CONFIGS['narrator'], 'enabled': False}
    assert applies(context()) is False


def test_declines_a_gallery_too_small_to_compose():
    CONFIGS['narrator'] = {**CONFIGS['narrator'], 'min_photos': 40}
    assert applies(context(gallery_frame(n=20))) is False


def test_declines_an_empty_frame():
    assert applies(context(gallery_frame(n=0))) is False


# -- the adapter -----------------------------------------------------------


def build_gallery(photos=None, ctx=None):
    from src.narrator.config import Config

    cfg = Config()
    ctx = ctx or context(photos)
    return narrator_stage._gallery_from_frame(ctx.photos, ctx, cfg), cfg


def test_every_photo_crosses_the_boundary():
    photos = gallery_frame(n=25)
    gallery, _ = build_gallery(photos)

    assert len(gallery.photos) == 25
    assert [p.photo_id for p in gallery.photos] == list(photos[Col.IMAGE_ID])


def test_the_clip_vector_is_the_embedding_column():
    photos = gallery_frame(n=5)
    gallery, cfg = build_gallery(photos)

    for i, entry in enumerate(gallery.photos):
        assert entry.clip.shape == (cfg.env.clip_dim,)
        assert np.allclose(entry.clip, photos[Col.EMBEDDING].iloc[i])


def test_greyscale_comes_from_colorenum_zero():
    """`image_color` is `bg_segmentation.colorEnum`: 0 grayscale, 1 colour."""
    photos = gallery_frame(n=4)
    photos.loc[photos.index[:2], Col.IMAGE_COLOR] = 0

    gallery, _ = build_gallery(photos)

    assert [p.is_bw for p in gallery.photos] == [True, True, False, False]


def test_selection_score_is_ranking_not_image_order():
    """`ranking` holds `photo.selectionScore`; `image_order` is selectionOrder,
    a rank, and a different quantity entirely. Feeding the rank in as a score
    would invert the signal for every photo."""
    photos = gallery_frame(n=3)
    photos[Col.RANKING] = [0.25, 0.5, 0.75]
    photos[Col.IMAGE_ORDER] = [900.0, 2.0, 40.0]

    gallery, _ = build_gallery(photos)

    assert [p.selection_score for p in gallery.photos] == [0.25, 0.5, 0.75]


def test_an_out_of_range_score_becomes_neutral_not_a_giant_feature():
    """`flatnessScore` is 1e6 across the real corpus -- a "not computed"
    sentinel. Passed through it dwarfs every other feature by six orders of
    magnitude and LayerNorm erases the rest."""
    photos = gallery_frame(n=2)
    photos[Col.RANKING] = [1e6, -3.0]

    gallery, _ = build_gallery(photos)

    assert [p.selection_score for p in gallery.photos] == [0.0, 0.0]
    assert all(p.flatness_score == 0.5 for p in gallery.photos)


def test_the_cast_is_the_identity_set():
    photos = gallery_frame(n=3)
    photos[Col.PERSONS_IDS] = [[1, 2], [], [7]]

    gallery, _ = build_gallery(photos)

    assert [set(p.cast) for p in gallery.photos] == [{1, 2}, set(), {7}]


def test_a_missing_column_does_not_raise():
    """Non-wedding galleries skip the wedding enrichments, so the adapter must
    tolerate an absent column rather than assume the wedding frame's shape."""
    photos = gallery_frame(n=6).drop(columns=[Col.FACES_INFO, Col.BODIES_INFO,
                                              Col.SCENE_ORDER, Col.DIAMETER])

    gallery, _ = build_gallery(photos)

    assert len(gallery.photos) == 6
    assert all(p.face_emb is None and p.body_emb is None for p in gallery.photos)


def test_detection_embeddings_are_pooled_and_width_checked():
    """Vectors arrive as raw bytes on the proto; a wrong-width one is a
    different model's output and is dropped rather than reshaped."""
    from src.narrator.config import Config

    good = np.arange(512, dtype=np.float32)
    face = types.SimpleNamespace(embedding=good.tobytes())
    wrong = types.SimpleNamespace(embedding=np.arange(7, dtype=np.float32).tobytes())

    pooled = narrator_stage._pooled([face, face], Config().env.face_dim)
    assert pooled is not None and np.allclose(pooled, good)
    assert narrator_stage._pooled([wrong], Config().env.face_dim) is None
    assert narrator_stage._pooled([], Config().env.face_dim) is None
    assert narrator_stage._pooled(None, Config().env.face_dim) is None


def test_the_time_axis_prefers_general_time():
    """`general_time` is the rebuilt monotonic day when the EXIF is unusable,
    and the policy's temporal terms measure gaps along it."""
    photos = gallery_frame(n=3)
    photos[Col.GENERAL_TIME] = [0, 60, 120]
    photos[Col.IMAGE_TIME] = [999, 999, 999]

    gallery, _ = build_gallery(photos)

    assert [p.date_taken for p in gallery.photos] == [0.0, 60.0, 120.0]


def test_attribute_axes_are_neutral_without_the_file():
    photos = gallery_frame(n=4)
    original = CONFIGS['narrator']
    CONFIGS['narrator'] = {**original, 'attribute_axes': 'does/not/exist.npz'}
    try:
        gallery, _ = build_gallery(photos)
    finally:
        CONFIGS['narrator'] = original

    assert all(p.candid_score == 0.5 and p.indoor_score == 0.5
               and p.lighting == 0.5 and p.bgcolor == 0.5 for p in gallery.photos)


def test_the_shape_spec_keeps_the_trained_range():
    """The page bounds are hard masking bounds the policy composed under, not a
    preference to be replaced with the product's own page limits."""
    from src.narrator.config import Config
    from src.narrator.data.schema import ShapeSpec

    trained = ShapeSpec.from_env(Config().env)
    gallery, _ = build_gallery()

    assert gallery.shape.min_pages == trained.min_pages
    assert gallery.shape.max_pages == trained.max_pages


def test_a_tighter_design_caps_the_page_count():
    from src.narrator.config import Config

    ctx = context(pages={'firstPage': True, 'maxPages': 6})
    gallery = narrator_stage._gallery_from_frame(ctx.photos, ctx, Config())

    assert gallery.shape.max_pages == 6
    assert gallery.shape.min_pages <= 6


# -- declining safely ------------------------------------------------------


def test_a_missing_checkpoint_leaves_the_gallery_to_the_old_selection():
    """Declining must cost the composition, never the album: `pick` still runs
    and `predefined` stays unset."""
    narrator_stage._POLICY = None
    CONFIGS['narrator'] = {**CONFIGS['narrator'], 'checkpoint': 'no/such/policy.pt'}

    ctx = build_select().run(context())

    assert ctx.predefined is None
    assert get('select.pick')().applies_to(ctx) is True


def test_pick_stands_down_once_something_composed_a_grouping():
    ctx = context()
    get('select.route')()(ctx)
    assert get('select.pick')().applies_to(ctx) is True

    ctx.predefined = object()

    assert get('select.pick')().applies_to(ctx) is False


# -- the real policy -------------------------------------------------------


@needs_weights
def test_the_policy_composes_selection_and_grouping_together():
    narrator_stage._POLICY = None
    photos = gallery_frame(n=60)

    ctx = build_select().run(context(photos))

    assert not ctx.failed
    assert ctx.predefined is not None, "the policy did not compose"
    spreads = ctx.predefined.spreads
    assert spreads, "composed an album with no spreads"

    gallery_ids = set(photos[Col.IMAGE_ID])
    picked = [pid for s in spreads for pid in s.photo_ids]
    assert set(picked) <= gallery_ids, "composed a photo the gallery does not have"
    assert len(picked) == len(set(picked)), "the same photo is on two spreads"
    assert set(ctx.selection.photo_ids) == set(ctx.predefined.all_photo_ids())


@needs_weights
def test_the_covers_are_not_also_spreads():
    """A hero emitted as a spread opens the album on a one-photo spread."""
    narrator_stage._POLICY = None
    ctx = build_select().run(context(gallery_frame(n=60)))

    body = {pid for s in ctx.predefined.spreads for pid in s.photo_ids}
    covers = set(ctx.predefined.first_page_photo_ids or []) | \
             set(ctx.predefined.last_page_photo_ids or [])

    assert covers, "no covers were lifted out"
    assert not (covers & body)


@needs_weights
def test_the_grouping_reaches_the_message_processstage_reads():
    """`content['predefined_layout']` is the key ProcessStage takes the
    predefined-layout path on -- the same one an external narrator service
    reaches through the request's `predefinedLayout` block."""
    narrator_stage._POLICY = None
    ctx = build_select().run(context(gallery_frame(n=60)))
    # `sync_to_message` writes onto the message the context was built from.
    ctx.message = types.SimpleNamespace(content={}, pagesInfo={}, designsInfo={})

    message = ctx.sync_to_message()

    assert 'predefined_layout' in message.content
    assert message.content['predefined_layout'] is ctx.predefined


@needs_weights
def test_page_sizes_respect_the_policys_own_bounds():
    from src.narrator.config import Config

    narrator_stage._POLICY = None
    env = Config().env
    ctx = build_select().run(context(gallery_frame(n=80)))

    sizes = [len(s.photo_ids) for s in ctx.predefined.spreads]
    assert all(env.min_photos_per_page <= n <= env.max_photos_per_page for n in sizes), sizes
