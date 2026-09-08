"""Phase 0 of `docs/multi_album_plan.md` — pin the contamination.

Multiple albums per gallery means running selection more than once over one
message. That does not work today, and it does not fail loudly: the second
album is composed from the first album's *output*, and what comes out is a
plausible album that nobody would look at twice.

Measured on the synthetic gallery below, three passes over one message:

    pass 1: saw  60 photos -> selected 5
    pass 2: saw   5 photos -> selected 5
    pass 3: saw   5 photos -> selected 5

The gallery collapses to the first selection and stays there.

Three mechanisms cause it, and only the first is obvious:

1. `select.publish` narrows ``content['gallery_photos_info']`` to the chosen
   photos. That is **intended** for one album -- ProcessStage lays out the
   selected frame and reads it from exactly there -- which is why the fix
   cannot be "stop narrowing". It has to be that siblings do not share the
   frame in the first place.
2. `AlbumContext.for_message` reuses the context object cached on the message
   (`_MESSAGE_SLOT`), so pass 2 inherits pass 1's `facts`, `selection` and
   `predefined` rather than starting clean.
3. ProcessStage writes the *laid-out* frame back over the same key, so a later
   pass would inherit `time_cluster` / `group_sub_index` / `cropped_*` as if
   they were gallery state. Not exercised here (this module stops at SELECT),
   but the same shared key.

The three tests that state the goal are ``xfail(strict=True)``: they document
what Phase 1 must achieve, they keep the suite honest today, and the moment
Phase 1 lands they turn into XPASS -- which strict xfail reports as a failure,
so the markers cannot be forgotten. Remove them then; do not weaken them.

    python -m pytest tests/test_multi_album_isolation.py -v
"""

from __future__ import annotations

import logging
import os
import random
import sys
import types
from typing import List, NamedTuple

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import AlbumContext, Col, build_select  # noqa: E402
from utils.configs import CONFIGS  # noqa: E402

GALLERY_SIZE = 60

_QUIET = logging.getLogger("test_multi_album_isolation")
_QUIET.addHandler(logging.NullHandler())
_QUIET.setLevel(logging.CRITICAL)


@pytest.fixture(autouse=True)
def hermetic():
    """Keep the narrator out of it.

    A non-wedding gallery would otherwise be composed by the policy, which needs
    46 MB of weights that are gitignored -- so the test would pass or skip
    depending on the machine. With it off, selection takes the
    `smart_non_wedding_selection` path: no torch, no checkpoint, same
    contamination.
    """
    original = CONFIGS['narrator']
    CONFIGS['narrator'] = {**original, 'enabled': False}
    yield
    CONFIGS['narrator'] = original


def gallery_frame(n: int = GALLERY_SIZE) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(n, 768)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return pd.DataFrame({
        Col.IMAGE_ID: [800000 + i for i in range(n)],
        Col.EMBEDDING: list(vectors),
        Col.MODEL_VERSION: 2,
        Col.GENERAL_TIME: [i * 60 for i in range(n)],
        Col.IMAGE_TIME: [1_700_000_000 + i * 60 for i in range(n)],
        Col.IMAGE_COLOR: 1,
        Col.PERSONS_IDS: [[1, 2] for _ in range(n)],
        Col.RANKING: rng.uniform(0, 1, size=n),
        Col.IMAGE_ORDER: list(range(n)),
        Col.IMAGE_AS: 1.5,
        Col.CLUSTER_LABEL: rng.integers(0, 5, size=n),
        Col.SCENE_ORDER: [float(i // 8) for i in range(n)],
        Col.DIAMETER: 0.3,
        Col.BACKGROUND_CENTROID: [None] * n,
        Col.FACES_INFO: [[] for _ in range(n)],
        Col.BODIES_INFO: [[] for _ in range(n)],
        'number_bodies': 1,
    })


def fresh_message(frame: pd.DataFrame | None = None):
    """A message as ReadStage would hand one on, for a non-wedding AI request.

    `photoIds: []` rather than null: an AI request in which the user picked
    nothing, which is the case a composed album is for. Null would route as
    manual and skip selection entirely.
    """
    return types.SimpleNamespace(
        content={
            'gallery_photos_info': (gallery_frame() if frame is None else frame).copy(),
            'projectId': 1,
            'aiMetadata': {'photoIds': [], 'focus': [], 'personIds': [],
                           'subjects': [], 'density': 3},
            'photos': [],
            'is_wedding': False,
        },
        pagesInfo={}, designsInfo={}, error=None)


class Pass(NamedTuple):
    """What one selection pass saw and produced."""
    saw: int                 # photos in the frame handed to this pass
    selected: List[int]      # image ids it chose
    context: AlbumContext
    failed: bool


#: Seeded before every pass, because `smart_non_wedding_selection` is not
#: deterministic: `select_random_image` calls `random.choice` on Python's global
#: RNG, which `np.random.seed(42)` in main.py does not touch. Without this the
#: outcome comparison below would fail for two reasons at once -- contamination
#: and randomness -- and could not distinguish them. Seeding identically per
#: pass leaves the comparison sensitive to exactly one thing: whether the pass
#: saw the whole gallery.
PASS_SEED = 12345


def selection_passes(message, count: int) -> List[Pass]:
    """Run SELECT `count` times over one message, recording each pass.

    Deliberately the naive loop -- this is the thing the plan says does not
    work, so the test drives it exactly as a first implementation would.
    """
    passes: List[Pass] = []
    for _ in range(count):
        random.seed(PASS_SEED)
        np.random.seed(PASS_SEED)
        saw = len(message.content['gallery_photos_info'])
        context = AlbumContext.for_message(message, logger=_QUIET)
        context.facts.is_wedding = False
        result = build_select(logger=_QUIET).run(context)
        result.sync_to_message()
        chosen = list(result.selection.photo_ids) if result.selection else []
        passes.append(Pass(saw=saw, selected=chosen, context=result,
                           failed=result.failed))
    return passes


# -- the mechanism, pinned as fact ----------------------------------------
#
# These pass today and must keep passing: narrowing is correct for one album,
# and is exactly why siblings cannot share the frame.


def test_one_pass_selects_a_subset_of_the_gallery():
    """The premise. Without this the isolation tests below prove nothing."""
    single = selection_passes(fresh_message(), 1)[0]

    assert not single.failed
    assert single.saw == GALLERY_SIZE
    assert 0 < len(single.selected) < GALLERY_SIZE, (
        "selection must narrow, or there is nothing for a second pass to inherit")


def test_publish_narrows_the_message_frame_to_the_selection():
    """Intended behaviour, not the bug: ProcessStage lays out the selected
    frame and reads it from this key. Phase 1 must keep this true for a single
    album while stopping it from leaking between albums."""
    message = fresh_message()
    single = selection_passes(message, 1)[0]

    assert len(message.content['gallery_photos_info']) == len(single.selected)


def test_the_context_is_cached_on_the_message():
    """Also intended -- it is what lets a later stage reuse the read's context
    instead of rebuilding from `content`. It is the reason a second pass
    inherits `facts` and `selection`, so Phase 1 has to give each album its own
    context without breaking this for the single-album flow."""
    message = fresh_message()
    first = AlbumContext.for_message(message, logger=_QUIET)
    again = AlbumContext.for_message(message, logger=_QUIET)

    assert first is again


# -- what Phase 1 must achieve -------------------------------------------
#
# xfail(strict=True): expected to fail now, and reported as a failure the
# moment it starts passing, so the markers get removed deliberately.


@pytest.mark.xfail(strict=True, reason="Phase 0: siblings share "
                   "content['gallery_photos_info']; a second pass sees only "
                   "the first pass's selection (60 -> 5 -> 5)")
def test_every_pass_sees_the_whole_gallery():
    """The core isolation requirement: album 2 composes from the gallery, not
    from album 1."""
    passes = selection_passes(fresh_message(), 3)

    assert [p.saw for p in passes] == [GALLERY_SIZE] * 3, (
        f"each pass must see all {GALLERY_SIZE} photos, saw {[p.saw for p in passes]}")


@pytest.mark.xfail(strict=True, reason="Phase 0: AlbumContext.for_message "
                   "reuses the context cached on the message, so pass 2 "
                   "inherits pass 1's facts, selection and predefined")
def test_every_pass_gets_its_own_context():
    passes = selection_passes(fresh_message(), 3)

    contexts = [id(p.context) for p in passes]
    assert len(set(contexts)) == 3, "each album needs its own context object"


@pytest.mark.xfail(strict=True, reason="Phase 0: a later pass selects from the "
                   "earlier pass's output, so it cannot match what that album "
                   "would be on its own")
def test_a_later_pass_selects_what_that_album_would_select_alone():
    """The requirement stated as an outcome rather than as plumbing: with
    identical inputs, album 2 must be the album a single run produces. This is
    what Phase 2 scales up to N identical variants.

    Both sides run under the same seed (see `PASS_SEED`), so the only thing
    this can detect is the pass having seen a different gallery. Verified
    satisfiable: with a fresh message per album it holds, so it is a test of
    isolation and not of the RNG.
    """
    reference = selection_passes(fresh_message(), 1)[0]
    passes = selection_passes(fresh_message(), 2)

    assert passes[1].selected == reference.selected, (
        "album 2 differs from the same album run on its own")
