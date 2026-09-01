"""Tests for ``enrich.key_pages``.

The rule itself is `src/core/key_pages.py`, which ProcessStage has always used
and which this substage calls unchanged. What is worth guarding is the wiring:
that the substage runs over the whole gallery without disturbing it, that its
answer survives the message boundary, and that a gallery it cannot serve loses
its covers rather than its album.

    python -m pytest tests/test_key_pages.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import ENRICH, AlbumContext, Col, KeyPages  # noqa: E402
from src.pipeline.contracts import DesignSpec, GalleryFacts  # noqa: E402
from src.pipeline.enrich.key_pages import CLOSING, OPENING  # noqa: E402
from src.pipeline.registry import get  # noqa: E402

BRIDE, GROOM = 101, 202


def couple_gallery(n=24, orientation="landscape"):
    """A gallery whose couple photos span the day, earliest first.

    Half carry an opening-worthy subquery and half a closing-worthy one, so the
    priority ladder has something to find at both ends.
    """
    openers = ["bride and groom smiling at each other", "bride and groom posing for a portrait"]
    closers = ["bride and groom dancing", "bride and groom smiling at each other"]
    rows = []
    for i in range(n):
        subquery = (openers if i < n // 2 else closers)[i % 2]
        rows.append({
            Col.IMAGE_ID: 1000 + i,
            Col.IMAGE_ORDER: float(i % 7),
            Col.IMAGE_ORIENTATION: orientation,
            Col.IMAGE_TIME: 1_700_000_000 + i * 60,
            Col.PERSONS_IDS: [BRIDE, GROOM],
            Col.N_FACES: 2,
            Col.CLUSTER_CONTEXT: "bride and groom",
            Col.IMAGE_SUBQUERY_CONTENT: subquery,
            Col.BRIDE_ID: BRIDE,
            Col.GROOM_ID: GROOM,
            Col.EMBEDDING: np.linspace(i, i + 1, 8).astype(np.float32),
        })
    return pd.DataFrame(rows)


def run(photos, is_wedding=True, pages=None):
    context = AlbumContext(
        photos=photos,
        facts=GalleryFacts(is_wedding=is_wedding),
        designs=DesignSpec(pages=pages if pages is not None else {}),
    )
    return get("enrich.key_pages")()(context)


# -- composition -----------------------------------------------------------


def test_registered_in_enrich():
    assert "enrich.key_pages" in ENRICH
    assert get("enrich.key_pages").name == "enrich.key_pages"


def test_runs_after_the_enrichments_it_reads():
    """It needs `cluster_context`, the identities and the subquery tags, so it
    has to come after the substages that derive them."""
    order = list(ENRICH)
    for earlier in ("enrich.content_class", "enrich.identities", "enrich.semantic_tags"):
        assert order.index(earlier) < order.index("enrich.key_pages")


# -- the wedding path ------------------------------------------------------


def test_tags_one_opening_and_one_closing():
    context = run(couple_gallery())

    assert not context.failed
    assert len(context.key_pages.opening) == 1
    assert len(context.key_pages.closing) == 1
    assert context.key_pages.opening != context.key_pages.closing

    tagged = context.photos[context.photos[Col.KEY_PAGE] != ""]
    assert sorted(tagged[Col.KEY_PAGE]) == [CLOSING, OPENING]


def test_the_tag_matches_the_sidecar():
    context = run(couple_gallery())
    photos = context.photos

    opening = photos.loc[photos[Col.KEY_PAGE] == OPENING, Col.IMAGE_ID].tolist()
    closing = photos.loc[photos[Col.KEY_PAGE] == CLOSING, Col.IMAGE_ID].tolist()

    assert opening == context.key_pages.opening
    assert closing == context.key_pages.closing


def test_every_row_gets_the_column():
    """Empty string rather than NaN, so consumers can compare without a null
    check and the `provides` contract holds across the whole frame."""
    context = run(couple_gallery())
    assert Col.KEY_PAGE in context.photos.columns
    assert not context.photos[Col.KEY_PAGE].isna().any()


def test_the_gallery_is_left_intact():
    """`choose_good_wedding_images` returns the frame with the covers removed --
    that pruning is for ProcessStage's benefit and must not reach enrich, or
    selection would never see the two best couple photos."""
    photos = couple_gallery()
    before = list(photos[Col.IMAGE_ID])

    context = run(photos)

    assert list(context.photos[Col.IMAGE_ID]) == before


def test_portrait_only_gallery_still_gets_covers():
    """Landscape is a preference, not a requirement."""
    context = run(couple_gallery(orientation="portrait"))
    assert context.key_pages.opening and context.key_pages.closing


# -- gating and failure ----------------------------------------------------


def test_skipped_on_a_non_wedding_gallery():
    """The rule is built around the couple and reads `cluster_context`, which
    non-wedding galleries never get. ProcessStage keeps handling those."""
    context = run(couple_gallery(), is_wedding=False)

    assert context.key_pages is None
    assert Col.KEY_PAGE not in context.photos.columns
    assert context.diagnostics[-1].note == "skipped"


def test_skipped_when_the_album_has_no_first_page():
    context = run(couple_gallery(), pages={"lastPage": True})

    assert context.key_pages is None
    assert Col.KEY_PAGE not in context.photos.columns
    assert context.diagnostics[-1].note == "skipped"


def test_runs_when_there_is_no_design_data_at_all():
    """An offline run or a test has no `pagesInfo`; the covers are still worth
    computing there."""
    context = run(couple_gallery(), pages={})
    assert context.key_pages is not None


def test_a_gallery_it_cannot_serve_loses_the_covers_not_the_album():
    """Missing `cluster_context` on a wedding is a broken gallery, but a cover
    is not worth failing an album over -- the substage is optional."""
    photos = couple_gallery().drop(columns=[Col.CLUSTER_CONTEXT])

    context = run(photos)

    assert not context.failed
    assert not context.diagnostics[-1].ok


# -- the message boundary --------------------------------------------------


class FakeMessage:
    def __init__(self, content=None):
        self.content = content if content is not None else {"base_url": "x://y", "projectId": 1}
        self.pagesInfo = {}
        self.designsInfo = {}


def test_survives_a_round_trip_through_the_message():
    message = FakeMessage()
    context = AlbumContext.from_message(message)
    context.photos = couple_gallery()
    context.key_pages = KeyPages(opening=[7], closing=[9])
    context.sync_to_message()

    assert message.content["key_pages"] == {"opening": [7], "closing": [9]}

    rebuilt = AlbumContext.for_message(FakeMessage(dict(message.content)))

    assert rebuilt.key_pages.opening == [7]
    assert rebuilt.key_pages.closing == [9]


def test_absent_key_pages_stay_absent():
    """A message from before this substage existed must not grow an empty
    KeyPages that a consumer would read as 'no cover found'."""
    message = FakeMessage({"base_url": "x://y", "projectId": 1,
                           "gallery_photos_info": couple_gallery()})

    rebuilt = AlbumContext.for_message(message)

    assert rebuilt.key_pages is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
