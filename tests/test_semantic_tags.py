"""Tests for ``enrich.semantic_tags``, and mainly for how it fails.

A gallery with no embeddings should not happen in practice. When it does, the
failure has to name the cause rather than surface as a broken contract three
lines later.

    python -m pytest tests/test_semantic_tags.py -v
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import AlbumContext, Col  # noqa: E402
from src.pipeline.contracts import GalleryFacts  # noqa: E402
from src.pipeline.enrich.classification import _usable_embeddings  # noqa: E402
from src.pipeline.registry import get  # noqa: E402


def quiet():
    logger = logging.getLogger("semantic-tags-test")
    logger.addHandler(logging.NullHandler())
    return logger


def gallery(embeddings):
    return pd.DataFrame({
        Col.IMAGE_ID: list(range(1000, 1000 + len(embeddings))),
        Col.EMBEDDING: embeddings,
        Col.MODEL_VERSION: [2] * len(embeddings),
        Col.CLUSTER_CONTEXT: ['bride and groom'] * len(embeddings),
    })


def run(photos):
    context = AlbumContext(logger=quiet(), photos=photos,
                           facts=GalleryFacts(is_wedding=True))
    return get("enrich.semantic_tags")()(context)


# -- the failure that matters ----------------------------------------------


def test_a_gallery_with_no_embeddings_says_so():
    """`generate_query` drops every row it cannot tag and only creates its two
    columns `if results:`. With nothing to tag it empties the photo table *and*
    leaves the columns absent, so the failure used to surface as this substage
    breaking its own `provides` contract -- which says nothing about why.
    """
    context = run(gallery([None, None, None]))

    assert context.failed
    assert "no usable image embedding" in context.error.lower()
    assert "3 photos" in context.error
    assert "provide" not in context.error, (
        "the error should name the cause, not the broken contract")


def test_the_photo_table_is_not_emptied_on_the_way_out():
    """Failing is right; silently destroying the frame first is not."""
    photos = gallery([None, None, None])

    context = run(photos)

    assert len(context.photos) == 3


def test_empty_arrays_count_as_missing():
    context = run(gallery([np.array([]), np.array([])]))

    assert context.failed


def test_a_missing_embedding_column_is_the_same_failure():
    photos = gallery([None]).drop(columns=[Col.EMBEDDING])

    context = AlbumContext(logger=quiet(), photos=photos,
                           facts=GalleryFacts(is_wedding=True))
    context = get("enrich.semantic_tags")()(context)

    # The requirement check catches it first, which is also a clear failure.
    assert context.failed


# -- the counter -----------------------------------------------------------


def test_usable_embeddings_counts_only_real_vectors():
    """Of these five, only the two real vectors count: None and the empty array
    have nothing to project, and the string will not convert to floats."""
    photos = gallery([np.ones(4), None, np.array([]), np.zeros(4), "not a vector"])

    assert _usable_embeddings(photos) == 2


def test_usable_embeddings_on_an_empty_frame():
    assert _usable_embeddings(pd.DataFrame()) == 0
    assert _usable_embeddings(None) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
