"""A gallery that arrived incomplete must not be read as a whole one.

Project 53753700 reached `enrich.gallery_type` on 2026-09-18 with 61 rows of
which 8 carried content data. It was called a *wedding* -- 8/61 = 0.13, and
`> 0.6` was the only route to non-wedding -- and went down the wedding path with
8 photos, published 2, lost both to the covers, and crashed in the grouping.
Re-read once the gallery was complete, the same 85 photos answer non-wedding,
which is what its `projectCategory: 26` says.

The classifier now takes its share over the photos the content model has
answered for, the shortfall is said out loud, and a gallery too thin to fill the
design's smallest album is refused rather than composed from its fragment.
"""
import logging

import numpy as np
import pandas as pd
import pytest

from src.pipeline.contracts import AlbumContext, DesignSpec
from src.pipeline.registry import get
from utils.reading_tools import check_gallery_type

LOGGER = logging.getLogger('test_partial_ingest')


def frame(classes):
    """A gallery frame carrying only what these paths read."""
    return pd.DataFrame({
        'image_id': list(range(len(classes))),
        'image_class': classes,
        'ranking': [1.0] * len(classes),
        'image_order': list(range(len(classes))),
        'cluster_label': [0] * len(classes),
        'cluster_class': [0] * len(classes),
    })


# --------------------------------------------------------------- the classifier

def test_a_classified_non_wedding_still_reads_non_wedding():
    assert check_gallery_type(frame([-1] * 9 + [3])) is False


def test_a_classified_wedding_still_reads_wedding():
    assert check_gallery_type(frame([3] * 9 + [-1])) is True


def test_unknowns_do_not_vote():
    """The regression: 8 known rows of 61, all unclassified content.

    Counted over the frame this is 8/61 = 0.13 and the gallery is a 'wedding'.
    Counted over what is known it is 8/8 = 1.0, which is what it is.
    """
    classes = [-1] * 8 + [np.nan] * 53
    assert len(classes) == 61
    assert check_gallery_type(frame(classes)) is False


def test_all_unknown_is_left_as_a_wedding():
    """No evidence either way keeps the historical default rather than guessing."""
    assert check_gallery_type(frame([np.nan] * 20)) is True


def test_a_frame_without_the_column_is_left_as_a_wedding():
    assert check_gallery_type(pd.DataFrame({'image_id': [1, 2]})) is True


# ------------------------------------------------------- the floor on a fragment

def context_with(photos, min_pages=11, max_pages=43):
    return AlbumContext(
        photos=photos,
        logger=LOGGER,
        designs=DesignSpec(designs={'minPages': min_pages, 'maxPages': max_pages}),
    )


def test_a_gallery_that_can_fill_its_smallest_album_passes():
    # minPages 11, maxPages 43 -> min 17 spreads -> 34 photos needed.
    context = get('enrich.require_cluster_data')()(context_with(frame([3] * 40)))
    assert len(context.photos) == 40


def test_a_fragment_is_refused_rather_than_composed():
    """The substage records a terminal error; the runner then stops the read and
    `read_messages` answers the queue with it, so the album fails rather than
    being composed out of the part of the gallery that was ready."""
    context = get('enrich.require_cluster_data')()(
        context_with(frame([3] * 8 + [np.nan] * 53)))

    assert context.failed
    assert '8 photos with content data' in context.error
    assert 'smallest album needs 34' in context.error
    assert '53 of 61 photos had no content data' in context.error


def test_no_design_page_counts_means_no_floor():
    """Nothing to size the album against, so nothing to refuse it for."""
    context = AlbumContext(photos=frame([3] * 2), logger=LOGGER, designs=DesignSpec())
    context = get('enrich.require_cluster_data')()(context)
    assert len(context.photos) == 2
