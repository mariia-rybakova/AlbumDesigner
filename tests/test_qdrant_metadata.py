"""The photographer's rating and keywords, read off the Qdrant payload.

The vector database carries one point per vectorised photo, and that point's
payload holds a star `rating`, a list of `keywords`, or neither. These tests
pin what the reader makes of each shape, and that the rating it finds lands in
`metadata_rating` -- a column of its own, never folded into the `user_rating`
the request names, because nothing establishes the two are the same number.

    python -m pytest tests/test_qdrant_metadata.py -v
"""
import logging

import numpy as np
import pandas as pd

from src.pipeline.contracts import AlbumContext, Col, GalleryFacts, Services
from src.pipeline.registry import get
from src.readers.qdrant import fetch_photo_metadata

LOGGER = logging.getLogger('test_qdrant_metadata')


class Point:
    def __init__(self, point_id, payload):
        self.id = point_id
        self.payload = payload


class FakeQdrant:
    """Scrolls a fixed list of points, `page` at a time."""

    def __init__(self, points, page=100):
        self.points = points
        self.page = page
        self.calls = []

    def scroll(self, collection_name, limit, offset, with_vectors, with_payload,
               scroll_filter):
        self.calls.append({'collection': collection_name, 'offset': offset,
                           'with_vectors': with_vectors,
                           'with_payload': with_payload})
        start = offset or 0
        chunk = self.points[start:start + self.page]
        next_offset = start + self.page
        return chunk, (next_offset if next_offset < len(self.points) else None)


#: The three shapes the payload actually comes in: nothing, everything, and a
#: rating without keywords.
POINTS = [
    Point(12523890098, {}),
    Point(12523890099, {'rating': 3, 'keywords': ['Keyword 1', 'ישראל', '  ']}),
    Point(12523890100, {'rating': 5}),
]


# ------------------------------------------------------------------ the reader

def test_the_payload_becomes_one_row_per_point():
    metadata = fetch_photo_metadata(FakeQdrant(POINTS), 'C', 53738273, LOGGER)

    assert list(metadata.columns) == ['image_id', 'metadata_rating', 'keywords']
    assert metadata['image_id'].tolist() == [12523890098, 12523890099, 12523890100]
    assert metadata['metadata_rating'].tolist() == [0.0, 3.0, 5.0]
    assert metadata['keywords'].tolist() == [[], ['Keyword 1', 'ישראל'], []]


def test_image_ids_survive_as_int64():
    """Photo ids are past 2^32; a float column would round them."""
    metadata = fetch_photo_metadata(FakeQdrant(POINTS), 'C', 53738273, LOGGER)
    assert metadata['image_id'].dtype == np.int64


def test_the_vectors_are_left_on_the_server():
    """A payload-only scroll is what makes a second pass cheap."""
    client = FakeQdrant(POINTS)
    fetch_photo_metadata(client, 'C', 53738273, LOGGER)
    assert client.calls[0]['with_vectors'] is False
    assert client.calls[0]['with_payload'] == ['rating', 'keywords']


def test_every_page_is_read():
    points = [Point(i, {'rating': 1}) for i in range(250)]
    metadata = fetch_photo_metadata(FakeQdrant(points, page=100), 'C', 1, LOGGER)
    assert len(metadata) == 250


def test_a_project_with_no_points_reads_empty_rather_than_failing():
    metadata = fetch_photo_metadata(FakeQdrant([]), 'C', 1, LOGGER)
    assert metadata.empty
    assert list(metadata.columns) == ['image_id', 'metadata_rating', 'keywords']


def test_a_rating_that_is_not_a_number_reads_as_unrated():
    points = [Point(1, {'rating': 'five'}), Point(2, {'keywords': 'not a list'})]
    metadata = fetch_photo_metadata(FakeQdrant(points), 'C', 1, LOGGER)
    assert metadata['metadata_rating'].tolist() == [0.0, 0.0]
    assert metadata['keywords'].tolist() == [[], []]


# --------------------------------------------------------------- the substages

def context_with(**kwargs):
    context = AlbumContext(logger=LOGGER, project_id=53738273,
                           facts=GalleryFacts(model_version=2),
                           services=Services(qdrant_client=FakeQdrant(POINTS)))
    for key, value in kwargs.items():
        setattr(context, key, value)
    return context


def photos(ids, ratings=None):
    frame = pd.DataFrame({'image_id': np.array(ids, dtype=np.int64)})
    if ratings is not None:
        frame['user_rating'] = ratings
    return frame


def test_a_vectorised_project_is_read():
    context = get('ingest.photo_metadata')()(context_with(is_in_vector_db=True))
    assert len(context.photo_metadata) == 3


def test_a_project_in_blob_storage_is_left_alone():
    """Its metadata sits in a .pai file that nothing reads, and that is the
    whole policy -- no second source, no inferred ratings."""
    for state in (False, None):
        context = get('ingest.photo_metadata')()(context_with(is_in_vector_db=state))
        assert context.photo_metadata is None
        assert not context.failed


def test_a_vector_database_failure_costs_the_ratings_not_the_album():
    class Broken:
        def scroll(self, **kwargs):
            raise RuntimeError('qdrant is unreachable')

    context = context_with(is_in_vector_db=True)
    context.services.qdrant_client = Broken()
    context = get('ingest.photo_metadata')()(context)

    assert context.photo_metadata is None
    assert not context.failed


def merged(frame, metadata=None):
    context = context_with(photos=frame)
    context.photo_metadata = (
        metadata if metadata is not None
        else fetch_photo_metadata(FakeQdrant(POINTS), 'C', 53738273, LOGGER))
    return get('ingest.merge_photo_metadata')()(context)


def test_keywords_reach_the_photo_table():
    context = merged(photos([12523890098, 12523890099, 12523890100]))
    assert context.photos[Col.KEYWORDS].tolist() == [
        [], ['Keyword 1', 'ישראל'], []]


def test_a_photo_the_payload_never_mentions_gets_an_empty_list():
    context = merged(photos([12523890099, 999]))
    assert context.photos[Col.KEYWORDS].tolist() == [['Keyword 1', 'ישראל'], []]
    assert context.photos[Col.METADATA_RATING].tolist() == [3.0, 0.0]


def test_the_payload_rating_lands_in_a_column_of_its_own():
    context = merged(photos([12523890099, 12523890100]))
    assert context.photos[Col.METADATA_RATING].tolist() == [3.0, 5.0]


def test_the_request_rating_is_not_touched():
    """The two ratings come from two places and nothing says they are the same
    number on the same scale, so neither is allowed to stand in for the other.
    Here the request rated one photo 1 and the payload rates it 3; both
    survive, side by side."""
    context = merged(photos([12523890099, 12523890100], ratings=[1.0, 0.0]))

    assert context.photos[Col.USER_RATING].tolist() == [1.0, 0.0]
    assert context.photos[Col.METADATA_RATING].tolist() == [3.0, 5.0]


def test_a_gallery_the_request_never_rated_still_gets_no_user_rating():
    """Reading the payload must not invent a `user_rating` column where
    `ingest.merge_ratings` made none."""
    context = merged(photos([12523890099]))

    assert Col.USER_RATING not in context.photos.columns
    assert context.photos[Col.METADATA_RATING].tolist() == [3.0]


def test_a_gallery_with_no_payload_at_all_is_left_as_it_was():
    frame = photos([12523890099], ratings=[2.0])
    context = merged(frame, metadata=pd.DataFrame(
        columns=['image_id', 'metadata_rating', 'keywords']))

    assert Col.KEYWORDS not in context.photos.columns
    assert Col.METADATA_RATING not in context.photos.columns
    assert context.photos[Col.USER_RATING].tolist() == [2.0]
