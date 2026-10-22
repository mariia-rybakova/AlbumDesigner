"""Reading the photographer's own metadata off the vector database.

Qdrant holds one point per vectorised photo, keyed by the photo's id, and its
payload carries a star ``rating`` and a list of ``keywords``. Both are optional
-- plenty of points carry neither -- so this reads whatever is there and says
nothing about the rest.

The ``rating`` here is reported as ``metadata_rating``, not ``user_rating``.
Where it is written and by what is not established; it is not assumed to be the
rating the album request names, and nothing merges the two.

Only the payload is read here. The vectors come down a separate scroll
(``src.request_processing.fetch_vectors_from_qdrant``), which asks for no
payload at all; a payload-only scroll carries almost nothing over the wire, so
two narrow passes cost less than one pass dragging the vectors through the
payload decode.

A gallery that was never vectorised has no points here. Its metadata lives in
blob storage instead, and nothing reads it from there -- the album is composed
without it, exactly as it was before this module existed.
"""

import numpy as np
import pandas as pd

#: Payload keys as Qdrant spells them.
RATING_KEY = 'rating'
KEYWORDS_KEY = 'keywords'

#: Columns this module produces, in the photo table's spelling.
COLUMNS = ['image_id', 'metadata_rating', 'keywords']

#: Points per scroll request. The same page size the vector scroll uses.
PAGE = 100


def _rating(value):
    """A star rating as a float, or 0.0 for anything that is not one."""
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        rating = float(value)
    except (TypeError, ValueError):
        return 0.0
    return rating if np.isfinite(rating) else 0.0


def _keywords(value):
    """The keyword list, stripped of blanks. A missing list reads as empty."""
    if not isinstance(value, (list, tuple)):
        return []
    return [str(word).strip() for word in value if str(word).strip()]


def empty_metadata():
    """The frame this module returns when there is nothing to return."""
    return pd.DataFrame({'image_id': pd.Series(dtype=np.int64),
                         'metadata_rating': pd.Series(dtype=float),
                         'keywords': pd.Series(dtype=object)})


def fetch_photo_metadata(client, collection_name, project_id, logger=None):
    """Scroll one project's points and return their rating and keywords.

    One row per point in the collection, whether or not it carried either
    field; the photos the project never vectorised are simply absent.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    rows = []
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=PAGE,
            offset=offset,
            with_vectors=False,
            with_payload=[RATING_KEY, KEYWORDS_KEY],
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="projectId",
                        match=MatchValue(value=project_id)
                    )
                ]
            )
        )

        for point in points:
            payload = point.payload or {}
            rows.append({
                'image_id': point.id,
                'metadata_rating': _rating(payload.get(RATING_KEY)),
                'keywords': _keywords(payload.get(KEYWORDS_KEY)),
            })

        if offset is None:
            break

    if not rows:
        if logger:
            logger.info(
                f'No payload for projectId {project_id} in {collection_name}')
        return empty_metadata()

    metadata = pd.DataFrame(rows, columns=COLUMNS)
    metadata['image_id'] = metadata['image_id'].astype(np.int64)

    if logger:
        rated = int((metadata['metadata_rating'] > 0).sum())
        tagged = int(metadata['keywords'].map(bool).sum())
        logger.info(
            f'Read Qdrant payload for {len(metadata)} photos of project '
            f'{project_id}: {rated} rated, {tagged} with keywords')

    return metadata
