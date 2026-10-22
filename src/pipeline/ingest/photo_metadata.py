"""The photographer's rating and keywords, read from the vector database."""

from __future__ import annotations

import pandas as pd

from src.pipeline.contracts import AlbumContext, Col, photo
from src.pipeline.registry import register
from src.pipeline.substage import SubStage
from src.readers.qdrant import fetch_photo_metadata
from utils.configs import CONFIGS


@register
class PhotoMetadataSubStage(SubStage):
    """Read ``rating`` and ``keywords`` off the project's Qdrant payload.

    Skipped when the project is not in the vector database: that gallery's
    metadata sits in blob storage, which nothing reads, so the album is
    composed without it rather than from a second source.

    Best-effort. The payload is extra colour on photos the album can be built
    without, so a vector-database hiccup costs the ratings, not the album --
    unlike ``ingest.embeddings``, which has nothing to lay out without its
    vectors.
    """

    name = "ingest.photo_metadata"
    optional = True

    def applies_to(self, context: AlbumContext) -> bool:
        return context.is_in_vector_db is True

    def execute(self, context: AlbumContext) -> AlbumContext:
        collection_name = CONFIGS["QDRANT_COLLECTION"][context.facts.model_version]

        context.photo_metadata = fetch_photo_metadata(
            context.services.qdrant_client,
            collection_name,
            context.project_id,
            logger=context.logger,
        )
        return context


@register
class MergePhotoMetadataSubStage(SubStage):
    """Join the payload onto the photo table, as columns of its own.

    Separate from the read for the same reason ``ingest.merge_ratings`` is:
    the photo table does not exist yet when the read happens.

    ``metadata_rating`` is written beside ``user_rating``, never into it.
    They are two numbers from two places -- one the request named, one the
    payload held -- and nothing says they are produced by the same process or
    live on the same scale. Reconciling them is a decision somebody has to
    make with that knowledge; this substage only puts both on the table so it
    can be made. Nothing reads ``metadata_rating`` yet, and that is deliberate:
    folding it into scoring would be that decision, taken silently.
    """

    name = "ingest.merge_photo_metadata"
    requires = frozenset({photo(Col.IMAGE_ID)})
    provides = frozenset({photo(Col.METADATA_RATING), photo(Col.KEYWORDS)})
    optional = True

    def applies_to(self, context: AlbumContext) -> bool:
        metadata = context.photo_metadata
        return (metadata is not None and not metadata.empty
                and context.photos is not None and not context.photos.empty)

    def execute(self, context: AlbumContext) -> AlbumContext:
        photos = context.photos
        metadata = context.photo_metadata.drop_duplicates(
            subset=Col.IMAGE_ID, keep='last').set_index(Col.IMAGE_ID)

        keywords = photos[Col.IMAGE_ID].map(metadata[Col.KEYWORDS])
        photos[Col.KEYWORDS] = [
            list(words) if isinstance(words, (list, tuple)) else []
            for words in keywords
        ]

        # A photo the payload never mentions is unrated by the payload, which
        # is 0 -- the same thing `ingest.merge_ratings` means by it.
        photos[Col.METADATA_RATING] = pd.to_numeric(
            photos[Col.IMAGE_ID].map(metadata[Col.METADATA_RATING]),
            errors='coerce').fillna(0)

        context.photos = photos

        if context.logger:
            rated = int(photos[Col.METADATA_RATING].gt(0).sum())
            tagged = int(photos[Col.KEYWORDS].map(bool).sum())
            context.logger.info(
                f"Merged the Qdrant payload into gallery_info_df: "
                f"{rated}/{len(photos)} photos carry a metadata_rating, "
                f"{tagged}/{len(photos)} carry keywords. "
                f"Neither touches user_rating.")
        return context
