"""User ratings: read them, then merge them onto the photo table."""

from __future__ import annotations

from src.pipeline.contracts import AlbumContext, Col, ctx, photo
from src.pipeline.registry import register
from src.pipeline.substage import SubStage
from src.request_processing import read_rating_data


@register
class RatingSubStage(SubStage):
    """Read the rating list from the request or its blob location."""

    name = "ingest.rating"

    def execute(self, context: AlbumContext) -> AlbumContext:
        message = read_rating_data(context.message, context.request, logger=context.logger)
        context.message = message
        context.ratings = getattr(message, 'rating_df', None)
        return context


@register
class MergeRatingsSubStage(SubStage):
    """Join ratings onto the photo table, defaulting unrated photos to 0.

    Separate from the read because the photo table does not exist yet when the
    ratings are read.
    """

    name = "ingest.merge_ratings"
    requires = frozenset({photo(Col.IMAGE_ID)})

    def applies_to(self, context: AlbumContext) -> bool:
        return context.ratings is not None and not context.photos.empty

    def execute(self, context: AlbumContext) -> AlbumContext:
        photos = context.photos.merge(context.ratings, on=Col.IMAGE_ID, how='left')
        photos[Col.USER_RATING] = photos[Col.USER_RATING].fillna(0)
        context.photos = photos

        if context.logger:
            matched = int(photos[Col.USER_RATING].gt(0).sum())
            context.logger.info(
                f"Merged user_rating into gallery_info_df: {matched}/{len(photos)} photos have a rating."
            )
        return context
