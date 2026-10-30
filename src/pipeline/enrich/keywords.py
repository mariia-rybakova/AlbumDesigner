"""Acting on the keywords and ratings the photographer addressed to the album."""

from __future__ import annotations

from src.core.keywords import (HIGHLIGHTS_COLUMN, RATING_COLUMN,
                               is_for_album, is_not_for_album)
from src.pipeline.contracts import AlbumContext, Col
from src.pipeline.registry import register
from src.pipeline.substage import SubStage


@register
class AlbumKeywordsSubStage(SubStage):
    """Remove the photographs marked ``notforalbum``, and count the rest.

    First of everything, and that placement is the rule rather than a detail.
    It runs before `enrich.duplicate_shots`, so an excluded frame can neither
    win the coin toss for which copy of a shot is kept nor be set aside as some
    kept photo's `treatment_twin` -- which is the one way a dropped photo gets
    back onto a page, when the layout swaps a frame for its other treatment.
    It runs before `enrich.gallery_type`, `select.budget` and
    `select.route`'s `all_photos` snapshot, so the album is sized against the
    photographs it may actually use and the covers are chosen from them too.

    There is no competing rule and no scoring to lose to. `album` on the same
    photograph does not save it: that photograph is not on the table by the
    time anything reads `album`.

    The counting is here rather than anywhere later because this is the last
    moment the whole gallery is on the table -- every number after this is of
    a pool something has already narrowed. The *acting* on `album` and on
    `metadata_rating` is not here at all; both are a lift on the score, applied
    by `core.keywords.priority_lift` wherever a photograph is ranked.

    Skipped, silently, on a gallery with no keyword column -- one that was
    never vectorised, whose metadata sits in blob storage that nothing reads.
    Absent keywords are not an instruction either way, and `read=False` on the
    summary is what tells the report the zeros mean "nothing was read".
    """

    name = "enrich.album_keywords"

    def applies_to(self, context: AlbumContext) -> bool:
        return (context.photos is not None and not context.photos.empty
                and Col.KEYWORDS in context.photos.columns)

    def execute(self, context: AlbumContext) -> AlbumContext:
        photos = context.photos
        summary = context.keyword_summary
        summary.read = True

        excluded = photos[Col.KEYWORDS].map(is_not_for_album)
        summary.excluded = int(excluded.sum())
        if summary.excluded:
            photos = photos[~excluded]
            context.photos = photos

        summary.prioritised = int(photos[Col.KEYWORDS].map(is_for_album).sum())
        if RATING_COLUMN in photos.columns:
            summary.rated = int((photos[RATING_COLUMN] > 0).sum())
        if HIGHLIGHTS_COLUMN in photos.columns:
            summary.highlighted = int(photos[HIGHLIGHTS_COLUMN].fillna(False).sum())

        if context.logger:
            context.logger.info(
                f"Album keywords: dropped {summary.excluded} photos keyworded "
                f"notforalbum; {len(photos)} remain, of which "
                f"{summary.prioritised} are keyworded album, "
                f"{summary.rated} carry a metadata rating and "
                f"{summary.highlighted} are in Highlights")
        return context
