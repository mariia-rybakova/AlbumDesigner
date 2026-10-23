"""Acting on the keywords the photographer addressed to the album."""

from __future__ import annotations

from src.pipeline.contracts import AlbumContext, Col
from src.pipeline.registry import register
from src.pipeline.substage import SubStage
from src.core.keywords import is_for_album, is_not_for_album


@register
class AlbumKeywordsSubStage(SubStage):
    """Remove every photograph the photographer marked ``notforalbum``.

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

    The *acting* on `album` is not here at all -- it is a lift on the score,
    applied by `core.keywords.for_album_lift` wherever a photograph is ranked.
    What is here is the count, because this is the last moment the whole
    gallery is on the table and every number after it is of a pool something
    has already narrowed.

    Skipped, silently, on a gallery with no keyword column -- one that was
    never vectorised, whose metadata sits in blob storage that nothing reads.
    Absent keywords are not an instruction either way.
    """

    name = "enrich.album_keywords"

    def applies_to(self, context: AlbumContext) -> bool:
        return (context.photos is not None and not context.photos.empty
                and Col.KEYWORDS in context.photos.columns)

    def execute(self, context: AlbumContext) -> AlbumContext:
        photos = context.photos

        excluded = photos[Col.KEYWORDS].map(is_not_for_album)
        dropped = int(excluded.sum())
        if dropped:
            photos = photos[~excluded]
            context.photos = photos

        if context.logger:
            prioritised = int(photos[Col.KEYWORDS].map(is_for_album).sum())
            context.logger.info(
                f"Album keywords: dropped {dropped} photos keyworded "
                f"notforalbum; {len(photos)} remain, of which "
                f"{prioritised} are keyworded album")
        return context
