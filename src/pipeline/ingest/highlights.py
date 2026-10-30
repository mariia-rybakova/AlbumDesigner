"""Whether a photograph is in the gallery's Highlights reel."""

from __future__ import annotations

from src.pipeline.contracts import AlbumContext, Col, photo
from src.pipeline.registry import register
from src.pipeline.substage import SubStage
from src.readers.gallery import highlight_photo_ids


@register
class HighlightsSubStage(SubStage):
    """Flag the photographs the photographer put in Highlights.

    Last in INGEST, so the flag is on the table before `enrich.album_keywords`
    counts the priority set and before anything ranks a photograph.

    Best-effort, like `ingest.photo_metadata`: the scene list comes from blob
    storage, and a gallery whose scenes cannot be read is one whose
    photographs are unmarked rather than one with no album. The column is
    written either way, so no later rule has to ask whether this gallery had a
    reel -- an absent mark and an unreadable one both read as False.
    """

    name = "ingest.highlights"
    requires = frozenset({photo(Col.IMAGE_ID)})
    provides = frozenset({photo(Col.IN_HIGHLIGHTS)})
    optional = True

    def applies_to(self, context: AlbumContext) -> bool:
        return (context.photos is not None and not context.photos.empty
                and context.project_url is not None)

    def execute(self, context: AlbumContext) -> AlbumContext:
        ids = highlight_photo_ids(context.project_url, context.logger)

        photos = context.photos
        photos[Col.IN_HIGHLIGHTS] = photos[Col.IMAGE_ID].isin(ids)
        context.photos = photos

        if context.logger:
            marked = int(photos[Col.IN_HIGHLIGHTS].sum())
            context.logger.info(
                f"Highlights: {marked}/{len(photos)} photos of this album's "
                f"pool are in the reel")
        return context
