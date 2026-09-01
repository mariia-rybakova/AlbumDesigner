"""Classification: what kind of gallery is this, and what is each photo of.

Three substages, all derivation rather than reading:

``enrich.gallery_type``
    Wedding or not, from the share of unclassified photos.
``enrich.content_class``
    ``cluster_class`` (an int from the content-clustering model) becomes
    ``cluster_context`` (the category name the whole downstream pipeline keys
    off).
``enrich.semantic_tags``
    Projects every photo embedding against a bank of pre-computed text queries
    and keeps the best match. This is a search over the gallery, not a read of
    it, which is why it sits here.
"""

from __future__ import annotations

from src.pipeline.contracts import AlbumContext, Col, photo
from src.pipeline.registry import register
from src.pipeline.substage import SubStage
from utils.read_protos_files import add_content_class, add_semantic_tags, classify_gallery_type


@register
class GalleryTypeSubStage(SubStage):
    """Decide whether this gallery is a wedding."""

    name = "enrich.gallery_type"
    requires = frozenset({photo(Col.IMAGE_CLASS)})

    def execute(self, context: AlbumContext) -> AlbumContext:
        context.facts.is_wedding = classify_gallery_type(context.photos)
        if context.logger:
            context.logger.info(f"Gallery type: is_wedding={context.facts.is_wedding}")
        return context


@register
class ContentClassSubStage(SubStage):
    """Name each photo's content category.

    Wedding-only, matching the original: non-wedding galleries never get a
    ``cluster_context`` column at all.
    """

    name = "enrich.content_class"
    requires = frozenset({photo(Col.CLUSTER_CLASS)})
    provides = frozenset({photo(Col.CLUSTER_CONTEXT)})

    def applies_to(self, context: AlbumContext) -> bool:
        return bool(context.facts.is_wedding)

    def execute(self, context: AlbumContext) -> AlbumContext:
        context.photos = add_content_class(context.photos)
        return context


@register
class SemanticTagsSubStage(SubStage):
    """Tag each photo by nearest pre-computed text query."""

    name = "enrich.semantic_tags"
    requires = frozenset({photo(Col.EMBEDDING), photo(Col.MODEL_VERSION)})
    provides = frozenset({photo(Col.IMAGE_QUERY_CONTENT), photo(Col.IMAGE_SUBQUERY_CONTENT)})

    def execute(self, context: AlbumContext) -> AlbumContext:
        context.photos = add_semantic_tags(context.photos, context.logger)
        return context
