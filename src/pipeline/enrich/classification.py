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

import numpy as np
import pandas as pd

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
        # A gallery with no embeddings at all should say so. `generate_query`
        # drops every row it cannot tag and only creates its two columns `if
        # results:`, so with nothing to tag it empties the photo table *and*
        # leaves the columns absent -- and the failure then surfaces as this
        # substage breaking its own `provides` contract, which says nothing
        # about the cause. It should not happen in practice; when it does, the
        # embeddings were never fetched (the project is not in the vector
        # database, or the service was unreachable -- locally, a dropped VPN).
        usable = _usable_embeddings(context.photos)
        if usable == 0:
            return context.fail(
                f"No usable image embedding on any of {len(context.photos)} photos, so "
                f"nothing can be tagged, scored or matched against a concept. The "
                f"embeddings were not fetched: either the project is not in the vector "
                f"database, or the embedding source could not be reached."
            )

        if usable < len(context.photos) and context.logger:
            context.logger.warning(
                f"{len(context.photos) - usable} of {len(context.photos)} photos have no "
                f"embedding and will be dropped by the tagging")

        context.photos = add_semantic_tags(context.photos, context.logger)
        return context


def _usable_embeddings(photos: pd.DataFrame) -> int:
    """How many rows carry an embedding the tagger can actually project."""
    if photos is None or photos.empty or Col.EMBEDDING not in photos.columns:
        return 0

    def usable(value) -> bool:
        if value is None:
            return False
        try:
            return np.asarray(value, dtype=float).ravel().size > 0
        except (TypeError, ValueError):
            return False

    return int(photos[Col.EMBEDDING].apply(usable).sum())
