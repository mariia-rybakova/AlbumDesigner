"""The fallback: spread the picks across content clusters."""

from __future__ import annotations

from src.pipeline.select.contracts import CategoryPicks, CategoryRequest
from src.pipeline.select.strategies.base import CategoryStrategy
from utils.selection.wedding_selection_tools import get_clusters, select_non_similar_images


class ContentClusterStrategy(CategoryStrategy):
    """Round-robin across the content-clustering labels.

    Used by every category without a bespoke rule — details, settings, food,
    rings and so on — where the only requirement is not picking five versions
    of the same shot.
    """

    handles = ()

    def pick(self, request: CategoryRequest) -> CategoryPicks:
        clusters = get_clusters(request.color.reset_index())
        preferred = select_non_similar_images(clusters, request.order_index, request.need)
        return CategoryPicks(preferred=preferred)
