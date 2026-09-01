"""Data-quality gates between enrichment steps."""

from __future__ import annotations

from src.pipeline.contracts import AlbumContext, Col, photo
from src.pipeline.registry import register
from src.pipeline.substage import SubStage
from utils.read_protos_files import require_cluster_data


@register
class RequireClusterDataSubStage(SubStage):
    """Drop photos missing the cluster columns everything downstream assumes.

    Runs after the semantic tagging, matching the original order: tagging is
    what drops rows with unusable embeddings, and this drops rows the
    content-clustering model had nothing to say about.
    """

    name = "enrich.require_cluster_data"
    requires = frozenset({photo(Col.RANKING), photo(Col.CLUSTER_LABEL)})

    def execute(self, context: AlbumContext) -> AlbumContext:
        context.photos = require_cluster_data(context.photos, context.logger)
        return context
