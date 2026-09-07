"""Pic-Time gallery scene ordering."""

from __future__ import annotations

from src.pipeline.contracts import AlbumContext, Col, photo
from src.pipeline.registry import register
from src.pipeline.substage import SubStage
from src.request_processing import add_scenes_info


@register
class ScenesSubStage(SubStage):
    """Overlay the gallery's scene ordering onto the photo table.

    Note: ``add_scenes_info`` currently returns before applying the mapping it
    builds (there is unreachable code after its ``return``), so in practice this
    is a no-op and ``scene_order`` keeps the value read from
    ``bg_segmentation.pb``. Kept as its own substage — behaviour unchanged —
    precisely so the fix is a change to one replaceable unit.
    """

    name = "ingest.scenes"
    requires = frozenset({photo(Col.IMAGE_ID)})
    optional = True

    def execute(self, context: AlbumContext) -> AlbumContext:
        context.photos = add_scenes_info(context.photos, context.project_url, context.logger)
        return context
