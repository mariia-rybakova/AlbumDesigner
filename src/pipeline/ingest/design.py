"""Read the album's design/layout data (from the request or its blob)."""

from __future__ import annotations

from src.pipeline.contracts import AlbumContext, DesignSpec
from src.pipeline.registry import register
from src.pipeline.substage import SubStage
from src.request_processing import read_layouts_data


@register
class DesignSubStage(SubStage):
    """Populate ``message.designsInfo`` / ``message.pagesInfo`` and mirror the
    summary onto ``context.designs``."""

    name = "ingest.design"

    def execute(self, context: AlbumContext) -> AlbumContext:
        message = context.message

        result = read_layouts_data(message, context.request, logger=context.logger)

        # read_layouts_data returns either the message or a (None, error) tuple.
        if result is None or isinstance(result, tuple):
            error = result[1] if isinstance(result, tuple) and len(result) > 1 else 'unknown error'
            raise ValueError('read_layouts_data failed: {}'.format(error))

        context.message = result
        context.request = result.content

        context.designs = DesignSpec(
            album_ar=result.content.get('album_ar', {'anyPage': 2}),
            pages=dict(getattr(result, 'pagesInfo', {}) or {}),
            designs=dict(getattr(result, 'designsInfo', {}) or {}),
        )
        return context
