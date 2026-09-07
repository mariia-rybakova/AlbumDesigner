"""Parse and validate the incoming request. No I/O."""

from __future__ import annotations

from src.pipeline.contracts import AiHints, AlbumContext, ctx
from src.pipeline.registry import register
from src.pipeline.substage import SubStage


@register
class RequestSubStage(SubStage):
    """Validate the request envelope and lift its fields onto the context."""

    name = "ingest.request"
    provides = frozenset({ctx("project_url")})

    def execute(self, context: AlbumContext) -> AlbumContext:
        content = context.request

        if not isinstance(content, (dict, list)):
            context.logger.warning('Incorrect message format: {}.'.format(content))

        if 'photos' not in content or 'base_url' not in content:
            return context.fail(
                'There are missing fields in input request: {}. Skipping.'.format(content)
            )

        context.project_url = content['base_url']
        context.project_id = content.get('projectId')
        context.available_photo_ids = content.get('photos', []) or []
        context.hints = AiHints.from_request(content)

        return context
