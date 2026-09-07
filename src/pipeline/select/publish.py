"""Narrow the photo table to the selection and hand it to album processing."""

from __future__ import annotations

from src.pipeline.contracts import AlbumContext, Col
from src.pipeline.registry import register
from src.pipeline.substage import SubStage


@register
class PublishSubStage(SubStage):
    """Finalise the selection outcome on the context.

    Manual requests skip this: ``select.route`` already aligned the photo table
    with the user's list and there is no budget to publish.
    """

    name = "select.publish"

    def applies_to(self, context: AlbumContext) -> bool:
        return not context.selection.manual

    def execute(self, context: AlbumContext) -> AlbumContext:
        outcome = context.selection
        plan = context.selection_plan

        if plan is not None:
            outcome.spreads = plan.spreads or {}
            outcome.min_total_spreads = plan.min_total_spreads
            outcome.max_total_spreads = plan.max_total_spreads

        context.photos = context.photos[
            context.photos[Col.IMAGE_ID].isin(outcome.photo_ids)
        ]

        self._cache_couple_photos(context)
        return context

    @staticmethod
    def _cache_couple_photos(context: AlbumContext) -> None:
        """First-page generation picks its hero shot from the couple's photos,
        so stash them while the categories are still on the frame."""
        message = context.message
        pages = getattr(message, 'pagesInfo', {}) if message is not None else {}

        if not pages.get("firstPage"):
            if message is not None:
                message.content['bride and groom'] = None
            return

        if context.facts.is_wedding is False:
            return

        couple = context.photos[context.photos[Col.CLUSTER_CONTEXT] == "bride and groom"]
        if message is not None:
            message.content['bride and groom'] = couple
