"""Time normalisation and the time-derived event detectors."""

from __future__ import annotations

from src.pipeline.contracts import AlbumContext, Col, photo
from src.pipeline.registry import register
from src.pipeline.substage import SubStage
from src.request_processing import identify_kiss_ceremony
from utils.time_processing import process_gallery_time


@register
class TemporalSubStage(SubStage):
    """Normalise timestamps into a usable timeline.

    Produces ``image_time_date`` (a real timestamp), ``general_time`` (seconds
    from the first photo) and the ``is_artificial_time`` flag that tells the
    rest of the pipeline the EXIF times could not be trusted and scene order
    should be used instead.
    """

    name = "enrich.temporal"
    requires = frozenset({photo(Col.IMAGE_TIME)})
    provides = frozenset({photo(Col.IMAGE_TIME_DATE), photo(Col.GENERAL_TIME)})

    def execute(self, context: AlbumContext) -> AlbumContext:
        photos, is_artificial_time = process_gallery_time(
            context.message, context.photos, context.logger
        )
        context.photos = photos
        context.facts.is_artificial_time = is_artificial_time
        return context


@register
class CeremonyKissSubStage(SubStage):
    """Find the ceremony kiss and give it its own category.

    Anchors on the last "officiant leading wedding ceremony" frame inside the
    ceremony window and re-labels kiss photos within +/-6 minutes of it as
    ``may kiss bride``. Self-gating: galleries that fail the eligibility rules
    (no ceremony, multi-day ceremony, missing columns) pass through untouched.
    """

    name = "enrich.ceremony_kiss"

    def execute(self, context: AlbumContext) -> AlbumContext:
        context.photos = identify_kiss_ceremony(context.photos, logger=context.logger)
        return context
