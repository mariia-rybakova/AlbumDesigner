"""One line per request saying what the photographer's metadata changed."""

from __future__ import annotations

from src.core.keywords import HIGHLIGHTS_COLUMN, RATING_COLUMN, is_for_album
from src.pipeline.contracts import AlbumContext, Col
from src.pipeline.registry import register
from src.pipeline.substage import SubStage


@register
class KeywordReportSubStage(SubStage):
    """Report the priority set against what the album actually kept.

    Last in SELECT, because `used` is the only one of these numbers that is not
    known until the picker has finished. The rest were counted by
    `enrich.album_keywords` over the whole gallery, before anything narrowed
    it, and ride along on the context.

    Reports rather than decides -- nothing downstream reads it. It exists so
    that a photographer asking why their marked photograph is not in the album
    can be answered from the log alone: either it was never in the priority set
    (keyword misspelt, or the gallery was not vectorised at all, which `read`
    distinguishes), or it was and the budget for its category went elsewhere.

    Runs on every request, the manual path included, where the numbers say what
    the photographer's marks would have meant had the album been chosen for
    them. Logs nothing at all for a gallery whose keywords were never read.
    """

    name = "select.keyword_report"

    def applies_to(self, context: AlbumContext) -> bool:
        return context.keyword_summary.read and context.logger is not None

    def execute(self, context: AlbumContext) -> AlbumContext:
        summary = context.keyword_summary
        kept = context.photos

        used = rated_used = reel_used = 0
        if kept is not None and not kept.empty:
            if Col.KEYWORDS in kept.columns:
                used = int(kept[Col.KEYWORDS].map(is_for_album).sum())
            if RATING_COLUMN in kept.columns:
                rated_used = int((kept[RATING_COLUMN] > 0).sum())
            if HIGHLIGHTS_COLUMN in kept.columns:
                reel_used = int(kept[HIGHLIGHTS_COLUMN].fillna(False).sum())

        album = context.variant_name or "album"
        context.logger.info(
            f"Keyword report [{album}]: excluded {summary.excluded}; "
            f"priority set {summary.prioritised} keyworded album, "
            f"{summary.rated} rated, {summary.highlighted} in Highlights; "
            f"used {used}/{summary.prioritised} keyworded, "
            f"{rated_used}/{summary.rated} rated and "
            f"{reel_used}/{summary.highlighted} from Highlights "
            f"in the {len(kept) if kept is not None else 0} photos selected")
        return context
