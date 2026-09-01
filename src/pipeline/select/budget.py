"""How many photos each category gets, and how many spreads it should fill."""

from __future__ import annotations

from src.pipeline.contracts import AlbumContext, Col, photo
from src.pipeline.registry import register
from src.pipeline.select.contracts import SelectionPlan
from src.pipeline.substage import SubStage
from src.selection.ai_wedding_selection import calculate_optimal_selection, load_event_mapping
from utils.configs import CONFIGS, relations
from utils.lookup_table_tools import wedding_lookup_table


@register
class BudgetSubStage(SubStage):
    """Turn the focus profile into a per-category photo and spread allowance.

    The focus profile (``files/focus_csv.csv``) says what share of the album
    each category deserves for this relationship to the couple. That share is
    reconciled against what the gallery actually contains: categories that come
    up short hand their unused spreads to categories with a surplus.
    """

    name = "select.budget"
    requires = frozenset({photo(Col.CLUSTER_CONTEXT)})

    def applies_to(self, context: AlbumContext) -> bool:
        return bool(context.facts.is_wedding) and not context.selection.manual

    def execute(self, context: AlbumContext) -> AlbumContext:
        inputs = context.selection_inputs
        logger = context.logger

        focus_table, _relation_table = _profile_for(inputs.focus, logger)

        available = {
            category: len(frame)
            for category, frame in context.photos.groupby(Col.CLUSTER_CONTEXT)
        }

        images, spreads, min_total, max_total = calculate_optimal_selection(
            available,
            _relation_table,
            wedding_lookup_table,
            focus_table,
            inputs.density,
            context.photos,
            logger,
        )

        if images is None:
            return context.fail("No images got selected!")

        context.selection_plan = SelectionPlan(
            images=images,
            spreads=spreads,
            min_total_spreads=min_total,
            max_total_spreads=max_total,
            lookup_table=context.selection.lookup_table,
        )
        return context


def _profile_for(focus, logger):
    """Focus profile + relation table for the requested relationship.

    Falls back to the couple's own profile when the request names one we do not
    have a column for.
    """
    event_mapping = load_event_mapping(CONFIGS['focus_csv_path'], logger)

    if len(focus) > 0:
        return (
            event_mapping.get(focus[0], event_mapping['brideAndGroom']),
            relations.get(focus[0], relations['brideAndGroom']),
        )
    return event_mapping['brideAndGroom'], relations['brideAndGroom']
