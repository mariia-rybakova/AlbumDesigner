"""How many photos each category gets, and how many spreads it should fill."""

from __future__ import annotations

from src.pipeline.contracts import AlbumContext, Col, photo
from src.pipeline.registry import register
from src.pipeline.select.allocation import allocate
from src.pipeline.select.contracts import SelectionPlan
from src.pipeline.substage import SubStage
from src.selection.ai_wedding_selection import load_event_mapping
from utils.configs import CONFIGS
from utils.lookup_table_tools import wedding_lookup_table


@register
class BudgetSubStage(SubStage):
    """Turn the focus profile into a per-category photo and spread allowance.

    The focus profile (``files/focus_csv.csv``) says what share of the album
    each category deserves for this relationship to the couple. That share is
    reconciled against what the gallery actually contains: categories that come
    up short hand their unused spreads to categories with a surplus.

    The arithmetic lives in :mod:`src.pipeline.select.allocation`, as named
    steps rather than one pass, because the ceremony's ``yes`` classes -- the
    kiss, the processionals, the send-off -- need settling between two of them.
    See that module's docstring for what they cost the album and why.
    """

    name = "select.budget"
    requires = frozenset({photo(Col.CLUSTER_CONTEXT)})

    def applies_to(self, context: AlbumContext) -> bool:
        return bool(context.facts.is_wedding) and not context.selection.manual

    def execute(self, context: AlbumContext) -> AlbumContext:
        inputs = context.selection_inputs
        logger = context.logger

        focus_table = _profile_for(inputs.focus, logger)

        available = {
            category: len(frame)
            for category, frame in context.photos.groupby(Col.CLUSTER_CONTEXT)
        }

        allocation = allocate(
            available,
            focus_table,
            wedding_lookup_table,
            inputs.density,
            context.photos,
            logger,
        )

        if not allocation.images:
            return context.fail("No images got selected!")

        if logger:
            logger.info(f"Budget: {allocation.summary()}")

        context.selection_plan = SelectionPlan(
            images=allocation.images,
            spreads=allocation.spreads,
            min_total_spreads=allocation.min_total_spreads,
            max_total_spreads=allocation.max_total_spreads,
            lookup_table=context.selection.lookup_table,
            yes_categories=tuple(allocation.yes_categories),
        )
        return context


def _profile_for(focus, logger):
    """The focus profile for the requested relationship.

    Falls back to the couple's own profile when the request names one we do not
    have a column for.

    The matching ``relations`` entry used to be read here too and handed on as
    ``calculate_optimal_selection``'s ``image_lookup_table``, which never
    referenced it -- the parameter is dead in the original and always was. It is
    no longer read.
    """
    event_mapping = load_event_mapping(CONFIGS['focus_csv_path'], logger)

    if len(focus) > 0:
        return event_mapping.get(focus[0], event_mapping['brideAndGroom'])
    return event_mapping['brideAndGroom']
