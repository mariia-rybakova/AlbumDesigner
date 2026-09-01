"""Decide how this request gets selected, and resolve the shared inputs."""

from __future__ import annotations

import pandas as pd

from src.pipeline.contracts import AlbumContext, Col, SelectionOutcome, photo
from src.pipeline.registry import register
from src.pipeline.select.contracts import SelectionInputs
from src.pipeline.select.scoring import build_ratings
from src.pipeline.substage import SubStage
from src.selection.auto_selection import get_tags_bins
from utils.configs import CONFIGS
from utils.lookup_table_tools import wedding_lookup_table

#: Categories whose spread budget is uncapped when the user picked the photos
#: themselves — their choices should not be squeezed into a default allowance.
MANUAL_UNCAPPED = ('other', 'None')


@register
class RouteSubStage(SubStage):
    """Split manual from AI, and prepare whatever the chosen path needs.

    Manual requests are finished here: the user already chose, so all that is
    left is to align the photo table with their list and widen the lookup table.
    AI requests get their pool, their density-scaled lookup table and their
    resolved inputs, and continue to ``select.budget``.
    """

    name = "select.route"
    requires = frozenset({photo(Col.IMAGE_ID)})

    def execute(self, context: AlbumContext) -> AlbumContext:
        if not context.hints.present:
            return self._manual(context)
        return self._ai(context)

    # -- manual -------------------------------------------------------------

    @staticmethod
    def _manual(context: AlbumContext) -> AlbumContext:
        logger = context.logger
        if logger:
            logger.info("aiMetadata not found. Continue with chosen photos.")

        chosen = pd.DataFrame(context.available_photo_ids, columns=[Col.IMAGE_ID])
        context.photos = chosen.merge(context.photos, how='inner', on=Col.IMAGE_ID)

        lookup_table = None
        if context.facts.is_wedding:
            lookup_table = wedding_lookup_table.copy()
            for category in MANUAL_UNCAPPED:
                lookup_table[category] = (24, 4)

        context.selection = SelectionOutcome(
            photo_ids=list(context.available_photo_ids),
            manual=True,
            lookup_table=lookup_table,
        )
        return context

    # -- ai -----------------------------------------------------------------

    @staticmethod
    def _ai(context: AlbumContext) -> AlbumContext:
        photos = context.photos
        if photos.empty:
            return context.fail("Gallery photos info DataFrame is empty")

        if context.available_photo_ids:
            photos = photos[photos[Col.IMAGE_ID].isin(context.available_photo_ids)]
            context.photos = photos

        # Snapshot before selection narrows it; first/last page generation and
        # the cover need the full pool.
        context.all_photos = photos.copy()

        if photos.empty:
            return context.fail("Gallery photos info DataFrame is empty")

        hints = context.hints

        lookup_table = None
        if context.facts.is_wedding:
            lookup_table = _scaled_lookup_table(hints.density)

        model_version = photos.iloc[0][Col.MODEL_VERSION]
        tags_features = get_tags_bins(hints.subjects, model_version, context.logger)

        context.selection_inputs = SelectionInputs(
            user_selected_ids=hints.photo_ids,
            person_ids=hints.person_ids,
            tags_features=tags_features,
            ratings=build_ratings(context.request.get('rating', [])),
            density=hints.density,
            focus=hints.focus,
        )
        context.selection = SelectionOutcome(manual=False, lookup_table=lookup_table)
        return context


def _scaled_lookup_table(density: int) -> dict:
    """Photos-per-spread, scaled by the requested density and clamped."""
    factor = CONFIGS['density_factors'].get(density, 1)
    return {
        category: (min(24, max(1, photos_per_spread * factor)), std)
        for category, (photos_per_spread, std) in wedding_lookup_table.copy().items()
    }
