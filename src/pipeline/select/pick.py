"""The picker: walk the categories and let each one's strategy choose.

The driver owns everything that is the same for every category — scoring,
gating, honouring the user's own picks, temporal narrowing, the colour split and
the greyscale top-up. What differs per category lives in a
:class:`~src.pipeline.select.strategies.base.CategoryStrategy`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.pipeline.contracts import AlbumContext, Col, photo
from src.pipeline.registry import register
from src.pipeline.select import narrowing
from src.pipeline.select.contracts import CategoryRequest
from src.pipeline.select.scoring import CandidateGate, Scorer
from src.pipeline.select.strategies import StrategyRegistry, default_registry
from src.pipeline.substage import SubStage
from src.selection.ai_non_wedding_selection import smart_non_wedding_selection
from utils.selection.refactoring import select_remove_similar

#: Categories where a photo the user picked wins outright, and where no
#: diversity pass runs — there is rarely more than one shot of the dress or the
#: rings worth having. Handled here rather than as a strategy because they are
#: resolved before the pool is narrowed.
USER_PREFERENCE_CATEGORIES = ('accessories', 'wedding dress')

#: At most this many greyscale photos when a category has no colour at all.
GRAYSCALE_ONLY_CAP = 2


@register
class PickSubStage(SubStage):
    """Choose the photos. Wedding galleries go category by category;
    non-wedding galleries are picked by people composition instead."""

    name = "select.pick"
    requires = frozenset({photo(Col.IMAGE_ID)})

    def __init__(self, strategies: Optional[StrategyRegistry] = None, **options):
        super().__init__(**options)
        self.strategies = strategies or default_registry()

    def applies_to(self, context: AlbumContext) -> bool:
        return not context.selection.manual

    def execute(self, context: AlbumContext) -> AlbumContext:
        if context.facts.is_wedding:
            picker = WeddingPicker(context, self.strategies)
            chosen, per_category = picker.run()
            context.selection.per_category = per_category
        else:
            chosen, error = smart_non_wedding_selection(context.photos, logger=context.logger)
            if error:
                return context.fail(error)

        context.logger.info(f"Total images: {len(chosen or [])}")
        # De-duplicate while keeping the order categories contributed in.
        context.selection.photo_ids = list(dict.fromkeys(chosen or []))
        return context


class WeddingPicker:
    """One pass over the gallery's content categories."""

    def __init__(self, context: AlbumContext, strategies: StrategyRegistry):
        self.context = context
        self.strategies = strategies
        self.logger = context.logger
        self.plan = context.selection_plan
        self.inputs = context.selection_inputs

        self.user_selected = context.photos[
            context.photos[Col.IMAGE_ID].isin(self.inputs.user_selected_ids)
        ]

        scorer = Scorer(
            self.user_selected,
            self.inputs.person_ids,
            self.inputs.tags_features,
            self.inputs.ratings,
            self.logger,
        )
        self.gate = CandidateGate(scorer, self.inputs.unscored, self.plan.images, self.logger)

        # Whatever `select.preselect` committed is already in the album; the
        # loop below starts from it and does not offer those photos again.
        self.committed: Dict = dict(self.plan.committed)
        self.chosen: List = list(self.committed)
        self.per_category: Dict[str, Dict[str, int]] = {}

        # Reproduces a quirk of the monolith this replaced: one branch could
        # finish without assigning its ranked list, in which case the category
        # silently reused the previous category's. Keeping it explicit here
        # preserves behaviour and makes the fix a one-line deletion.
        self._carry_over: Optional[List] = None

    # -- main loop ----------------------------------------------------------

    def run(self) -> Tuple[List, Dict[str, Dict[str, int]]]:
        for category, frame in self.context.photos.groupby(Col.CLUSTER_CONTEXT):
            self._run_category(category, frame)
        return self.chosen, self.per_category

    def _run_category(self, category: str, frame: pd.DataFrame) -> None:
        available = len(frame)
        self.per_category.setdefault(category, {})
        self.per_category[category]['actual'] = available

        # `actual` stays the count the gallery has; the preselected photos come
        # off the working frame so they are not weighed against themselves.
        # Their allowance was already charged in `select.preselect`.
        if self.committed:
            settled = frame[Col.IMAGE_ID].isin(self.committed)
            if settled.any():
                entry = self.per_category[category]
                entry['selected'] = entry.get('selected', 0) + int(settled.sum())
                frame = frame[~settled]
                if frame.empty:
                    return

        need = self.plan.images[category]

        # A category with one slot and almost nothing in it is not worth
        # scoring.
        if available <= 2 and need == 1:
            self._take(category, frame[Col.IMAGE_ID].values.tolist()[:need])
            return

        if need == 0:
            return

        scored, candidate_ids, used_image_order = self.gate.candidates(frame, category)
        if scored is None or len(candidate_ids) == 0:
            return

        if category in USER_PREFERENCE_CATEGORIES:
            self._pick_user_preference(category, candidate_ids, need)
            return

        user_ids, remaining_ids = self._split_out_user_picks(candidate_ids)
        if user_ids:
            # Note: the user's picks are added on top of the allowance rather
            # than out of it — `need` was already read above.
            self.chosen.extend(user_ids)
            self.per_category[category]['selected'] = len(user_ids)
            self.plan.images[category] -= len(user_ids)

        scored = scored[scored[Col.IMAGE_ID].isin(remaining_ids)]
        scored = narrowing.add_timestamps(scored)

        valid = narrowing.drop_temporal_orphans(scored, need, self.logger)
        if valid.empty:
            self.logger.info(f"There are no images to select for {category}")
            return

        pool = narrowing.add_time_clusters(valid, self.logger)
        color, grayscale = narrowing.split_by_color(pool)

        if len(color) == 0 and len(grayscale) > 0:
            self._take_grayscale_only(category, grayscale)
            return

        order_index = narrowing.order_index(valid, scored=not used_image_order)

        if len(pool) <= need:
            self._take_all_distinct(category, valid, need)
            return

        request = CategoryRequest(
            category=category,
            need=need,
            color=color,
            grayscale=grayscale,
            pool=pool,
            scored=not used_image_order,
            order_index=order_index,
            user_selected=self.user_selected,
            is_artificial_time=self.context.facts.is_artificial_time,
            logger=self.logger,
        )

        picks = self.strategies.for_category(category).pick(request)

        if picks.forced:
            self.chosen.extend(picks.forced)
        if picks.remaining_need is not None:
            need = picks.remaining_need
        if picks.skip:
            return

        preferred = picks.preferred if picks.preferred is not None else self._carry_over
        if preferred is None:
            raise UnboundLocalError(
                f"no ranked list for {category!r} and no earlier category left one behind"
            )
        self._carry_over = preferred

        self._take(category, self._top_up_with_grayscale(category, preferred, grayscale, need))

    # -- shared steps -------------------------------------------------------

    def _split_out_user_picks(self, candidate_ids) -> Tuple[List, List]:
        picked = set(self.user_selected[Col.IMAGE_ID].values)
        user_ids = [image_id for image_id in candidate_ids if image_id in picked]
        chosen = set(user_ids)
        return user_ids, [image_id for image_id in candidate_ids if image_id not in chosen]

    def _pick_user_preference(self, category: str, candidate_ids, need: int) -> None:
        picked = set(self.user_selected[Col.IMAGE_ID].values)
        user_ids = [image_id for image_id in candidate_ids if image_id in picked]
        source = user_ids if user_ids else candidate_ids
        self._take(category, source[:need])

    def _take_grayscale_only(self, category: str, grayscale: pd.DataFrame) -> None:
        """No colour in this category, so allow a couple of black-and-whites."""
        count = min(GRAYSCALE_ONLY_CAP, len(grayscale))
        picks = (
            grayscale.sort_values(by='image_order', ascending=True)[Col.IMAGE_ID]
            .values.tolist()[:count]
        )
        self._take(category, picks)
        self.logger.info(
            f"No color candidates for {category}, selected {len(picks)} grayscale images only"
        )

    def _take_all_distinct(self, category: str, valid: pd.DataFrame, need: int) -> None:
        """Supply is at or below demand: take everything, minus near-duplicates
        of the same people doing the same thing."""
        picks = (
            valid.assign(_pid=valid[Col.PERSONS_IDS].apply(tuple))
            .sort_values("image_order", ascending=True)
            .drop_duplicates(subset=["_pid", Col.IMAGE_SUBQUERY_CONTENT], keep="first")
            .head(need)[Col.IMAGE_ID]
            .tolist()
        )
        self._take(category, picks)
        self.logger.info(
            f"it has less than needed so we select them all {category} no filtering"
        )

    def _top_up_with_grayscale(
        self, category: str, preferred: List, grayscale: pd.DataFrame, need: int
    ) -> List:
        """Fill a colour shortfall from the greyscale pool."""
        if len(preferred) >= need:
            return preferred[:need]

        gap = need - len(preferred)
        if gap <= 0 or grayscale.empty:
            return list(preferred)

        if len(grayscale) <= gap:
            filler = grayscale[Col.IMAGE_ID].tolist()
        elif gap == 1:
            filler = (
                grayscale.sort_values('total_score', ascending=False)
                .head(1)[Col.IMAGE_ID].tolist()
            )
        else:
            filler = select_remove_similar(
                self.context.facts.is_artificial_time,
                need=gap,
                df=grayscale.reset_index(),
                cluster_name=category,
                logger=self.logger,
                target_group_size=10,
            )

        return list(preferred) + list(filler)

    def _take(self, category: str, picks) -> None:
        picks = list(picks)
        self.chosen.extend(picks)
        entry = self.per_category[category]
        entry['selected'] = entry.get('selected', 0) + len(picks)
