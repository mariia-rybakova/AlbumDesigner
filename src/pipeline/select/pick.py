"""The picker: walk the categories and let each one's strategy choose.

The driver owns everything that is the same for every category — scoring,
gating, honouring the user's own picks, temporal narrowing, the colour split and
the greyscale top-up. What differs per category lives in a
:class:`~src.pipeline.select.strategies.base.CategoryStrategy`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.pipeline.contracts import AlbumContext, Col, photo
from src.pipeline.registry import register
from src.pipeline.select import cpsat, narrowing
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
        if context.selection.manual:
            return False
        # `select.narrator` composes selection and grouping together, so when it
        # answered there is nothing left to choose. It declines far more often
        # than it answers (weddings, v1 embeddings, no checkpoint), and then this
        # runs as before.
        return context.predefined is None

    def execute(self, context: AlbumContext) -> AlbumContext:
        if context.facts.is_wedding:
            chosen, per_category = self._pick_wedding(context)
            context.selection.per_category = per_category
        else:
            chosen, error = smart_non_wedding_selection(context.photos, logger=context.logger)
            if error:
                return context.fail(error)

        context.logger.info(f"Total images: {len(chosen or [])}")
        # De-duplicate while keeping the order categories contributed in.
        context.selection.photo_ids = list(dict.fromkeys(chosen or []))
        return context

    def _pick_wedding(self, context: AlbumContext) -> Tuple[List, Dict[str, Dict[str, int]]]:
        """The per-category loop, or the one-shot solve when it is switched on.

        `cpsat` states the same problem as a single constrained optimisation
        rather than a sequence of independent category decisions. It is off by
        default, and anything at all going wrong in there -- ortools absent, no
        solution inside the time limit, a modelling mistake -- falls back to the
        loop, so the album is never the casualty of an experiment.
        """
        if cpsat.is_enabled():
            try:
                result = cpsat.CpSatPicker(context).run()
                if result is not None:
                    return result
                context.logger.warning("cp-sat returned nothing; falling back to the loop")
            except Exception as exc:  # noqa: BLE001 - an experiment must not lose the album
                context.logger.error(f"cp-sat failed ({type(exc).__name__}: {exc}); "
                                     f"falling back to the loop")

        return WeddingPicker(context, self.strategies).run()


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
        self.gate = CandidateGate(scorer, self.inputs.unscored, self.plan.images, self.logger,
                                  bride_id=context.facts.bride_id,
                                  groom_id=context.facts.groom_id)

        # Whatever `select.preselect` committed is already in the album; the
        # loop below starts from it and does not offer those photos again.
        self.committed: Dict = dict(self.plan.committed)
        self.chosen: List = list(self.committed)
        #: Who the committed photos already put in the album. Empty when
        #: nothing was committed, so the picker behaves exactly as before.
        settled_rows = context.photos[context.photos[Col.IMAGE_ID].isin(self.committed)]
        self.covered_people: set = _people_in(settled_rows)
        self.covered_embeddings: List = _unit_embeddings(settled_rows)
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
                # Tracked apart from `selected` because their allowance was
                # charged in `select.preselect`: measuring what the picker
                # itself chose against `need` means subtracting these first.
                entry['committed'] = int(settled.sum())
                entry['selected'] = entry.get('selected', 0) + int(settled.sum())
                frame = frame[~settled]
                if frame.empty:
                    self._note(category, 'all_committed')
                    return

        need = self.plan.images[category]
        self.per_category[category]['need'] = need

        # A category with one slot and almost nothing in it is not worth
        # scoring.
        if available <= 2 and need == 1:
            self._take(category, frame[Col.IMAGE_ID].values.tolist()[:need])
            self._note(category, 'tiny_shortcut')
            return

        if need == 0:
            self._note(category, 'no_allowance')
            return

        scored, candidate_ids, used_image_order = self.gate.candidates(frame, category)
        if scored is None or len(candidate_ids) == 0:
            self._note(category, 'gate_declined')
            return

        if category in USER_PREFERENCE_CATEGORIES:
            self._pick_user_preference(category, candidate_ids, need)
            self._note(category, 'user_preference')
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
            self._note(category, 'temporal_narrowing')
            return

        pool = narrowing.add_time_clusters(valid, self.logger)
        color, grayscale = narrowing.split_by_color(pool)

        if len(color) == 0 and len(grayscale) > 0:
            self._take_grayscale_only(category, grayscale)
            self._note(category, 'greyscale_only')
            return

        order_index = narrowing.order_index(valid, scored=not used_image_order)

        if len(pool) <= need:
            self._take_all_distinct(category, valid, need)
            self._note(category, 'take_all_distinct')
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
            covered_people=self.covered_people,
            covered_embeddings=self.covered_embeddings,
            is_artificial_time=self.context.facts.is_artificial_time,
            logger=self.logger,
        )

        strategy = self.strategies.for_category(category)
        self.per_category[category]['strategy'] = type(strategy).__name__
        picks = strategy.pick(request)

        if picks.forced:
            self.chosen.extend(picks.forced)
            # These reach the album -- `walking the aisle`'s two scripted beats
            # are forced picks -- so they are the category's photos and belong
            # in its tally. Counting them only in `self.chosen` made the class
            # read as short of an allowance it had in fact spent.
            entry = self.per_category[category]
            entry['selected'] = entry.get('selected', 0) + len(picks.forced)
        if picks.remaining_need is not None:
            need = picks.remaining_need
        if picks.skip:
            self._note(category, 'strategy_skip')
            return

        preferred = picks.preferred if picks.preferred is not None else self._carry_over
        if preferred is None:
            raise UnboundLocalError(
                f"no ranked list for {category!r} and no earlier category left one behind"
            )
        if picks.preferred is None:
            # The monolith's carry-over accident, reproduced in ..pick. Marked
            # so a weight fit against the loop can exclude the class: this is
            # behaviour worth matching only until someone deletes it.
            self._note(category, 'carry_over')
        self._carry_over = preferred

        self._take(category, self._top_up_with_grayscale(category, preferred, grayscale, need))
        # Not `_note`: it records the first mechanism to settle a category, and
        # a carry-over above already claimed this one.
        self.per_category[category].setdefault('bound_by', 'strategy')

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
        """Fill a colour shortfall from the greyscale pool.

        Note that a frame and the black-and-white copy of itself are never
        compared here -- the colour and greyscale pools are filled
        independently. `enrich.duplicate_shots` is what stops the second copy
        ever reaching selection.
        """
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
                already_selected=self.covered_embeddings,
            )

        return list(preferred) + list(filler)

    def _take(self, category: str, picks) -> None:
        picks = list(picks)
        self.chosen.extend(picks)
        entry = self.per_category[category]
        entry['selected'] = entry.get('selected', 0) + len(picks)

    def _note(self, category: str, mechanism: str) -> None:
        """Record which decision point settled this category.

        Diagnostic only -- nothing reads `per_category`. It exists because the
        loop's pick count is bound by one of a dozen different things and the
        difference between them is invisible from the outside: a class can come
        back short because the gate declined it, because temporal narrowing
        emptied it, because `_take_all_distinct` deduplicated it, or because a
        strategy saturated. `docs/cpsat_scoring_plan.md` is an argument about
        exactly that distinction, and Phase 0 of it is this line.

        First writer wins, so the mechanism named is the one that settled the
        category rather than whatever ran last.
        """
        self.per_category.setdefault(category, {}).setdefault('bound_by', mechanism)


def _people_in(frame: pd.DataFrame) -> set:
    """Every identity appearing anywhere in a frame."""
    if frame is None or frame.empty or Col.PERSONS_IDS not in frame.columns:
        return set()
    people: set = set()
    for ids in frame[Col.PERSONS_IDS]:
        if isinstance(ids, (list, tuple, set)):
            people.update(ids)
    return people


def _unit_embeddings(frame: pd.DataFrame) -> List:
    """L2-normalised embeddings of a frame, skipping rows without one."""
    if frame is None or frame.empty or Col.EMBEDDING not in frame.columns:
        return []
    vectors = []
    for value in frame[Col.EMBEDDING]:
        if value is None:
            continue
        vector = np.asarray(value, dtype=float).ravel()
        norm = float(np.linalg.norm(vector))
        if vector.size and norm:
            vectors.append(vector / norm)
    return vectors
