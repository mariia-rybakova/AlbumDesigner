"""Constraints: the photos the album has already committed to.

Some photos are in the album because something decided so before any ranking
happened, and running them through `select.pick` either adds nothing or risks
losing them. This substage settles those and hands `select.pick` a shorter job.

Four kinds, in the order they are honoured -- the user's own intent first, so
that a later rule finds its slot already filled rather than competing for it:

``user_picks``
    Photos from ``aiMetadata.photoIds``. The picker did honour these, but only
    the ones that survived scoring and the candidate cut first: a photo the user
    chose by hand could be scored below the floor and silently dropped. A hand
    pick is not a candidate, it is a decision.

``identities``
    The user named people in ``aiMetadata.personIds``. Until now that only fed
    ``person_score``, so a requested person could be *ranked up* everywhere and
    still end up in no photo at all. Each named identity is now guaranteed its
    photos. **How many is** ``CONFIGS['preselect']['photos_per_identity']``,
    defaulting to one -- guaranteeing coverage, not saturating the album, which
    is the only reading of "mandatory" that leaves an album to build.

``key_pages``
    The opening and closing photos `enrich.key_pages` chose. They were picked
    over the whole gallery, so nothing guaranteed selection would keep them --
    and ProcessStage takes its covers from the selected pool. Committing the
    first still-available candidate from each ranked list is what those lists
    were made for.

``yes_categories``
    Every category the focus profile budgets as `yes` rather than a percentage:
    the ring shot, the invitation, the dress, and the ceremony highlights. A
    `yes` category is promised one photo if the thing happened, so there is no
    allowance to divide and nothing for the ranked picker to weigh. Taking the
    best and moving on is the whole decision.

Each is separately switchable under ``CONFIGS['preselect']``, so a constraint
that turns out to cost more than it is worth is one line to disable.

Ranking, where a choice remains, is the same one `select.pick` would apply: the
request's own scoring when it has anything to score against, and ``image_order``
ascending when it does not. Note the direction -- ``image_order`` is the content
model's ``selectionOrder``, a rank where **0 is best**, which is why
``update_photos_ranks`` sets a hand-picked photo to 0.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.pipeline.contracts import AlbumContext, Col, ctx, photo
from src.pipeline.registry import register
from src.pipeline.select.scoring import Scorer
from src.pipeline.substage import SubStage
from utils.configs import CONFIGS


@register
class PreselectSubStage(SubStage):
    """Commit the photos that are decided before ranking.

    Wedding-only, like `select.budget`: it spends the per-category allowance
    that substage produces, and the non-wedding path has none.
    """

    name = "select.preselect"
    requires = frozenset({photo(Col.IMAGE_ID), photo(Col.IMAGE_ORDER),
                          ctx("selection_plan")})

    def applies_to(self, context: AlbumContext) -> bool:
        return bool(context.facts.is_wedding) and not context.selection.manual

    def execute(self, context: AlbumContext) -> AlbumContext:
        preselector = Preselector(context)
        committed = preselector.run()

        context.selection_plan.committed = committed
        if context.logger:
            context.logger.info(f"Preselected {len(committed)} photos: {preselector.summary()}")
        return context


class Preselector:
    """One pass over the constraints."""

    def __init__(self, context: AlbumContext):
        self.context = context
        self.logger = context.logger
        self.plan = context.selection_plan
        self.inputs = context.selection_inputs
        self.pool = context.photos

        self.user_selected = self.pool[
            self.pool[Col.IMAGE_ID].isin(self.inputs.user_selected_ids)
        ]
        self.scorer = Scorer(
            self.user_selected,
            self.inputs.person_ids,
            self.inputs.tags_features,
            self.inputs.ratings,
            self.logger,
        )

        #: {image_id: reason}
        self.committed: Dict[Any, str] = {}
        self._settings = CONFIGS['preselect']

    # -- main ---------------------------------------------------------------

    def run(self) -> Dict[Any, str]:
        if self._settings.get('user_picks', True):
            self._commit_user_picks()
        if self._settings.get('identities', True):
            self._commit_identity_coverage()
        if self._settings.get('key_pages', True):
            self._commit_key_pages()
        if self._settings.get('yes_categories', True):
            self._commit_yes_categories()
        return self.committed

    def summary(self) -> str:
        counts: Dict[str, int] = {}
        for reason in self.committed.values():
            kind = reason.split(':', 1)[0]
            counts[kind] = counts.get(kind, 0) + 1
        return ", ".join(f"{kind}={count}" for kind, count in sorted(counts.items())) or "none"

    # -- the four constraints ----------------------------------------------

    def _commit_user_picks(self) -> None:
        """Every hand-picked photo still in the pool, unconditionally.

        Not charged to any category's allowance -- see :meth:`_commit`.
        """
        for image_id in self.user_selected[Col.IMAGE_ID].tolist():
            self._commit(image_id, "user")

    def _commit_identity_coverage(self) -> None:
        """Each named identity gets its photos, whatever the ranking says."""
        wanted = self._settings.get('photos_per_identity', 1)
        for identity in self.inputs.person_ids or []:
            gap = wanted - len(self._committed_with(identity))
            if gap <= 0:
                continue
            candidates = self._with_identity(self._remaining(), identity)
            for image_id in self._best_of(candidates, gap):
                self._commit(image_id, f"identity:{identity}")

    def _commit_key_pages(self) -> None:
        """The opening and closing photos, taking the best still available."""
        key_pages = self.context.key_pages
        if key_pages is None:
            return

        in_pool = set(self.pool[Col.IMAGE_ID])
        for role, ranked in (("opening", key_pages.opening), ("closing", key_pages.closing)):
            for image_id in ranked:
                if image_id in in_pool and self._commit(image_id, f"key_page:{role}"):
                    break

    def _commit_yes_categories(self) -> None:
        """Fill each `yes` category's whole allowance and be done with it.

        An earlier constraint may already have taken photos from here -- a
        hand-picked ring shot settles the ring category, and asking for another
        on top would give it two photos where it is allowed one. So only the gap
        is filled, and the allowance is then zeroed either way: the category is
        settled, and `select.pick` has nothing left to decide for it.
        """
        for category in self.plan.yes_categories:
            need = self.plan.images.get(category, 0)
            if need <= 0:
                continue

            frame = self._remaining()
            frame = frame[frame[Col.CLUSTER_CONTEXT] == category]
            for image_id in self._best_of(frame, need - self._committed_in(category)):
                self._commit(image_id, f"yes:{category}")

            self.plan.images[category] = 0

    # -- shared -------------------------------------------------------------

    def _commit(self, image_id: Any, reason: str) -> bool:
        """Record a photo. Returns False if it was already committed.

        No category's allowance is decremented here. A `yes` category is settled
        wholesale by :meth:`_commit_yes_categories`, which zeroes it; the other
        three constraints are *added* to the budget rather than taken out of it.

        Charging them was the first thing tried and it made the album shorter
        rather than more certain, because a committed photo is usually one the
        picker would have chosen anyway -- charging its category then costs a
        second photo for nothing. Measured on the equivalence fixture, charging
        identity coverage lost a `walking the aisle` frame that *both* paths had
        already selected. Left uncharged, a constraint costs the album a slot
        only when it actually adds a photo, which is also the rule the original
        applied to hand-picked photos.
        """
        if image_id in self.committed:
            return False
        self.committed[image_id] = reason
        return True

    def _remaining(self) -> pd.DataFrame:
        return self.pool[~self.pool[Col.IMAGE_ID].isin(self.committed)]

    def _committed_frame(self) -> pd.DataFrame:
        return self.pool[self.pool[Col.IMAGE_ID].isin(self.committed)]

    def _committed_in(self, category: str) -> int:
        frame = self._committed_frame()
        return int((frame[Col.CLUSTER_CONTEXT] == category).sum())

    def _committed_with(self, identity) -> pd.DataFrame:
        return self._with_identity(self._committed_frame(), identity)

    @staticmethod
    def _with_identity(frame: pd.DataFrame, identity) -> pd.DataFrame:
        if frame.empty or Col.PERSONS_IDS not in frame.columns:
            return frame.iloc[0:0]
        return frame[frame[Col.PERSONS_IDS].apply(
            lambda ids: isinstance(ids, (list, tuple, set)) and identity in ids)]

    def _best_of(self, frame: pd.DataFrame, count: int) -> List[Any]:
        """The ``count`` best photos of a frame, ranked as the picker would.

        Falls back to ``image_order`` when there is nothing to score against, or
        when scoring fails -- the picker treats a scoring failure as a reason to
        skip the category, but a constraint that quietly goes unmet is worse
        than one met by the gallery's own ranking.
        """
        if frame.empty or count <= 0:
            return []

        if not self.inputs.unscored:
            try:
                ranked, _scored = self.scorer.score(frame)
                if ranked:
                    by_score = sorted(ranked, key=lambda pair: pair[1], reverse=True)
                    return [image_id for image_id, _ in by_score][:count]
            except Exception as exc:  # noqa: BLE001 - fall back, do not drop the constraint
                if self.logger:
                    self.logger.warning(f"Preselect could not score {len(frame)} photos: {exc}")

        return (
            frame.sort_values(Col.IMAGE_ORDER, ascending=True)[Col.IMAGE_ID]
            .tolist()[:count]
        )
