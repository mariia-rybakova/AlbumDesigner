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

from typing import Any, Dict, List, Optional

import pandas as pd

from src.pipeline.contracts import AlbumContext, Col, ctx, photo
from src.pipeline.registry import register
from src.pipeline import subject
from src.pipeline.select.scoring import Scorer
from src.pipeline.select.strategies import default_registry
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

        #: The couple, so a category can be narrowed to the photos it is about.
        self._bride_id = context.facts.bride_id
        self._groom_id = context.facts.groom_id

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

        **Charged to its own category.** A photo the user picked out of the
        dancing spends one of dancing's slots, not one of some other category's
        and not nothing at all. Added on top -- which is what the monolith did
        and what this did at first -- the album grows by however many photos
        were picked, so the requested density stops meaning anything: on one
        gallery a density-4 album budgeted 139 photos and then carried 36 more.

        The charge is per photo and by class, so a category the user did not
        touch keeps its full allowance.
        """
        for image_id in self.user_selected[Col.IMAGE_ID].tolist():
            if self._commit(image_id, "user"):
                self._charge(image_id)

    def _commit_identity_coverage(self) -> None:
        """Each named identity gets its photos, whatever the ranking says.

        Ranked within a preference over *classes* first. Any photo of the person
        satisfies the guarantee, so score alone used to decide it -- and on
        49995684 that put two `other` photos in the album to cover identities 58
        and 71, a class budgeted at 0% precisely because it carries nothing
        worth a spread. `pipeline.subject.identity_tiers` prefers a real class,
        and one the couple is not the subject of, falling through only when the
        person appears nowhere better.
        """
        wanted = self._settings.get('photos_per_identity', 1)
        for identity in self.inputs.person_ids or []:
            gap = wanted - len(self._committed_with(identity))
            if gap <= 0:
                continue
            candidates = self._with_identity(self._remaining(), identity)
            for image_id in self._best_by_tier(candidates, gap, identity):
                self._commit(image_id, f"identity:{identity}")

    def _best_by_tier(self, frame: pd.DataFrame, count: int, identity: Any) -> List[Any]:
        """``_best_of``, but exhausting each class tier before the next."""
        chosen: List[Any] = []
        for tier, pool in enumerate(subject.identity_tiers(frame), start=1):
            if len(chosen) >= count or pool is None or pool.empty:
                continue
            picked = self._best_of(pool, count - len(chosen))
            if picked and tier > 1 and self.logger:
                self.logger.info(
                    f"Preselect: identity {identity} covered from tier {tier} "
                    f"(no photo of them in a better class)")
            chosen.extend(picked)
        return chosen

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

        **Except where it does.** The whole premise for resolving a `yes`
        category here is that the ranked picker adds nothing: one photo of the
        rings is one photo of the rings. That is false wherever a category has a
        strategy of its own. `getting hair-makeup` carries `yes` *and* is handled
        by `BridePrepStrategy`, whose entire job is to keep the bride and keep
        the same bride -- and taking it here zeroed the allowance so that
        strategy never ran. On gallery 53459898 the photo that reached the album
        contained `persons=[6]`, someone unrelated to the couple, chosen on rank
        alone. Those categories are left to the picker.
        """
        handled = set(default_registry().categories())

        for category in self.plan.yes_categories:
            if category in handled:
                if self.logger:
                    self.logger.debug(
                        f"Leaving '{category}' to its own strategy rather than "
                        f"resolving it here")
                continue

            need = self.plan.images.get(category, 0)
            if need <= 0:
                continue

            frame = self._remaining()
            frame = frame[frame[Col.CLUSTER_CONTEXT] == category]
            # A `yes` category gets exactly one photo and no second chance, so
            # it is the worst place to rank a frame that does not hold the
            # subject. `may kiss bride` on 49995684 went to the only one of six
            # frames with no faces in it.
            frame = subject.prefer_subject(frame, category, self._bride_id, self._groom_id,
                                           self.logger)
            for image_id in self._best_of(frame, need - self._committed_in(category)):
                self._commit(image_id, f"yes:{category}")

            self.plan.images[category] = 0

    # -- shared -------------------------------------------------------------

    def _charge(self, image_id: Any) -> None:
        """Take one photo out of the allowance of the class it belongs to."""
        category = self._category_of(image_id)
        if category is not None and category in self.plan.images:
            self.plan.images[category] = max(0, self.plan.images[category] - 1)

    def _category_of(self, image_id: Any) -> Optional[str]:
        row = self.pool.loc[self.pool[Col.IMAGE_ID] == image_id, Col.CLUSTER_CONTEXT]
        return None if row.empty else row.iloc[0]

    def _commit(self, image_id: Any, reason: str) -> bool:
        """Record a photo. Returns False if it was already committed.

        No allowance is decremented here; the caller decides. A hand pick is
        charged to its own class (:meth:`_charge`) because it is a photo the
        album is spending a slot on. A `yes` category is settled wholesale by
        :meth:`_commit_yes_categories`, which zeroes it.

        Identity coverage and the covers are **not** charged. Those are
        guarantees rather than choices, and charging them made the album shorter
        rather than more certain: a committed photo is usually one the picker
        would have chosen anyway, so charging its category cost a second photo
        for nothing -- on the equivalence fixture it lost a `walking the aisle`
        frame that *both* paths had already selected.
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
