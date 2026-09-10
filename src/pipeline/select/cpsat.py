"""One global CP-SAT model instead of a per-category loop.

Reference for every mechanism here, and for the five that were built and
measured and left inert: `docs/cpsat_picker.md`.

`WeddingPicker` decides one category at a time: score it, gate it, narrow it,
hand it to a strategy. Each category therefore chooses without knowing what the
others chose, and the only thing spreading picks across the day is the diversity
pass inside a category. This module states the whole thing as one constrained
optimisation over the whole gallery, so coverage of the day, the size of the
runs, the per-class quotas and the ranks are traded off against each other
rather than in sequence.

The formulation follows *Algorithms for Constrained Sequence Selection*:

    x_i in {0,1}                          one variable per photo
    sum(class c) + shortage_c == Q_c      quota, with a penalised slack
    x_i == 1                              for everything already committed
    per-window deviation penalties        coverage of the day ("representation")
    pair_i <= x_i, pair_i <= x_j          rewarded, so picks form small runs
    maximise  ranks + cohesion - shortage - deviation

Three things the paper does not cover had to be added, or the model produces
albums that are worse in ways the current picker never is:

* **Greyscale is penalised, not pooled.** The driver keeps colour and greyscale
  in separate pools and only tops up from greyscale. A single model has one
  pool, so a per-photo penalty stands in for that preference.
* **Near-duplicates are made mutually exclusive** above a cosine threshold,
  within a class. Cohesion rewards neighbours; without this it happily rewards
  two frames of the same instant. The exclusion also covers the committed
  photos, which is what `select_remove_similar`'s `already_selected` does for
  the loop.
* **The axis is `general_time` rank, not the row index.** The paper assumes
  position == list index. Here the sequence is the day, which is
  `enrich.timeline.ordered` -- and on an artificial-time gallery the row order
  and the day order are not the same thing.

**The default picker** (`CONFIGS['pick_cpsat']['enabled']`). Failure of any
kind -- ortools missing, no solution inside the time limit, a modelling error
-- logs and hands back to `WeddingPicker`, so the loop is still the floor
under it; `process_gallery.py --loop` forces that path for a comparison.
`ortools` is imported inside the solve rather than at module scope, so nothing
here is a load-time dependency; it is already installed as a dependency of
`k_means_constrained`, which is why `requirements.txt` needs no change.

Two things were brought into line with the loop after measuring where the two
diverge (`docs/cpsat_scoring_plan.md` §1):

* **Scores are normalised over the free rows only.** `get_scores` min-max
  normalises within the frame it is handed, and the loop scores a class only
  after dropping what `select.preselect` committed. Scoring the whole class
  ranked it against a different population: on 53459898, with 42 photos
  committed, this shifted `total_score` in all 20 affected classes and flipped
  the ranking order in 10 of them.
* **`CandidateGate` decides what is even a variable.** A class it declines
  stays empty, as it does in the loop, and only its shortlist gets a variable.

**Status: unrefined.** The model solves and the constraints hold, but the
albums it produces are not yet better than the loop's -- the weights are
untuned starting points and coverage is dominated by whatever
`select.preselect` committed.

The remaining divergence is not a missing constraint but a different idea of
what a quota is. The loop treats it as a ceiling its diversity passes routinely
leave unmet -- `_take_all_distinct` deduplicates on people and subquery, the
person-coverage pass stops when a slot adds no new guest, temporal narrowing
can empty a class outright -- while the equality-plus-shortage below treats it
as a target and fills it. That is why the model selects more, and it is what
the coverage-aware scoring in `docs/cpsat_scoring_plan.md` is meant to replace.

Measured on project 53147741 (571 photos, 27 classes, artificial time):
OPTIMAL in 0.04s, every quota met except two classes where the near-duplicate
exclusion outranked the shortage penalty, and 56 of 138 photos in the first of
six windows -- because 65 photos were already committed, 50 of them the user's
own picks, so most of the coverage was decided before the solve.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.pipeline.contracts import AlbumContext, Col
from src.pipeline.enrich import timeline as tl
from src.pipeline.select import narrowing
from src.pipeline.subject import IDENTITY_RULES, _people_of
from src.pipeline.select.scoring import CandidateGate, Scorer
from utils.configs import CONFIGS

#: Objective coefficients must be integers, so every score is scaled by this.
SCORE_SCALE = 1000


def settings() -> Dict[str, Any]:
    return CONFIGS.get('pick_cpsat', {})


def is_enabled() -> bool:
    return bool(settings().get('enabled', False))


class CpSatPicker:
    """Pick the whole album in one solve.

    Same inputs and outputs as :class:`~src.pipeline.select.pick.WeddingPicker`
    -- ``run()`` returns ``(chosen_ids, per_category)`` -- so the driver can
    swap between them without anything downstream noticing.
    """

    def __init__(self, context: AlbumContext):
        self.context = context
        self.logger = context.logger
        self.plan = context.selection_plan
        self.inputs = context.selection_inputs
        self.cfg = settings()

        self.committed: Dict = dict(self.plan.committed)
        self.per_category: Dict[str, Dict[str, int]] = {}
        #: (variables, constraints) of the model actually built, for the log.
        self.model_size: Tuple[int, int] = (0, 0)
        #: Length of the whole day in positions, set by `_pool`. Distinct from
        #: `len(frame)` once the gate has removed rows.
        self.positions: int = 0

    # -- entry point --------------------------------------------------------

    def run(self) -> Optional[Tuple[List, Dict[str, Dict[str, int]]]]:
        """``(chosen, per_category)``, or None when the model cannot answer."""
        from ortools.sat.python import cp_model

        frame = self._pool()
        if frame.empty:
            self.logger.warning("cp-sat: no photos to pick from")
            return None

        model = cp_model.CpModel()
        x = {index: model.NewBoolVar(f"x_{index}") for index in frame.index}

        # Everything `select.preselect` settled is fixed, not re-decided. The
        # point of doing it this way rather than removing those rows: they stay
        # in the window counts and the cohesion pairs, so the solver treats them
        # as anchors and fills around them.
        for index in frame.index[frame['_committed']]:
            model.Add(x[index] == 1)

        penalties: List = []
        rewards: List = []

        self._add_quotas(model, frame, x, penalties)
        if self._coverage_on():
            self._add_coverage(model, frame, x, rewards)
        else:
            self._add_windows(model, frame, x, penalties)
            self._add_spacing(model, frame, x)
        self._add_exclusions(model, frame, x)
        # A committed photo is fixed at 1, so it is never blocked -- the user's
        # own pick is not second-guessed on identity.
        for index in self.contradictions(frame):
            model.Add(x[index] == 0)
        self._add_distinct_shots(model, frame, x)
        self._add_similarity_penalty(model, frame, x, penalties)
        self._add_repeat_penalty(model, frame, x, penalties)
        self._add_cohesion(model, frame, x, rewards)

        # Rank net of what a page costs. Committed photos are fixed at 1, so
        # charging them would only shift the objective by a constant and would
        # make the reported score harder to read -- `_admission_costs` leaves
        # them at zero.
        costs = self._admission_costs(frame)
        bonus = self._identity_bonus(frame)
        ranks = [
            x[index] * (int(round(score * SCORE_SCALE))
                        - int(costs.at[index]) + int(bonus.at[index]))
            for index, score in frame['_score'].items()
        ]
        grey = [
            x[index] * int(self.cfg.get('grayscale_penalty', 200))
            for index in frame.index[frame[Col.IMAGE_COLOR] == 0]
        ]

        model.Maximize(
            int(self.cfg.get('rank_weight', 1)) * sum(ranks)
            + sum(rewards)
            - sum(penalties)
            - sum(grey)
        )

        # Recorded before the solve so the log says what was actually built.
        # The coverage dimensions each add a boolean per (class, bucket), and
        # "how big did that get" is the question a queue worker cares about.
        proto = model.Proto()
        self.model_size = (len(proto.variables), len(proto.constraints))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(self.cfg.get('time_limit_seconds', 30))
        solver.parameters.num_search_workers = int(self.cfg.get('workers', 8))
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            self.logger.error(f"cp-sat: no solution ({solver.StatusName(status)})")
            return None

        chosen = self._collect(frame, x, solver)
        self._report(frame, x, solver, status)
        return chosen, self.per_category

    # -- the pool -----------------------------------------------------------

    def _pool(self) -> pd.DataFrame:
        """The gallery on the day's axis, with a score and a committed flag.

        Scoring is deliberately the *existing* per-category scoring: whatever
        else changes, "which of these two ceremony photos is better" should not,
        or a comparison against the current picker measures two things at once.

        Only the photos the loop would have put in front of a strategy survive:
        committed ones, plus each class's shortlist from `CandidateGate`.
        Everything else is dropped here rather than constrained to zero, so no
        variable is created for it and it counts towards no window.
        """
        photos = self.context.photos
        if photos is None or photos.empty:
            return pd.DataFrame()

        # The day is the axis every constraint here is expressed on, so its
        # absence is not something to work around -- without it "coverage of
        # the day" and "runs of neighbours" mean nothing, and the loop, which
        # needs no such axis, is the better answer.
        missing = [c for c in (Col.GENERAL_TIME, Col.IMAGE_CLASS) if c not in photos.columns]
        if missing:
            self.logger.error(f"cp-sat: no sequence axis -- {missing} absent from the photo "
                              f"table, so there is nothing to spread picks over")
            return pd.DataFrame()

        frame = tl.ordered(photos)
        #: Positions in the whole day. Window edges are cut from this rather
        #: than from whatever survives the gate, or dropping a class's photos
        #: would silently redraw the timeline the other classes are spread over.
        self.positions = len(frame)
        frame['_committed'] = frame[Col.IMAGE_ID].isin(self.committed)

        user_selected = photos[photos[Col.IMAGE_ID].isin(self.inputs.user_selected_ids)]
        scorer = Scorer(user_selected, self.inputs.person_ids, self.inputs.tags_features,
                        self.inputs.ratings, self.logger)
        # `prefer_subject=False`: the same table drives `_identity_bonus` and
        # `contradictions` below, where it can be traded off instead of removing
        # candidates the model never gets to see.
        gate = CandidateGate(scorer, self.inputs.unscored, self.plan.images, self.logger,
                             prefer_subject=False)

        frame['_score'] = 0.0
        frame['_eligible'] = False

        for category, group in frame.groupby(Col.CLUSTER_CONTEXT):
            # `actual` is what the gallery holds, before any of this narrowing.
            self.per_category.setdefault(category, {})['actual'] = len(group)

            # Committed photos are fixed, not re-decided, so they stay whatever
            # the gate thinks of them.
            frame.loc[group.index[group['_committed']], '_eligible'] = True
            settled = int(group['_committed'].sum())
            if settled:
                self.per_category[category]['committed'] = settled

            need = int(self.plan.images.get(category, 0))
            self.per_category[category]['need'] = need

            free = group[~group['_committed']]
            if free.empty:
                self.per_category[category]['bound_by'] = 'all_committed'
                continue
            if need <= 0:
                self.per_category[category]['bound_by'] = 'no_allowance'
                continue

            # Scored over the free rows alone. `get_scores` min-max normalises
            # within the frame it is handed, so including the committed photos
            # would rank this class against a different population than the
            # loop does -- which scores the class only after dropping them.
            frame.loc[free.index, '_score'] = self._scores_for(scorer, free)

            shortlist = self._shortlist(gate, free, category)
            shortlist = self._not_orphans(free.loc[shortlist], category)
            frame.loc[shortlist, '_eligible'] = True
            self.per_category[category]['bound_by'] = (
                'solver' if len(shortlist) else 'gate_declined')

        return frame[frame['_eligible']].copy()

    def _shortlist(self, gate: CandidateGate, free: pd.DataFrame, category: str):
        """The rows of ``free`` the loop would have shown this class's strategy.

        `CandidateGate` is what stands between a category's photos and its
        strategy: it drops a class whose photos do not score at all, and
        otherwise caps the field at ``allocation * 3`` by score. Without it the
        model fills quotas the loop leaves empty -- `cake cutting` and `may
        kiss bride` both went 0 -> 1 on the validation galleries -- and is free
        to pick photos no strategy would ever have been offered.
        """
        try:
            scored, candidate_ids, _used = gate.candidates(free.copy(), category)
        except Exception as exc:  # noqa: BLE001 - the gate is not worth an album
            self.logger.warning(f"cp-sat: candidate gate failed for {category!r} "
                                f"({type(exc).__name__}: {exc}); keeping the whole class")
            return free.index

        if scored is None or not candidate_ids:
            self.logger.info(f"cp-sat: {category!r} has no candidates worth scoring, "
                             f"so it stays empty -- as it would in the loop")
            return free.index[[False] * len(free)]

        return free.index[free[Col.IMAGE_ID].isin(set(candidate_ids))]

    def _not_orphans(self, rows: pd.DataFrame, category: str):
        """Drop the temporally isolated photos, as `select.pick` does.

        A frame with no neighbour within twenty minutes is almost always an
        outlier rather than part of a moment worth a spread. The loop applies
        this to every class through `narrowing.drop_temporal_orphans`; the model
        had no equivalent, and nothing else in it can express one -- Phase 3
        established that coverage cannot, because an isolated photo is its own
        bucket in every dimension and coverage therefore *rewards* taking it.
        So it is a hard eligibility rule, the same shape as the distinct-shot
        and identity exclusions.

        Missing this cost `bride`, `first dance` and `speech` on 52282159: four
        isolated photos the loop rejects went into the album, and restraint
        scored 0 of 4. It went unnoticed for so long because on every other
        validation gallery temporal narrowing binds at most one class -- and on
        that one it was `may kiss bride`, which became a `yes` class and stopped
        reaching the picker at all.

        The loop's own escape hatch is kept: a pool thinner than
        ``need * NARROW_HEADROOM`` is left alone, because there is no room to be
        picky. So is its scope -- only the free rows are judged, since a
        committed photo is in the album whatever it neighbours.
        """
        if not self.cfg.get('temporal_narrowing', {}).get('enabled', False):
            return rows.index
        if rows.empty:
            return rows.index

        need = int(self.plan.images.get(category, 0))
        try:
            timed = narrowing.add_timestamps(rows.copy())
            kept = narrowing.drop_temporal_orphans(timed, need, self.logger)
        except Exception as exc:  # noqa: BLE001 - a narrowing failure is not an album
            self.logger.warning(f"cp-sat: temporal narrowing failed for "
                                f"{category!r} ({type(exc).__name__}: {exc}); "
                                f"keeping the class")
            return rows.index

        # Map back by `image_id`, never by index. `identify_temporal_clusters`
        # resets the index, so its labels no longer refer to the rows they came
        # from: taking `kept.index` selected 1000-1010 where the survivors were
        # 1001-1011 -- the right *number* of photos and the wrong ones, which no
        # count-based check would have caught.
        survivors = set(kept[Col.IMAGE_ID]) if Col.IMAGE_ID in kept.columns else set()
        keep = rows.index[rows[Col.IMAGE_ID].isin(survivors)]

        dropped = len(rows) - len(keep)
        if dropped:
            self.logger.debug(f"cp-sat: {category!r} dropped {dropped} "
                              f"temporally isolated photos")
        return keep

    def _scores_for(self, scorer: Scorer, group: pd.DataFrame) -> pd.Series:
        """A 0..1 desirability per photo, by the same rules the loop uses."""
        if self.inputs.unscored:
            # Nothing to score against, so `image_order` decides -- 0 is best,
            # so invert it into a descending desirability the objective can add.
            order = group[Col.IMAGE_ORDER].rank(method='first', ascending=True)
            return 1.0 - (order - 1) / max(len(group), 1)

        _ranked, scored = scorer.score(group.copy())
        if scored is None or Col.TOTAL_SCORE not in scored.columns:
            self.logger.warning("cp-sat: scoring failed for a category, falling back to rank")
            return pd.Series(0.5, index=group.index)
        return scored[Col.TOTAL_SCORE].reindex(group.index).fillna(0.0)

    # -- constraints --------------------------------------------------------

    def _add_quotas(self, model, frame, x, penalties) -> None:
        """`sum(class) <= Q_c` — a ceiling, with a penalised floor where one is real.

        The allowance from `select.budget` is what the lookup table sizes
        spreads against, so overshooting it is not a free win; but the loop
        does not treat it as a target either. Measured over 60 classes, the
        loop comes in **11 photos short of its allowance and never once over**,
        because its diversity passes return fewer items than they were asked
        for. An equality with a penalised slack cannot express that: the
        shortage weight sits far above every other term, so the model always
        fills.

        A floor is kept only where one is real. A `yes` class is promised a
        photo when the thing happened at all, and coming back empty is a
        failure rather than restraint; everywhere else, stopping early is the
        behaviour being reproduced.

        A ceiling on its own changes nothing -- Phase 2 established that the
        hard way. While every admitted photo earns a flat positive rank, more
        photos is always better and the solve fills to the ceiling regardless.
        The admission cost in `run()` is the other half.
        """
        weight = int(self.cfg.get('shortage_weight', 4000))
        floors = set(getattr(self.plan, 'yes_categories', ()) or ())

        for category, group in frame.groupby(Col.CLUSTER_CONTEXT):
            need = int(self.plan.images.get(category, 0))
            free = group.index[~group['_committed']]
            if need <= 0:
                # No allowance left: nothing beyond what is already committed.
                for index in free:
                    model.Add(x[index] == 0)
                continue

            if len(free) == 0:
                # The gate emptied this class, so the shortage would be a
                # constant `need` in the objective and the quota would say
                # nothing. Leaving it out is what makes the class stay empty
                # rather than merely expensive.
                continue

            picked = sum(x[index] for index in free)

            if not self._ceiling_on():
                shortage = model.NewIntVar(0, need, f"shortage_{_slug(category)}")
                model.Add(picked + shortage == need)
                penalties.append(shortage * weight)
                continue

            model.Add(picked <= need)
            if category in floors:
                shortage = model.NewIntVar(0, need, f"shortage_{_slug(category)}")
                model.Add(picked + shortage >= need)
                penalties.append(shortage * weight)

    def _ceiling_on(self) -> bool:
        return bool(self.cfg.get('quota_ceiling', {}).get('enabled', False))

    def _identity_bonus(self, frame: pd.DataFrame) -> pd.Series:
        """Who a class is *about*, as a score rather than a filter.

        `bride` means the bride on her own, `bride and groom` means the two of
        them and nobody else, `getting hair-makeup` means the bride. The loop
        says this with hard filters in `CoupleTimelineStrategy` and
        `BridePrepStrategy`; the model said it nowhere, and it shows in the
        album -- a hair-and-makeup spread of someone else, `groom` frames with
        no groom in them.

        **Three-valued, because a wrong identity is not a missing one.** A
        frame whose `persons_ids` is empty is a detection that did not happen;
        a `groom` frame carrying identity 9 is positively the wrong person.
        Scoring both as "not a match" made them equally admissible filler, and
        one of each reached the album. So a match earns the bonus, an empty
        frame stays neutral, and a frame naming someone else is penalised.

        A **preference, not a predicate**, and deliberately. The loop's filters
        are hard, which is why they need `_recover_over_filtering` behind them:
        on a gallery where face detection missed the couple, a hard rule empties
        whole classes. A bonus above `SCORE_SCALE` means a matching photo beats
        any non-matching one on rank, so it decides every class that has
        matches, and a class with none simply falls back to rank -- the same
        recovery, without the special case.
        """
        config = self.cfg.get('identity_preference', {})
        bonus = pd.Series(0, index=frame.index, dtype=int)
        if not config.get('enabled', False):
            return bonus

        # From the frame, as `CategoryRequest.bride_id` does. `resolve_bride_groom`
        # stamps both on every row, and they are there whether or not the caller
        # also put them on `facts` -- which a SELECT-only driver does not.
        bride = _identity_from(frame, Col.BRIDE_ID, self.context.facts.bride_id)
        groom = _identity_from(frame, Col.GROOM_ID, self.context.facts.groom_id)
        if bride is None and groom is None:
            return bonus
        default = int(config.get('weight', 0))
        per_class = config.get('per_class') or {}

        penalty = int(config.get('contradiction_penalty', 0))

        for category, group in frame.groupby(Col.CLUSTER_CONTEXT):
            entry = IDENTITY_RULES.get(category)
            if entry is None:
                continue
            weight = int(per_class.get(category, default))
            rule, subjects, exclusive = entry
            wanted = {i for i in (bride if 'bride' in subjects else None,
                                  groom if 'groom' in subjects else None)
                      if i is not None}

            for index, value in group[Col.PERSONS_IDS].items():
                people = _people_of(value)
                if rule(people, bride, groom):
                    bonus.at[index] = weight
                elif exclusive and people and not (people & wanted):
                    bonus.at[index] = -penalty

        return bonus

    def contradictions(self, frame: pd.DataFrame) -> List:
        """Rows an *exclusive* class cannot use: the identity is clear and it
        is not who the class is about.

        A predicate rather than a preference, unlike the rest of
        `_identity_bonus`, and it has to be. As a penalty it was outvoted: a
        contradicted photo still collects the time-coverage rewards for its
        class and window, +300 and +150, which together beat a 1200 charge once
        the rank is in. It only *looked* sufficient on the validation galleries
        because those classes had unknown-identity frames to fall back on.

        Only for the classes that are definitionally about one person, which is
        what `exclusive` marks. Others -- the party classes, the processional --
        legitimately hold other people, and excluding them there emptied
        `walking the aisle` outright.
        """
        blocked: List = []
        if not self.cfg.get('identity_preference', {}).get(
                'exclude_contradictions', False):
            return blocked

        bride = _identity_from(frame, Col.BRIDE_ID, self.context.facts.bride_id)
        groom = _identity_from(frame, Col.GROOM_ID, self.context.facts.groom_id)
        if bride is None and groom is None:
            return blocked

        for category, group in frame.groupby(Col.CLUSTER_CONTEXT):
            entry = IDENTITY_RULES.get(category)
            if entry is None or not entry[2]:
                continue
            _rule, subjects, _exclusive = entry
            wanted = {i for i in (bride if 'bride' in subjects else None,
                                  groom if 'groom' in subjects else None)
                      if i is not None}
            free = group[~group['_committed']]
            for index, value in free[Col.PERSONS_IDS].items():
                people = _people_of(value)
                if people and not (people & wanted):
                    blocked.append(index)
        return blocked

    def _admission_costs(self, frame: pd.DataFrame) -> pd.Series:
        """What a photo must be worth, per class, before it earns a page.

        A pick has to either be good or bring something new; below the bar it
        is taken only when its coverage makes up the difference. That is what
        turns the coverage dimensions from a reshuffle of a fixed count into a
        reason to stop, and a ceiling without it does nothing at all.

        **The bar is a quantile of the class, not a constant.** `get_scores`
        min-max normalises within each class, so every class has a photo at 1.0
        and one at 0.0 and a score of 0.4 means something different in each. A
        single global cost is therefore incomparable across classes, and
        measurably so: sweeping one over 53459898 left the count at 143 for
        every value from 0 to 300 and then dropped it to 120 at 400 -- a cliff
        where whole classes fall under the bar together, rather than a
        gradient. Against the class's own distribution, a quantile of 0.5 means
        the same thing everywhere: better than half your class, or bring
        something new.
        """
        costs = pd.Series(0, index=frame.index, dtype=int)
        if not self._ceiling_on():
            return costs

        settings_for = self.cfg.get('quota_ceiling', {})
        quantile = float(settings_for.get('admission_quantile', 0.0))
        flat = int(settings_for.get('admission_cost', 0))

        for category, group in frame.groupby(Col.CLUSTER_CONTEXT):
            free = group.index[~group['_committed']]
            if not len(free):
                continue

            # A class with no slack is not ranked at all. The loop reaches
            # `_take_all_distinct` down the supply <= demand branch and takes
            # everything bar the repeated shots, without consulting a score --
            # so charging a page bar here drops photos it would have kept. It
            # cost `entertainment` and `kiss` one distinct shot each before
            # this: two keys apiece, the loop keeping both, the model one.
            need = int(self.plan.images.get(category, 0))
            if 0 < need and len(free) <= need:
                continue

            bar = flat
            if quantile > 0:
                scores = frame.loc[free, '_score']
                bar = max(bar, int(round(float(scores.quantile(quantile))
                                         * SCORE_SCALE)))
            costs.loc[free] = bar
        return costs

    # -- coverage (Phase 1 of docs/cpsat_scoring_plan.md) -------------------

    def _coverage_on(self) -> bool:
        return bool(self.cfg.get('coverage', {}).get('enabled', False))

    def coverage_weight(self, category, dimension: str) -> int:
        """``w[class][dimension]``, falling back to the dimension's default.

        The per-class table is the artefact the plan is really about; Phase 1
        ships the lookup with only a `time` row and a couple of overrides, and
        Phase 5 fits the rest. Keeping the shape now means later phases add
        rows rather than rework the call sites.
        """
        settings_for = self.cfg.get('coverage', {}).get(dimension, {})
        per_class = settings_for.get('per_class') or {}
        if category in per_class:
            return int(per_class[category])
        return int(settings_for.get('weight', 0))

    def _add_coverage(self, model, frame, x, rewards) -> None:
        """Every coverage dimension, through one mechanism.

        The dimensions differ only in what "the same thing" means -- a part of
        the day, a person, a kind of shot -- so they share the machinery and
        differ by a bucketing function and a weight. That is the whole claim of
        `docs/cpsat_scoring_plan.md`: the loop says "do not spend two slots on
        the same thing" four times in four incompatible ways, and it only needs
        saying once.

        Rewarding coverage is not the same as penalising drift from it, which
        is what `_add_windows` did. A deviation penalty measures every window
        against a proportional target, so it pushes picks into windows a class
        barely occupies and keeps pushing after the day is covered. A coverage
        reward is collected once: the first pick in a bucket earns it, the
        second earns nothing, and once the day is covered the remaining slots
        are free to sit where the ranks are best.

        Sparse classes need no separate branch either. A class with two slots
        reaches at most two buckets and takes the best two, which is what
        `_add_spacing`'s forbidden pairs were approximating, and a class that
        should not be spread at all takes a weight of zero.
        """
        self._prepare_windows(frame)
        self._cover(model, frame, x, rewards, 'time', _by_window)
        self._cover(model, frame, x, rewards, 'people', _by_identity)
        self._cover(model, frame, x, rewards, 'content',
                    _by_column(self.cfg.get('coverage', {})
                               .get('content', {}).get('column',
                                                        Col.IMAGE_SUBQUERY_CONTENT)))
        self._cover(model, frame, x, rewards, 'visual',
                    _by_appearance(float(self.cfg.get('coverage', {})
                                         .get('visual', {})
                                         .get('threshold', 0.9))))

    def _cover(self, model, frame, x, rewards, dimension: str, buckets_of) -> None:
        """One dimension: a boolean per (class, bucket), rewarded once.

        ``buckets_of(rows)`` returns ``{bucket: [index, ...]}``. A bucket a
        photo shares with nothing earns its reward on the first pick and
        nothing after, which is the diminishing return the loop's diversity
        passes produce by hand.

        The same dimension with ``class = ALL`` is the album-wide statement --
        one more application, not a separate mechanism. For `people` that
        global row is the truer one: `PersonCoverageStrategy` exists to
        maximise the guests who appear *somewhere in the album*, and only
        applies it per category because that is the loop it sits in.
        """
        settings_for = self.cfg.get('coverage', {}).get(dimension, {})
        cap = int(settings_for.get('max_buckets', 0))

        for category, group in frame.groupby(Col.CLUSTER_CONTEXT):
            weight = self.coverage_weight(category, dimension)
            if weight <= 0:
                continue
            for bucket, members in _largest(buckets_of(group), cap):
                covered = model.NewBoolVar(
                    f"cov_{dimension}_{_slug(category)}_{_slug(bucket)}")
                # Collectable only if something in the bucket is picked. The
                # reward is positive, so the solver raises it where it can.
                model.Add(covered <= sum(x[index] for index in members))
                rewards.append(covered * weight)

        overall = int(settings_for.get('global_weight', 0))
        if overall > 0:
            for bucket, members in _largest(buckets_of(frame), cap):
                covered = model.NewBoolVar(f"cov_{dimension}_all_{_slug(bucket)}")
                model.Add(covered <= sum(x[index] for index in members))
                rewards.append(covered * overall)

    def _prepare_windows(self, frame) -> None:
        """Cut the day into windows. Edges come from the whole gallery, not
        from what survived the gate, or dropping one class's photos would
        redraw the timeline every other class is spread over."""
        count = int(self.cfg.get('coverage', {}).get('time', {}).get('windows', 6))
        if count < 2:
            frame['_window'] = 0
            return
        edges = np.linspace(0, self.positions or len(frame), count + 1).astype(int)
        frame['_window'] = np.searchsorted(edges[1:-1], frame[tl.POSITION], side='right')

    def _add_windows(self, model, frame, x, penalties) -> None:
        """Coverage of the day, per class and overall.

        Per class, the target for a window is the class's own local density
        (`Q_c * N_cw / N_c`) rather than an even split -- a class that only
        happened in the afternoon cannot be spread over the morning. Sparse
        classes are handled by spacing instead; see `_add_spacing`.
        """
        count = int(self.cfg.get('windows', 6))
        if count < 2:
            return

        edges = np.linspace(0, self.positions or len(frame), count + 1).astype(int)
        frame['_window'] = np.searchsorted(edges[1:-1], frame[tl.POSITION], side='right')

        class_weight = int(self.cfg.get('class_window_weight', 300))
        sparse_max = int(self.cfg.get('sparse_quota', 3))

        for category, group in frame.groupby(Col.CLUSTER_CONTEXT):
            need = int(self.plan.images.get(category, 0))
            if need <= sparse_max or len(group) == 0:
                continue

            for window, members in group.groupby('_window'):
                target = int(round(need * len(members) / len(group)))
                deviation = model.NewIntVar(0, need, f"dev_{_slug(category)}_{window}")
                picked = sum(x[index] for index in members.index)
                model.Add(picked - target <= deviation)
                model.Add(target - picked <= deviation)
                penalties.append(deviation * class_weight)

        # And the album as a whole, so no window is starved by rounding.
        total = int(sum(max(0, int(v)) for v in self.plan.images.values())) + len(self.committed)
        overall_weight = int(self.cfg.get('window_weight', 150))
        for window, members in frame.groupby('_window'):
            target = int(round(total / count))
            deviation = model.NewIntVar(0, max(total, 1), f"dev_all_{window}")
            picked = sum(x[index] for index in members.index)
            model.Add(picked - target <= deviation)
            model.Add(target - picked <= deviation)
            penalties.append(deviation * overall_weight)

    def _add_spacing(self, model, frame, x) -> None:
        """Minimum distance between picks of the same sparse class.

        For a class with two or three slots, proportional window targets round
        to nothing useful, so the paper's second strategy applies: forbid two
        picks of the same class closer than `D_c` positions, where `D_c` comes
        from the class's own extent rather than a constant.
        """
        sparse_max = int(self.cfg.get('sparse_quota', 3))
        fraction = float(self.cfg.get('gap_fraction', 0.5))

        for category, group in frame.groupby(Col.CLUSTER_CONTEXT):
            need = int(self.plan.images.get(category, 0))
            if need <= 0 or need > sparse_max or len(group) < 2:
                continue

            positions = group[tl.POSITION].values
            span = int(positions.max() - positions.min())
            gap = int(span / (need + 1) * fraction)
            if gap < 1:
                continue

            index = list(group.index)
            for a in range(len(index)):
                for b in range(a + 1, len(index)):
                    first, second = index[a], index[b]
                    if abs(int(positions[b] - positions[a])) >= gap:
                        break  # sorted by position: everything later is further
                    # Two committed photos cannot be un-picked, so a hard pair
                    # constraint between them would make the model infeasible.
                    if frame.at[first, '_committed'] and frame.at[second, '_committed']:
                        continue
                    model.Add(x[first] + x[second] <= 1)

    def _add_distinct_shots(self, model, frame, x) -> None:
        """One photo per `(persons_ids, subquery)` in a class with no slack.

        `_take_all_distinct` in the loop: when a class's supply is at or below
        its allowance there is nothing to rank, so it takes everything, minus
        frames showing the same people doing the same thing. That is a *hard*
        rule and no weighting reproduced it -- four attempts are recorded in
        `docs/cpsat_scoring_plan.md` §7. In a three-photo pool every photo is
        its own bucket in every coverage dimension, so coverage rewards taking
        all three; only an exclusion can say two of them are the same shot.

        **The condition is the whole safety of it.** Applied to every class
        this would be catastrophic: every frame in `dancing` holds the same
        couple and carries the same subquery, so a blanket rule would cap the
        class at one photo. The loop only reaches it down the supply <= demand
        branch, and so does this.

        Committed photos are left out, exactly as in the loop -- it deduplicates
        a frame `select.preselect` has already emptied of them. Two committed
        photos sharing a key would also make the constraint infeasible, since
        neither can be given up.
        """
        if not self.cfg.get('distinct_shots', {}).get('enabled', False):
            return
        if Col.IMAGE_SUBQUERY_CONTENT not in frame.columns:
            return

        for category, group in frame.groupby(Col.CLUSTER_CONTEXT):
            free = group[~group['_committed']]
            need = int(self.plan.images.get(category, 0))
            if need <= 0 or len(free) > need:
                continue

            keys: Dict[Any, List] = {}
            for index, row in free.iterrows():
                people = row[Col.PERSONS_IDS]
                key = (tuple(people) if isinstance(people, (list, tuple)) else (),
                       row[Col.IMAGE_SUBQUERY_CONTENT])
                keys.setdefault(key, []).append(index)

            for members in keys.values():
                if len(members) > 1:
                    model.Add(sum(x[index] for index in members) <= 1)

    def _add_repeat_penalty(self, model, frame, x, penalties) -> None:
        """Charge for each extra photo of the same person within a class.

        The counterpart to the `people` coverage dimension, and not the same
        shape. Coverage caps its reward at one per bucket, so once a person is
        in the album a second photo of them is merely worth nothing; a penalty
        keeps charging, so it actively pushes a class to spread across people.
        That is what `person_max_union_selection` does for the group classes,
        and coverage could not reproduce it.

        **Per class, and per class only.** A global version is meaningless: the
        bride is in most of the gallery, and charging for that would price the
        album's subject out of her own album. Even per class it is wrong for
        most classes -- every photo in `bride` contains the bride, and every
        photo in `bride and groom` contains both, by construction -- which is
        why the weight is a per-class table defaulting to zero rather than one
        number. It is worth something only where a repeated face means a
        wasted slot: the group and crowd classes.
        """
        config = self.cfg.get('people_repeat', {})
        if not config.get('enabled', False):
            return

        allowed = int(config.get('free_repeats', 1))
        cap = int(config.get('max_people', 0))

        for category, group in frame.groupby(Col.CLUSTER_CONTEXT):
            weight = int((config.get('per_class') or {}).get(category,
                                                             config.get('weight', 0)))
            if weight <= 0:
                continue
            for person, members in _largest(_by_identity(group), cap):
                if len(members) <= allowed:
                    continue
                repeats = model.NewIntVar(0, len(members),
                                          f"rep_{_slug(category)}_{_slug(person)}")
                # Penalised, so the solver drives it to max(0, picked - allowed).
                model.Add(repeats >= sum(x[index] for index in members) - allowed)
                penalties.append(repeats * weight)

    def _add_similarity_penalty(self, model, frame, x, penalties) -> None:
        """Charge for picking two photos that are nearly the same shot.

        `_add_exclusions` is a wall at 0.97 and nothing stands below it, so the
        picker sits just underneath: on the two galleries measured the closest
        selected pairs land at 0.972 and 0.964, and the *median* photo in the
        body has a neighbour at 0.902 and 0.841. The rule almost never binds,
        and everything short of identical is free.

        Worse, it is free while `_add_cohesion` is actively *paying* for
        neighbours -- and the second frame of a burst is the nearest neighbour
        there is. So the album fills with the same shot twice.

        This is the ramp into that wall. The charge rises from nothing at
        `similar_soft_threshold` to the full weight at `duplicate_similarity`,
        where the hard rule takes over, so the closer two frames are the more a
        pair costs. That is what makes it prefer the second-closest frame over
        the closest without ever forbidding either: a pair worth having -- both
        well ranked, or needed to fill a quota -- can still be bought.

        Same class and inside the cohesion window, like the two terms it sits
        between, which keeps it O(n) rather than O(n^2) over the gallery.
        """
        weight = int(self.cfg.get('similar_penalty_weight', 0))
        soft = float(self.cfg.get('similar_soft_threshold', 1.0))
        hard = float(self.cfg.get('duplicate_similarity', 0.97))
        if weight <= 0 or soft >= 1.0 or soft >= hard:
            return
        if Col.EMBEDDING not in frame.columns:
            return

        reach = int(self.cfg.get('cohesion_max_gap', 10))
        span = hard - soft
        charged = 0

        for _category, group in frame.groupby(Col.CLUSTER_CONTEXT):
            index = list(group.index)
            vectors = {i: _unit(group.at[i, Col.EMBEDDING]) for i in index}
            positions = group[tl.POSITION].values

            for a in range(len(index)):
                first = index[a]
                if vectors[first] is None:
                    continue
                for b in range(a + 1, len(index)):
                    if int(positions[b] - positions[a]) > reach:
                        break
                    second = index[b]
                    if vectors[second] is None:
                        continue
                    similarity = float(vectors[first] @ vectors[second])
                    # At or above `hard` the pair is already forbidden outright,
                    # so charging it as well would be double counting.
                    if similarity < soft or similarity >= hard:
                        continue
                    if frame.at[first, '_committed'] and frame.at[second, '_committed']:
                        continue
                    charge = int(round(weight * (similarity - soft) / span))
                    if charge <= 0:
                        continue
                    # Only the lower bound is needed: this is a cost, so the
                    # solver drives the pair variable to 0 wherever it may.
                    pair = model.NewBoolVar(f"sim_{first}_{second}")
                    model.Add(pair >= x[first] + x[second] - 1)
                    penalties.append(pair * charge)
                    charged += 1

        if charged and self.logger:
            self.logger.info(
                f"cp-sat: {charged} near-duplicate pairs charged between "
                f"{soft} and {hard} (max {weight})")

    def _add_exclusions(self, model, frame, x) -> None:
        """Near-identical frames of the same class cannot both be picked.

        Cohesion rewards neighbours, and neighbours are exactly where the second
        copy of a shot lives, so without this the reward is collected by
        duplicates. Restricted to same-class pairs inside the cohesion window,
        which keeps it O(n) in practice rather than O(n^2) over the gallery.
        """
        threshold = float(self.cfg.get('duplicate_similarity', 0.97))
        treatment = float(self.cfg.get('treatment_duplicate_similarity', 1.0))
        if Col.EMBEDDING not in frame.columns:
            return
        if threshold >= 1.0 and treatment >= 1.0:
            return

        reach = int(self.cfg.get('cohesion_max_gap', 10))
        has_colour = Col.IMAGE_COLOR in frame.columns
        excluded = 0

        for _category, group in frame.groupby(Col.CLUSTER_CONTEXT):
            index = list(group.index)
            vectors = {i: _unit(group.at[i, Col.EMBEDDING]) for i in index}
            positions = group[tl.POSITION].values

            for a in range(len(index)):
                first = index[a]
                if vectors[first] is None:
                    continue
                for b in range(a + 1, len(index)):
                    if int(positions[b] - positions[a]) > reach:
                        break
                    second = index[b]
                    if vectors[second] is None:
                        continue
                    similarity = float(vectors[first] @ vectors[second])

                    # The same frame in two treatments. `enrich.duplicate_shots`
                    # is supposed to remove these, and cannot when the capture
                    # times are unusable: 53227528 carries **3 distinct
                    # `image_time` values across 636 photos**, so every
                    # same-second group is far too big to be a re-upload set and
                    # the stage does nothing at all. The album then closed with
                    # the groom's balcony portrait in colour *and* in black and
                    # white.
                    #
                    # Nothing here reads a clock, which is the point -- it is
                    # the time key that artificial time breaks. A colour and a
                    # greyscale frame of one class, this alike, are redundant
                    # whether they are one file exported twice or two frames a
                    # moment apart, and only one belongs in the album.
                    #
                    # A hard exclusion rather than a charge because the soft
                    # penalty cannot win here: `groom` needed eight photos and
                    # the run alternates treatments, so the solver rightly paid
                    # 600 rather than leave a quota slot empty against a
                    # shortage of 4000.
                    treatment_pair = (
                        has_colour
                        and similarity >= treatment
                        and group.at[first, Col.IMAGE_COLOR]
                        != group.at[second, Col.IMAGE_COLOR]
                    )
                    if similarity < threshold and not treatment_pair:
                        continue
                    if frame.at[first, '_committed'] and frame.at[second, '_committed']:
                        continue
                    model.Add(x[first] + x[second] <= 1)
                    excluded += 1

        if excluded and self.logger:
            self.logger.info(f"cp-sat: {excluded} pairs excluded as the same shot "
                             f"(identical above {threshold}, or one shot in two "
                             f"treatments above {treatment})")

    def _add_cohesion(self, model, frame, x, rewards) -> None:
        """Reward picking consecutive photos of the same class.

        This is the objective that competes with coverage: a run of neighbours
        reads as one moment, which is what a spread wants, while coverage wants
        picks spread over the day. The weight is where that argument is settled.
        """
        weight = int(self.cfg.get('cohesion_weight', 60))
        if weight <= 0:
            return

        reach = int(self.cfg.get('cohesion_max_gap', 10))

        for _category, group in frame.groupby(Col.CLUSTER_CONTEXT):
            index = list(group.index)
            positions = group[tl.POSITION].values
            for a in range(len(index) - 1):
                first, second = index[a], index[a + 1]
                if int(positions[a + 1] - positions[a]) > reach:
                    continue
                # A pair of already-committed photos is a constant in the
                # objective; leaving it out keeps the reported score dynamic.
                if frame.at[first, '_committed'] and frame.at[second, '_committed']:
                    continue
                pair = model.NewBoolVar(f"pair_{first}_{second}")
                model.Add(pair <= x[first])
                model.Add(pair <= x[second])
                rewards.append(pair * weight)

    # -- results ------------------------------------------------------------

    def _collect(self, frame, x, solver) -> List:
        """Chosen ids: committed first, then per category, as the loop emits."""
        picked = frame.index[[bool(solver.Value(x[i])) for i in frame.index]]
        chosen: List = list(self.committed)

        for category, group in frame.loc[picked].groupby(Col.CLUSTER_CONTEXT):
            group = group[~group['_committed']]
            ordered = group.sort_values('_score', ascending=False)
            ids = ordered[Col.IMAGE_ID].tolist()
            entry = self.per_category.setdefault(category, {})
            entry['selected'] = entry.get('selected', 0) + len(ids)
            chosen.extend(ids)

        # Committed photos count towards their own category's tally.
        settled = frame[frame['_committed']]
        for category, group in settled.groupby(Col.CLUSTER_CONTEXT):
            entry = self.per_category.setdefault(category, {})
            entry['selected'] = entry.get('selected', 0) + len(group)

        return chosen

    def _report(self, frame, x, solver, status) -> None:
        """Log what the solve decided, per class and per window."""
        from ortools.sat.python import cp_model

        picked = frame.index[[bool(solver.Value(x[i])) for i in frame.index]]
        chosen = frame.loc[picked]

        lines = [
            f"cp-sat: {solver.StatusName(status)} in {solver.WallTime():.2f}s, "
            f"objective {solver.ObjectiveValue():.0f}, "
            f"{len(chosen)} of {len(frame)} photos "
            f"({len(self.committed)} committed), "
            f"model {self.model_size[0]} vars / {self.model_size[1]} constraints",
            f"  {'class':<28} {'need':>4} {'got':>4} {'pool':>5}",
        ]
        for category, group in frame.groupby(Col.CLUSTER_CONTEXT):
            got = int(chosen[Col.CLUSTER_CONTEXT].eq(category).sum())
            need = int(self.plan.images.get(category, 0)) + int(group['_committed'].sum())
            lines.append(f"  {str(category)[:28]:<28} {need:>4} {got:>4} {len(group):>5}")

        if '_window' in frame.columns:
            spread = chosen['_window'].value_counts().sort_index().to_dict()
            lines.append(f"  per window: {spread}")

        self.logger.info("\n".join(lines))
        if status == cp_model.FEASIBLE:
            self.logger.info("cp-sat: time limit hit before proving optimality")


def _slug(value: Any) -> str:
    return str(value).replace(' ', '_').replace('|', '_')


def _unit(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    vector = np.asarray(value, dtype=float).ravel()
    norm = float(np.linalg.norm(vector))
    if not vector.size or not norm:
        return None
    return vector / norm


# -- bucketing ---------------------------------------------------------------
#
# `{bucket: [index, ...]}` for a set of rows. One per coverage dimension, and
# the only thing that differs between them.


def _by_window(rows: pd.DataFrame) -> Dict[Any, List]:
    """Parts of the day, from the column `_prepare_windows` stamped."""
    return {window: list(members.index)
            for window, members in rows.groupby('_window')}


def _by_identity(rows: pd.DataFrame) -> Dict[Any, List]:
    """People. Multi-valued -- a photo covers everyone in it, which is why the
    bucketing is a mapping rather than a column."""
    if Col.PERSONS_IDS not in rows.columns:
        return {}
    buckets: Dict[Any, List] = {}
    for index, people in rows[Col.PERSONS_IDS].items():
        for person in (people or []):
            buckets.setdefault(person, []).append(index)
    return buckets


def _by_column(column: str):
    """Kinds of shot, by a categorical column.

    `image_subquery_content` by default rather than `cluster_label`, which is
    what `ContentClusterStrategy` round-robins over: on 53459898 the labels run
    to 447 distinct values over 1069 photos, about two photos each, so covering
    them is barely different from rewarding every photo. The subqueries are 116,
    around nine photos each, which is the granularity "do not take two of the
    same kind of shot" actually needs.
    """
    def buckets_of(rows: pd.DataFrame) -> Dict[Any, List]:
        if column not in rows.columns:
            return {}
        values = rows[column]
        return {value: list(members.index)
                for value, members in rows.groupby(values.fillna('__none__'))}
    return buckets_of


def _largest(buckets: Dict[Any, List], cap: int):
    """The `cap` most populated buckets, or all of them when `cap` is 0.

    Bounds the model: people alone is 89 identities across 27 classes on
    53459898, and a bucket holding one photo rewards what the rank term already
    says.
    """
    items = [(bucket, members) for bucket, members in buckets.items() if members]
    if cap and len(items) > cap:
        items.sort(key=lambda pair: -len(pair[1]))
        items = items[:cap]
    return items


def _by_appearance(threshold: float):
    """Photos that look like each other, greedily grouped by cosine.

    The visual dimension, and the one place where the bucketing is not a
    groupby -- there is no column that says "these two frames are the same
    shot". `cluster_label` is the nearest thing and it is far too fine: 447
    values over 1069 photos, about two each, so covering it would be barely
    different from rewarding every photo.

    Single-pass and greedy against the first centroid a photo matches, which
    makes it deterministic given the frame's order (position, from
    `tl.ordered`) and O(n x buckets) rather than the O(n^2) of the pairwise
    exclusions it is meant to generalise. A photo with no usable embedding
    becomes its own bucket: nothing can be said about what it resembles, and
    silently pooling those together would make them exclude each other.
    """
    def buckets_of(rows: pd.DataFrame) -> Dict[Any, List]:
        if rows.empty or Col.EMBEDDING not in rows.columns:
            return {}
        buckets: Dict[Any, List] = {}
        centroids: List = []
        for index, value in rows[Col.EMBEDDING].items():
            vector = _unit(value)
            if vector is None:
                buckets[f"solo_{index}"] = [index]
                continue
            for bucket, centroid in centroids:
                if float(vector @ centroid) >= threshold:
                    buckets[bucket].append(index)
                    break
            else:
                bucket = len(centroids)
                centroids.append((bucket, vector))
                buckets[bucket] = [index]
        return buckets
    return buckets_of


def _identity_from(frame: pd.DataFrame, column: str, fallback):
    """The couple id stamped on the photo table, or whatever the caller knew."""
    if column in frame.columns and len(frame):
        value = frame[column].iloc[0]
        if value is not None and not pd.isna(value):
            return int(value)
    return fallback
