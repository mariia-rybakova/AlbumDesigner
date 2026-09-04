"""One global CP-SAT model instead of a per-category loop.

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

Off by default (`CONFIGS['pick_cpsat']['enabled']`). Failure of any kind --
ortools missing, no solution inside the time limit, a modelling error -- logs
and hands back to `WeddingPicker`, so turning it on cannot cost an album.
`ortools` is imported inside the solve rather than at module scope, so nothing
here is a load-time dependency; it is already installed as a dependency of
`k_means_constrained`, which is why `requirements.txt` needs no change.

**Status: unrefined.** The model solves and the constraints hold, but the
albums it produces are not yet better than the loop's -- the weights are
untuned starting points and coverage is dominated by whatever
`select.preselect` committed.

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
from src.pipeline.select.scoring import Scorer
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
        self._add_windows(model, frame, x, penalties)
        self._add_spacing(model, frame, x)
        self._add_exclusions(model, frame, x)
        self._add_cohesion(model, frame, x, rewards)

        ranks = [
            x[index] * int(round(score * SCORE_SCALE))
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
        frame['_committed'] = frame[Col.IMAGE_ID].isin(self.committed)

        user_selected = photos[photos[Col.IMAGE_ID].isin(self.inputs.user_selected_ids)]
        scorer = Scorer(user_selected, self.inputs.person_ids, self.inputs.tags_features,
                        self.inputs.ratings, self.logger)

        frame['_score'] = 0.0
        for category, group in frame.groupby(Col.CLUSTER_CONTEXT):
            self.per_category.setdefault(category, {})['actual'] = len(group)
            frame.loc[group.index, '_score'] = self._scores_for(scorer, group)

        return frame

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
        """`sum(class) + shortage == Q_c`, shortage penalised.

        Equality, not `>=`: the allowance from `select.budget` is what the
        lookup table sizes spreads against, so overshooting it is not a free
        win. Committed photos are excluded from the sum because
        `select.preselect` already charged them to their class.
        """
        weight = int(self.cfg.get('shortage_weight', 4000))

        for category, group in frame.groupby(Col.CLUSTER_CONTEXT):
            need = int(self.plan.images.get(category, 0))
            free = group.index[~group['_committed']]
            if need <= 0:
                # No allowance left: nothing beyond what is already committed.
                for index in free:
                    model.Add(x[index] == 0)
                continue

            shortage = model.NewIntVar(0, need, f"shortage_{_slug(category)}")
            model.Add(sum(x[index] for index in free) + shortage == need)
            penalties.append(shortage * weight)

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

        edges = np.linspace(0, len(frame), count + 1).astype(int)
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

    def _add_exclusions(self, model, frame, x) -> None:
        """Near-identical frames of the same class cannot both be picked.

        Cohesion rewards neighbours, and neighbours are exactly where the second
        copy of a shot lives, so without this the reward is collected by
        duplicates. Restricted to same-class pairs inside the cohesion window,
        which keeps it O(n) in practice rather than O(n^2) over the gallery.
        """
        threshold = float(self.cfg.get('duplicate_similarity', 0.97))
        if threshold >= 1.0 or Col.EMBEDDING not in frame.columns:
            return

        reach = int(self.cfg.get('cohesion_max_gap', 10))

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
                    if float(vectors[first] @ vectors[second]) < threshold:
                        continue
                    if frame.at[first, '_committed'] and frame.at[second, '_committed']:
                        continue
                    model.Add(x[first] + x[second] <= 1)

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
            f"({len(self.committed)} committed)",
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
