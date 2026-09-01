"""Parent portraits, balanced so the two families get equal billing."""

from __future__ import annotations

import pandas as pd

from src.pipeline.select.contracts import CategoryPicks, CategoryRequest
from src.pipeline.select.strategies.base import CategoryStrategy

BOTH_PARENTS = "bride and groom with parents"
BRIDE_PARENTS = "bride with her parents"
GROOM_PARENTS = "groom with his parents"

#: How far apart two shots may be and still read as the same posed moment.
PAIR_WINDOW = pd.Timedelta(minutes=10)


class ParentsPortraitStrategy(CategoryStrategy):
    """Take a couple-with-both-families shot, then match the two single-family
    shots into time-adjacent pairs so they can face each other on a spread.
    """

    handles = ("parents portrait",)

    def pick(self, request: CategoryRequest) -> CategoryPicks:
        need = request.need
        color = request.color

        selected_both = self._distinct_parent_sets(color, request)
        remaining_need = max(0, need - len(selected_both))

        # Pairing only makes sense in twos. On an odd remainder the original
        # monolith fell through without assigning anything, so the category
        # reused whatever the previously-processed category had left in the
        # variable. `preferred=None` reproduces that exactly — see
        # `_CARRY_OVER` in ..pick. Returning `selected_both` here instead would
        # be the fix, and is a one-line change once you want it.
        if remaining_need % 2 != 0:
            return CategoryPicks(preferred=None)

        pairs_needed = remaining_need // 2

        bride_side = color[color["parent_category"] == BRIDE_PARENTS].copy()
        groom_side = color[color["parent_category"] == GROOM_PARENTS].copy()

        if bride_side.empty and groom_side.empty:
            return CategoryPicks(preferred=selected_both["image_id"].tolist())

        bride_idx, groom_idx = self._pair_by_time(bride_side, groom_side, pairs_needed)

        final = pd.concat(
            [selected_both, color.loc[bride_idx], color.loc[groom_idx]], axis=0
        )
        return CategoryPicks(preferred=final["image_id"].tolist())

    @staticmethod
    def _distinct_parent_sets(color: pd.DataFrame, request: CategoryRequest) -> pd.DataFrame:
        """Up to two couple-with-both-families shots, preferring different
        parent line-ups over two takes of the same one."""
        both = color[color["parent_category"] == BOTH_PARENTS].copy()

        couple = {request.bride_id, request.groom_id}
        both["parent_ids"] = both["persons_ids"].apply(
            lambda ids: tuple(sorted(i for i in ids if i not in couple))
        )

        wanted = 2 if request.need >= 2 else 1
        chosen, seen = [], set()

        for idx, row in both.iterrows():
            if len(chosen) >= wanted:
                break
            parent_ids = row["parent_ids"]
            if not seen or parent_ids not in seen:
                chosen.append(idx)
                seen.add(parent_ids)

        return both.loc[chosen]

    @staticmethod
    def _pair_by_time(bride_side: pd.DataFrame, groom_side: pd.DataFrame, pairs_needed: int):
        """Greedily match each bride-side shot to the nearest unused groom-side
        shot within :data:`PAIR_WINDOW`."""
        bride_side["image_time"] = pd.to_datetime(bride_side["image_time"])
        groom_side["image_time"] = pd.to_datetime(groom_side["image_time"])

        bride_side = bride_side.sort_values("image_time")
        groom_side = groom_side.sort_values("image_time")

        bride_idx, groom_idx, used = [], [], set()

        for idx, row in bride_side.iterrows():
            if len(bride_idx) >= pairs_needed:
                break

            taken_at = row["image_time"]
            candidates = groom_side[
                (~groom_side.index.isin(used))
                & (groom_side["image_time"].between(taken_at - PAIR_WINDOW, taken_at + PAIR_WINDOW))
            ]
            if candidates.empty:
                continue

            best = (candidates["image_time"] - taken_at).abs().idxmin()
            bride_idx.append(idx)
            groom_idx.append(best)
            used.add(best)

        return bride_idx, groom_idx
