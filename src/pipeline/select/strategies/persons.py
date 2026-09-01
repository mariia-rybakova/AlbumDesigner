"""Group categories, chosen for people coverage rather than similarity."""

from __future__ import annotations

import pandas as pd

from src.pipeline.select.contracts import CategoryPicks, CategoryRequest
from src.pipeline.select.strategies.base import CategoryStrategy
from src.selection.person_clustering import person_max_union_selection

PERSONS_CATEGORIES = ('portrait', 'very large group', 'speech')

#: Posed, everyone-facing-camera portraits — the ones worth a formal spread.
FORMAL_QUERIES = [
    "formal studio-style wedding portrait, bride and groom centered, attendants standing still, "
    "bouquets held, symmetrical but there are no people behind them",
    "formal family portrait with bride in white dress and groom in suit AND these people are "
    "standing still AND these people are facing cameraAND these people are arranged in one or two rows",
]

#: Looser group shots, used only to make up the numbers.
INFORMAL_QUERIES = [
    "family with bride and groom group picture at night at the end of the wedding party",
    "a group with people with bride and groom posing AND people behind them in background OR "
    "on the side of picture",
]


class PersonCoverageStrategy(CategoryStrategy):
    """Maximise the number of distinct guests who appear somewhere in the album.

    For ``portrait`` the pool is first narrowed to formal, couple-present
    frames, widening in two steps if that is too strict.
    """

    handles = PERSONS_CATEGORIES

    def pick(self, request: CategoryRequest) -> CategoryPicks:
        pool = request.color

        if request.category == 'portrait':
            pool = self._narrow_to_formal_portraits(request)

        indexed = pool.set_index('image_id')
        ranked = indexed.sort_values('total_score', ascending=False).index.tolist()

        preferred = person_max_union_selection(
            images_for_category=ranked,
            df=indexed.reset_index(),
            needed_count=request.need,
            image_cluster_dict=request.order_index,
            logger=request.logger,
        )
        return CategoryPicks(preferred=preferred)

    @staticmethod
    def _narrow_to_formal_portraits(request: CategoryRequest) -> pd.DataFrame:
        everything = request.color
        bride_id, groom_id = request.bride_id, request.groom_id

        def with_couple(frame: pd.DataFrame) -> pd.Series:
            return frame["persons_ids"].apply(
                lambda x: isinstance(x, list) and bride_id in x and groom_id in x
            )

        formal = everything[
            everything["image_subquery_content"].isin(FORMAL_QUERIES) & with_couple(everything)
        ]

        if len(formal) == 0:
            return everything

        if len(formal) >= request.need:
            return formal

        # Not enough formal shots: widen to the informal group queries, and
        # failing that to any frame with the couple in it.
        extra = everything[
            everything["image_subquery_content"].isin(INFORMAL_QUERIES) & with_couple(everything)
        ]
        if extra.empty:
            extra = everything[with_couple(everything)]

        return pd.concat([formal, extra], ignore_index=True)
