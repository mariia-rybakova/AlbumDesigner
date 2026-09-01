"""The couple-and-timeline categories.

These are the categories where *who is in frame* is the whole point: the solo
portraits, the two party sides, the aisle walk and the reception beats. Each
gets an identity filter, then a shared safety net that puts photos back when
the filter was too aggressive, then the diversity pass.
"""

from __future__ import annotations

import pandas as pd

from src.pipeline.select.contracts import CategoryPicks, CategoryRequest
from src.pipeline.select.strategies.base import CategoryStrategy
from utils.selection.refactoring import select_remove_similar

#: Categories this strategy owns.
ORIENTATION_TIME_CATEGORIES = (
    'bride', 'groom', 'bride and groom', 'bride party', 'groom party',
    'full party', 'walking the aisle',
    'first dance', 'cake cutting', 'ceremony', 'dancing',
)


class CoupleTimelineStrategy(CategoryStrategy):
    """Identity filter -> over-filter recovery -> diversity."""

    handles = ORIENTATION_TIME_CATEGORIES

    def pick(self, request: CategoryRequest) -> CategoryPicks:
        need = request.need
        original_pool = request.color.copy()
        filtered = request.color.copy()

        groom_id = request.groom_id
        bride_id = request.bride_id

        forced = []

        category = request.category

        if category == 'bride':
            filtered = filtered[filtered['persons_ids'].apply(lambda x: x == [bride_id])]

        elif category == 'groom':
            # A thin groom pool is left alone rather than filtered to nothing.
            if len(request.color) >= need * 2:
                filtered = filtered[filtered['persons_ids'].apply(lambda x: x == [groom_id])]

        elif category == 'bride and groom':
            filtered = filtered[
                filtered.apply(
                    lambda row: _has_both(row, bride_id, groom_id)
                    and (row.get('n_faces', 0) == 2 or row.get('number_bodies', 0) == 2),
                    axis=1,
                )
            ]

        elif category == 'bride party':
            if bride_id:
                filtered = filtered[filtered.apply(lambda row: bride_id in row['persons_ids'], axis=1)]

        elif category == 'groom party':
            if groom_id:
                filtered = filtered[filtered.apply(lambda row: groom_id in row['persons_ids'], axis=1)]

        elif category == 'walking the aisle':
            filtered, original_pool, forced, need = self._walk_the_aisle(
                filtered, original_pool, bride_id, groom_id, need
            )

        filtered = self._recover_over_filtering(filtered, original_pool, request)

        indexed = filtered.set_index('image_id')
        preferred = request.ordered(indexed).index.tolist()

        if category == 'dancing' and need < len(preferred):
            preferred, indexed = _prefer_landscape(indexed, preferred, need)

        preferred = select_remove_similar(
            request.is_artificial_time,
            need=need,
            df=indexed.reset_index(),
            cluster_name=category,
            logger=request.logger,
            target_group_size=10,
        )

        if preferred is None:
            return CategoryPicks(forced=forced, remaining_need=need, skip=True)

        return CategoryPicks(preferred=preferred, forced=forced, remaining_need=need)

    # -- category-specific -------------------------------------------------

    @staticmethod
    def _walk_the_aisle(filtered, original_pool, bride_id, groom_id, need):
        """Two scripted beats: the couple together, then the bride arriving.

        Both are committed outright and come out of the budget, so the
        diversity pass fills only what is left.
        """
        forced = []

        bg_mask = filtered.apply(lambda row: _has_both(row, bride_id, groom_id), axis=1)
        bg_ids = filtered.loc[bg_mask, "image_id"].tolist()

        if bg_ids:
            original_pool = original_pool[~original_pool["image_id"].isin(bg_ids)]
            forced.append(bg_ids[0])
            filtered = filtered[~filtered["image_id"].isin(bg_ids)]
            need -= 1

        if bride_id and need > 0:
            bride_only_mask = filtered.apply(
                lambda row: _has_bride_only(row, bride_id, groom_id), axis=1
            )
            bride_only = filtered[bride_only_mask].copy()

            if len(bride_only) > 0:
                # Latest bride-without-groom frame: her arrival at the altar.
                chosen = bride_only.loc[bride_only['image_time_date'].idxmax(), 'image_id']
                forced.append(chosen)
                filtered = filtered[filtered["image_id"] != chosen]
                need -= 1

        return filtered, original_pool, forced, need

    @staticmethod
    def _recover_over_filtering(filtered, original_pool, request):
        """Put photos back when the identity filter cut too deep.

        Without this a gallery whose faces were poorly detected would lose
        whole categories.
        """
        remaining = len(filtered)

        if remaining == 0:
            return original_pool.head(len(original_pool) // 2).copy()

        rejected = original_pool[~original_pool['image_id'].isin(filtered['image_id'])]

        if len(rejected) > 0.8 * remaining:
            n_to_add = remaining // 2
            request.logger.info(
                f"We took out more than 80% from this cluster {request.category} "
                f"so we get {n_to_add} images back from filtering"
            )
            if n_to_add > 0:
                filtered = (
                    pd.concat([filtered, rejected.head(n_to_add)], ignore_index=True)
                    .drop_duplicates(subset='image_id', keep='first')
                )

        return filtered


def _has_both(row, bride_id, groom_id) -> bool:
    ids = {str(v) for v in row.get("persons_ids", [])}
    return {str(bride_id), str(groom_id)}.issubset(ids)


def _has_bride_only(row, bride_id, groom_id) -> bool:
    ids = {str(v) for v in row.get("persons_ids", [])}
    return str(bride_id) in ids and str(groom_id) not in ids


def _prefer_landscape(indexed, preferred, need):
    """Dancing reads better across a spread, so favour landscape frames."""
    reset = indexed.reset_index()
    landscape_ids = reset[
        reset['image_id'].isin(preferred) & (reset['image_orientation'] == 'landscape')
    ]['image_id'].values.tolist()

    if len(landscape_ids) < need:
        non_landscape = indexed[~indexed.index.isin(landscape_ids)].index.tolist()
        preferred = landscape_ids + non_landscape[:need - len(landscape_ids)]
    else:
        preferred = landscape_ids

    return preferred, indexed.loc[preferred]
