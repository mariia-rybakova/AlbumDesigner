"""Getting-ready categories: keep the bride, and keep the same bride."""

from __future__ import annotations

from collections import Counter

from src.pipeline.select.contracts import CategoryPicks, CategoryRequest
from src.pipeline.select.strategies.base import CategoryStrategy
from utils.selection.refactoring import select_remove_similar


class BridePrepStrategy(CategoryStrategy):
    """Narrow to bride-tagged frames, then to the single most-photographed
    person in them, so the run does not drift between subjects.

    Falls back to the whole colour pool ordered by ``image_order`` when nothing
    is tagged as bride.
    """

    handles = ("bride getting dressed", "getting hair-makeup")

    def pick(self, request: CategoryRequest) -> CategoryPicks:
        need = request.need
        color = request.color

        filtered = color[
            color["image_subquery_content"].str.contains("bride", case=False, na=False)
        ]

        if len(filtered) == 0:
            preferred = (
                color.sort_values(by='image_order', ascending=True)['image_id'].values.tolist()[:need]
            )
            return CategoryPicks(preferred=preferred)

        all_ids = [pid for sublist in filtered["persons_ids"] for pid in sublist]
        if len(all_ids) == 0:
            return CategoryPicks(preferred=list(filtered["image_id"].values[:need]))

        most_common_id = Counter(all_ids).most_common(1)[0][0]
        subject = filtered[filtered["persons_ids"].apply(lambda ids: most_common_id in ids)]

        if len(subject) <= need or len(subject) - need <= 1:
            request.logger.info(
                f"this cluster {request.category} has no enough images related to bride "
                f"less than needed we select them all no filtering"
            )
            preferred = request.ordered(subject)['image_id'].values.tolist()[:need]
            return CategoryPicks(preferred=preferred)

        preferred = select_remove_similar(
            request.is_artificial_time,
            need=need,
            df=subject.reset_index(),
            cluster_name=request.category,
            logger=request.logger,
            target_group_size=10,
            already_selected=request.covered_embeddings,
        )
        return CategoryPicks(preferred=preferred)
