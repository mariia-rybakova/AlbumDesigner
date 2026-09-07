"""Getting-ready categories: keep the bride, and keep the same bride."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from src.pipeline.select.contracts import CategoryPicks, CategoryRequest
from src.pipeline.select.strategies.base import CategoryStrategy
from utils.configs import CONFIGS
from utils.selection.refactoring import select_remove_similar


class BridePrepStrategy(CategoryStrategy):
    """Keep the bride, and keep the same bride.

    The subject is ``bride_id`` -- the identity `enrich.identities` already
    resolved. That was not always so: this narrowed by a **substring match** for
    `"bride"` in `image_subquery_content` and then took the most-photographed
    person among the frames that matched, which is a proxy for the bride rather
    than the bride. It failed in two ways at once on gallery 53459898, where
    seven of ten `getting hair-makeup` frames carry
    `unknown_getting_hair_makeup` and so match no text at all: the narrowing had
    three frames to work with, and the photo that reached the album contained
    `persons=[6]` -- not the bride. The substring also admits `bridesmaid`,
    `bridal suite` and `bride's mother`.

    The text filter is kept as a **fallback**, for a gallery where the identity
    model missed her in her own prep shots. Order matters: identity first, text
    second, and the whole colour pool only if neither says anything.
    """

    handles = ("bride getting dressed", "getting hair-makeup")

    def pick(self, request: CategoryRequest) -> CategoryPicks:
        need = request.need
        color = request.color

        subject, how = self._subject(request, color)

        if subject is None or subject.empty:
            request.logger.info(
                f"{request.category}: neither the bride's identity nor a bride "
                f"subquery matched anything; falling back to the whole pool")
            preferred = (
                color.sort_values(by='image_order', ascending=True)['image_id']
                .values.tolist()[:need]
            )
            return CategoryPicks(preferred=preferred)

        request.logger.info(
            f"{request.category}: {len(subject)} of {len(color)} frames kept, by {how}")

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

    # -- who the run is about ------------------------------------------------

    @staticmethod
    def _subject(request: CategoryRequest, color):
        """``(frames, how)`` -- the bride's frames, and which rule found them."""
        bride_id = None
        if CONFIGS.get('bride_prep_by_identity', True)                 and 'bride_id' in color.columns and not color.empty:
            candidate = color['bride_id'].iloc[0]
            if pd.notna(candidate):
                bride_id = candidate

        if bride_id is not None:
            by_identity = color[color['persons_ids'].apply(
                lambda ids: isinstance(ids, (list, tuple, set)) and bride_id in ids)]
            if not by_identity.empty:
                return by_identity, f"identity {bride_id}"

        # Fallback: the bride is not attached to any frame here, so trust the
        # classifier's words instead, and pin the run to one person so it does
        # not drift between subjects.
        by_text = color[
            color["image_subquery_content"].str.contains("bride", case=False, na=False)
        ]
        if by_text.empty:
            return None, "nothing"

        people = [pid for ids in by_text["persons_ids"]
                  if isinstance(ids, (list, tuple, set)) for pid in ids]
        if not people:
            return by_text, "bride subquery, no identities to pin to"

        most_common = Counter(people).most_common(1)[0][0]
        pinned = by_text[by_text["persons_ids"].apply(
            lambda ids: isinstance(ids, (list, tuple, set)) and most_common in ids)]
        return pinned, f"bride subquery, pinned to identity {most_common}"
