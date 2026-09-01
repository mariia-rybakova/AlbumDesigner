"""Scoring a category's photos and gating them down to candidates.

Two replaceable pieces, deliberately kept apart:

:class:`Scorer`
    Turns a category's photos into a ranked frame. Swap this to change *how*
    relevance is judged.
:class:`CandidateGate`
    Turns that ranking into the shortlist a strategy sees. Swap this to change
    *how many* survive and on what floor.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.selection.ai_wedding_selection import get_scores
from utils.configs import selection_threshold


class Scorer:
    """Blend the relevance signals into ``total_score`` for one category.

    The request-level constants (what the user picked, who they care about,
    which tags, their ratings) are bound once at construction, so scoring a
    category is just ``scorer.score(frame)``.
    """

    def __init__(self, user_selected: pd.DataFrame, person_ids, tags_features, ratings: Dict, logger):
        self.user_selected = user_selected
        self.person_ids = person_ids
        self.tags_features = tags_features
        self.ratings = ratings
        self.logger = logger

    def score(self, frame: pd.DataFrame) -> Tuple[Optional[List], Optional[pd.DataFrame]]:
        """Returns ``(ranked [(image_id, score)], scored frame)``."""
        return get_scores(
            frame, self.user_selected, self.person_ids, self.tags_features, self.logger, self.ratings
        )


class CandidateGate:
    """Cut a scored category down to the photos worth considering."""

    def __init__(self, scorer: Scorer, unscored: bool, allocation: Dict[str, int], logger):
        self.scorer = scorer
        #: True when the request carried no hints at all, so there is nothing to
        #: score against and ``image_order`` decides.
        self.unscored = unscored
        self.allocation = allocation
        self.logger = logger

    def candidates(self, frame: pd.DataFrame, category: str):
        """Returns ``(scored_frame, candidate_ids, used_image_order)``.

        ``used_image_order`` tells the caller which ranking convention applies
        downstream.
        """
        try:
            if self.unscored:
                scored = frame.copy()
                scored["total_score"] = 1.0
                scored = scored.sort_values(by='image_order', ascending=True)
                return scored, scored["image_id"].tolist(), True

            ranked, scored = self.scorer.score(frame)
            if ranked is None or all(score <= 0 for _, score in ranked):
                return None, [], True

            above_floor = [
                (image_id, score) for image_id, score in ranked
                if score > selection_threshold[category]
            ]

            # If anything at all fell below the floor, fall back to a fixed
            # top-N instead. In practice this fires almost every time, which is
            # why the per-category thresholds rarely bite.
            if len(above_floor) < len(scored):
                above_floor = [
                    (row['image_id'], row['total_score'])
                    for _, row in scored.head(self.allocation[category] * 3).iterrows()
                ]

            by_score = sorted(above_floor, key=lambda pair: pair[1], reverse=True)
            return scored, [image_id for image_id, _ in by_score], False

        except Exception as exc:  # noqa: BLE001 - matches the original: skip the category
            self.logger.error(f"Error in get_candidate_images: {exc}")
            return None, None, False


def build_ratings(rating: Optional[List[Dict[str, Any]]]) -> Dict[Any, Any]:
    """``[{photoId, rating}, ...]`` -> ``{photo_id: rating}``."""
    ratings: Dict[Any, Any] = {}
    for item in rating or []:
        photo_id = item.get('photoId')
        if photo_id is not None:
            ratings[photo_id] = item.get('rating', 0)
    return ratings
