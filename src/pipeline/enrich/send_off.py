"""The ceremony send-off: guests showering the couple as they leave.

Confetti, petals, bubbles, rice, sparklers — the substance varies by wedding,
so the tag is the *moment*, not the material. The content classifier has no
class for it (33 classes, none of them a send-off) and the query bank has no
subquery for it, so these photos land in whatever class is nearest — usually
``bride and groom``, ``other`` or a mislabelled ``ceremony`` / ``walking the
aisle``.

Detection is a sequence problem with a visual gate:

1. order the gallery by ``general_time`` — the only axis that works in both
   real-time and artificial-time galleries;
2. find the ceremony block and its climax (vows / ring exchange / kiss);
3. take photos after the climax whose *per-photo* label is one of the classes a
   send-off gets mislabelled into;
4. group them into bursts and keep bursts of at least
   ``send_off_min_photos`` — a send-off is never one or two frames;
5. require visual evidence: the burst must look like a send-off against a
   CLIP concept bank. Sequence alone never tags, because the plain recessional
   is structurally identical.

Both halves are needed. On a validation gallery the single highest-scoring
photo in the whole gallery was a couple portrait session with bubbles two hours
later — visual evidence alone picks the wrong event; sequence alone cannot tell
the send-off from the recessional.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.pipeline.contracts import AlbumContext, Col, photo
from src.pipeline.registry import register
from src.pipeline.substage import SubStage
from src.selection.auto_selection import load_pre_queries_embeddings
from utils.configs import CONFIGS
from utils.reading_tools import map_cluster_label

#: The content class this substage writes. Present in the lookup table, the
#: focus profile and the selection tables, like any other content class.
SEND_OFF = "send off"

#: Subqueries that mark the ceremony's climax — the send-off follows these.
CEREMONY_CLIMAX_QUERIES = (
    "bride and groom exchanging vows",
    "ring exchange during ceremony",
    "wedding kiss at ceremony",
    "bride and groom kissing romantically",
)


@register
class SendOffSubStage(SubStage):
    """Tag the ceremony exit celebration."""

    name = "enrich.send_off"
    requires = frozenset({
        photo(Col.EMBEDDING),
        photo(Col.IMAGE_CLASS),
        photo(Col.MODEL_VERSION),
        photo(Col.GENERAL_TIME),
        photo(Col.CLUSTER_CONTEXT),
    })
    provides = frozenset({photo(Col.SEND_OFF_SCORE)})

    def applies_to(self, context: AlbumContext) -> bool:
        return bool(context.facts.is_wedding)

    def execute(self, context: AlbumContext) -> AlbumContext:
        photos = context.photos
        logger = context.logger

        scores = self._concept_scores(photos)
        photos[Col.SEND_OFF_SCORE] = scores
        context.photos = photos

        burst, reason = self._find_burst(photos, scores)
        if burst is None:
            if logger:
                logger.info(f"No send-off detected: {reason}")
            return context

        photos.loc[burst, Col.CLUSTER_CONTEXT] = SEND_OFF
        context.photos = photos

        if logger:
            picked = photos.loc[burst]
            logger.info(
                f"Send-off detected: {len(burst)} photos, "
                f"score mean={picked[Col.SEND_OFF_SCORE].mean():.3f} "
                f"max={picked[Col.SEND_OFF_SCORE].max():.3f}"
            )
        return context

    # -- visual --------------------------------------------------------------

    @staticmethod
    def _concept_scores(photos: pd.DataFrame) -> np.ndarray:
        """Cosine of each photo against the nearest send-off phrase.

        Both sides are L2-normalised here so the result is a true cosine
        regardless of how the embeddings arrived.
        """
        model_version = int(photos[Col.MODEL_VERSION].iloc[0])
        concept = load_pre_queries_embeddings(
            CONFIGS['send_off_concept'], model_version).astype(np.float32)
        concept /= np.linalg.norm(concept, axis=1, keepdims=True)

        matrix = np.vstack(photos[Col.EMBEDDING].values).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

        return (matrix @ concept.T).max(axis=1)

    # -- sequence ------------------------------------------------------------

    def _find_burst(self, photos: pd.DataFrame, scores: np.ndarray):
        """Returns ``(index labels of the burst, None)`` or ``(None, reason)``."""
        frame = photos.assign(_score=scores)
        frame = frame.sort_values(Col.GENERAL_TIME)
        frame['_pos'] = range(len(frame))
        # Per-photo label, not the cluster's: the cluster label collapses to
        # 'None' for a fifth of a gallery, and the per-photo one never does.
        frame['_label'] = frame[Col.IMAGE_CLASS].apply(
            lambda c: map_cluster_label(int(c)) if pd.notna(c) else 'None')

        window, reason = self._search_window(frame)
        if window is None:
            return None, reason
        start, end = window

        eligible = frame[
            frame['_pos'].between(start, end)
            & frame['_label'].isin(CONFIGS['send_off_eligible_labels'])
            & (frame['_score'] >= CONFIGS['send_off_photo_floor'])
        ]
        if eligible.empty:
            return None, (f"nothing over {CONFIGS['send_off_photo_floor']} in "
                          f"positions {start}-{end}")

        bursts = _group_adjacent(eligible, CONFIGS['send_off_max_gap'])
        bursts = [b for b in bursts if len(b) >= CONFIGS['send_off_min_photos']]
        if not bursts:
            return None, f"no burst of at least {CONFIGS['send_off_min_photos']} photos"

        best = max(bursts, key=lambda b: frame.loc[b, '_score'].mean())
        mean = frame.loc[best, '_score'].mean()
        if mean < CONFIGS['send_off_burst_floor']:
            return None, (f"best burst scores {mean:.3f}, "
                          f"under {CONFIGS['send_off_burst_floor']}")
        return best, None

    @staticmethod
    def _search_window(frame: pd.DataFrame):
        """Positions to search: from the ceremony climax to a horizon past it.

        The climax anchor is the *median* of the climax photos, not the last:
        the subquery classifier scatters stray "exchanging vows" labels minutes
        after the real climax, and anchoring on the last one puts the window
        past the send-off it is meant to find. A backward slack absorbs the
        same noise in the other direction.
        """
        ceremony = frame[frame['_label'] == 'ceremony']
        if len(ceremony) < CONFIGS['send_off_min_photos']:
            return None, "no ceremony block to anchor on"

        core_start = int(np.percentile(ceremony['_pos'], 5))
        core_end = int(np.percentile(ceremony['_pos'], 95))

        climax = frame[
            frame['_pos'].between(core_start, core_end)
            & frame[Col.IMAGE_SUBQUERY_CONTENT].isin(CEREMONY_CLIMAX_QUERIES)
        ]
        anchor = int(np.median(climax['_pos'])) if len(climax) else core_end

        return (max(0, anchor - CONFIGS['send_off_back_slack']),
                core_end + CONFIGS['send_off_horizon']), None


def _group_adjacent(frame: pd.DataFrame, max_gap: int) -> List[List]:
    """Split rows into runs whose positions are at most ``max_gap`` apart."""
    groups, current = [], [frame.index[0]]
    positions = frame['_pos'].tolist()
    for label, position, previous in zip(frame.index[1:], positions[1:], positions[:-1]):
        if position - previous <= max_gap:
            current.append(label)
        else:
            groups.append(current)
            current = [label]
    groups.append(current)
    return groups
