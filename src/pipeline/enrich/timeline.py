"""The gallery timeline, and the ceremony's place in it.

Shared by every detector that reasons about *when* something happened rather
than what it looks like. Pure functions over the photo table — no substage
state, no context — so they can be composed, tested and replaced individually.

Two ideas carry most of the weight:

**Position, not time.** Order by ``general_time`` and work in ranks. It is the
only axis that survives both regimes: real EXIF seconds when the timestamps are
trustworthy, and a synthetic monotonic sequence derived from the gallery's own
arrangement when they are not. Galleries with unusable EXIF are common — one
validation gallery has 2 distinct ``image_time`` values across 528 photos — and
a detector keyed to wall-clock minutes simply does not run on them.

**The median climax, not the last.** The subquery classifier scatters stray
"exchanging vows" labels minutes after the real climax. On the validation
galleries the last climax frame sits 85, 171 and 72 positions after the median.
Anchoring on the last one puts the search window past the event it is meant to
find.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.pipeline.contracts import Col
from src.selection.auto_selection import load_pre_queries_embeddings
from utils.reading_tools import map_cluster_label

#: Working columns added by :func:`ordered`. Underscore-prefixed so they cannot
#: collide with a real photo-table column.
POSITION = "_pos"
LABEL = "_label"

#: Subqueries that mark the ceremony's climax — the vows, the rings, the kiss.
#: These are what the anchor is computed from; everything else is placed
#: relative to it.
CEREMONY_CLIMAX_QUERIES = (
    "bride and groom exchanging vows",
    "ring exchange during ceremony",
    "wedding kiss at ceremony",
    "bride and groom kissing romantically",
)


@dataclass
class CeremonyTimeline:
    """Where the ceremony sits in the gallery's sequence.

    ``frame`` is the photo table ordered by position and carrying the working
    columns; ``core`` is the trimmed extent of the ceremony; ``anchor`` is the
    climax position that detectors measure from.
    """

    frame: pd.DataFrame
    core: Tuple[int, int]
    anchor: int
    #: Positions of the climax frames the anchor was derived from. Empty when
    #: the anchor fell back to the end of the ceremony core.
    climax_positions: List[int]

    @property
    def core_start(self) -> int:
        return self.core[0]

    @property
    def core_end(self) -> int:
        return self.core[1]

    @property
    def has_climax(self) -> bool:
        return bool(self.climax_positions)

    def window(self, before: int = 0, after: int = 0) -> Tuple[int, int]:
        """Positions within ``before``/``after`` of the anchor."""
        return max(0, self.anchor - before), self.anchor + after

    def rows(self, start: int, end: int) -> pd.DataFrame:
        return self.frame[self.frame[POSITION].between(start, end)]


def ordered(photos: pd.DataFrame) -> pd.DataFrame:
    """Photo table ordered by ``general_time``, with position and label added.

    The label is the *per-photo* class, not the cluster's. The cluster label
    collapses to ``"None"`` for a large minority of a gallery (42 of 528 on one
    validation gallery) while the per-photo one never does, and they agree 88%
    of the time — so the per-photo label is both more complete and, for
    "which class did this photo land in", the more honest signal.
    """
    frame = photos.copy()
    frame[LABEL] = frame[Col.IMAGE_CLASS].apply(
        lambda c: map_cluster_label(int(c)) if pd.notna(c) else "None")
    frame = frame.sort_values(Col.GENERAL_TIME)
    frame[POSITION] = range(len(frame))
    return frame


#: Positions this far apart still count as one block of ceremony coverage.
CEREMONY_BLOCK_GAP = 30


def densest_run(positions: Sequence[int], max_gap: int = CEREMONY_BLOCK_GAP):
    """Start and end of the largest contiguous cluster of ``positions``."""
    ordered_positions = sorted(positions)
    runs, current = [], [ordered_positions[0]]
    for position, previous in zip(ordered_positions[1:], ordered_positions[:-1]):
        if position - previous <= max_gap:
            current.append(position)
        else:
            runs.append(current)
            current = [position]          # rebind: clearing would alias the appended run
    runs.append(current)
    longest = max(runs, key=len)
    return longest[0], longest[-1]


def ceremony_timeline(frame: pd.DataFrame, min_ceremony_photos: int = 5) -> Optional[CeremonyTimeline]:
    """Locate the ceremony and its climax. ``None`` when there is no ceremony.

    The core start is the **later** of the 5th percentile and the start of the
    largest contiguous block of ceremony frames. The two filters catch different
    noise: the percentile trims sparse outliers by count, the block start trims
    by contiguity. Either alone is not enough — on one validation gallery three
    stray ceremony labels 80 positions early survived the percentile (7% of the
    class) and dragged the core start back into the couple portrait session, so
    the processional window searched the wrong part of the day entirely.

    The end stays at the 95th percentile: single stray ceremony frames turn up
    hours later (one gallery's ceremony label spans 602 positions while its core
    spans 209), and including them would stretch the window across the whole day.
    """
    ceremony = frame[frame[LABEL] == "ceremony"]
    if len(ceremony) < min_ceremony_photos:
        return None

    positions_of_ceremony = ceremony[POSITION].tolist()
    block_start, _block_end = densest_run(positions_of_ceremony)
    core = (max(int(np.percentile(ceremony[POSITION], 5)), block_start),
            int(np.percentile(ceremony[POSITION], 95)))

    climax = frame[
        frame[POSITION].between(*core)
        & frame[Col.IMAGE_SUBQUERY_CONTENT].isin(CEREMONY_CLIMAX_QUERIES)
    ]
    positions = sorted(int(p) for p in climax[POSITION])
    anchor = int(np.median(positions)) if positions else core[1]

    return CeremonyTimeline(frame=frame, core=core, anchor=anchor, climax_positions=positions)


def model_version_of(photos: pd.DataFrame) -> int:
    return int(photos[Col.MODEL_VERSION].iloc[0])


def floor_for(setting, model_version: int) -> float:
    """Resolve a threshold that may be keyed by image model version.

    The v1 and v2 CLIP spaces put a gallery's cosines on different scales, so a
    single number cannot serve both. Plain floats are still accepted so a
    caller can pin one value deliberately.
    """
    if isinstance(setting, dict):
        return float(setting.get(model_version, setting[max(setting)]))
    return float(setting)


def concept_scores(photos: pd.DataFrame, concept: str) -> np.ndarray:
    """Cosine of each photo against the nearest phrase of a concept.

    Both sides are L2-normalised here, so the result is a true cosine whatever
    state the embeddings arrived in. The concept bin is cached by the loader.
    """
    model_version = int(photos[Col.MODEL_VERSION].iloc[0])
    bank = np.asarray(load_pre_queries_embeddings(concept, model_version), dtype=np.float32)
    bank = bank / np.linalg.norm(bank, axis=1, keepdims=True)

    matrix = np.vstack(photos[Col.EMBEDDING].values).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

    return (matrix @ bank.T).max(axis=1)


def group_adjacent(frame: pd.DataFrame, max_gap: int) -> List[List]:
    """Split rows into runs whose positions are at most ``max_gap`` apart.

    Returns lists of index labels, in position order. An empty frame gives no
    groups.
    """
    if frame.empty:
        return []
    positions = frame[POSITION].tolist()
    groups, current = [], [frame.index[0]]
    for label, position, previous in zip(frame.index[1:], positions[1:], positions[:-1]):
        if position - previous <= max_gap:
            current.append(label)
        else:
            groups.append(current)
            current = [label]
    groups.append(current)
    return groups


def eligible(frame: pd.DataFrame, labels: Sequence[str], start: int, end: int) -> pd.DataFrame:
    """Rows inside a position window whose per-photo label is one of ``labels``."""
    return frame[frame[POSITION].between(start, end) & frame[LABEL].isin(tuple(labels))]
