"""Ceremony-anchored moments: the kiss, and the send-off.

Both are the same question asked twice — *what happened around the climax of
the ceremony?* — so they share one anchor rather than deriving it separately.
That matters beyond saving work: two detectors computing the ceremony climax
independently can disagree on the same gallery, one tagging a frame the other's
window excludes.

The anchor is the median of the climax frames (vows, rings, kiss). The two
detectors read outward from it in different directions:

``may kiss bride``
    The anchor as a **centre**. The kiss is itself one of the climax signals,
    so it cannot anchor on itself — it is found by looking for kiss-subquery
    frames on either side of the climax the vows and rings also define.

``send off``
    The anchor as a **lower bound**. Guests shower the couple as they leave, so
    it only ever looks forward, and it needs a burst plus visual confirmation:
    the plain recessional is structurally identical on sequence alone.

The asymmetry in evidence is the interesting part. The kiss has vocabulary —
the query bank carries "wedding kiss at ceremony" and "bride and groom kissing
romantically" — so it is found from labels the pipeline already produced. The
send-off has none: no class in the 33-class taxonomy, no subquery in the 156,
so it needs its own CLIP concept bank.

Supersedes ``enrich.ceremony_kiss``, which anchored on the last "officiant
leading wedding ceremony" frame within a real-timestamp window and tolerated
+/-6 minutes. That combination tagged 1 photo across 3,361 in four validation
galleries: two were rejected outright for having unusable EXIF, and on a third
the kiss frames sat 11 minutes from the officiant anchor.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.pipeline.contracts import AlbumContext, Col, photo
from src.pipeline.enrich import timeline as tl
from src.pipeline.registry import register
from src.pipeline.substage import SubStage
from utils.configs import CONFIGS

#: Content classes this substage writes.
MAY_KISS_BRIDE = "may kiss bride"
SEND_OFF = "send off"

#: Subqueries that identify a kiss frame.
KISS_QUERIES = (
    "wedding kiss at ceremony",
    "bride and groom kissing romantically",
)


@register
class CeremonyAnchorSubStage(SubStage):
    """Find the ceremony climax, then the kiss and the send-off around it."""

    name = "enrich.ceremony_anchor"
    requires = frozenset({
        photo(Col.EMBEDDING),
        photo(Col.IMAGE_CLASS),
        photo(Col.MODEL_VERSION),
        photo(Col.GENERAL_TIME),
        photo(Col.CLUSTER_CONTEXT),
        photo(Col.IMAGE_SUBQUERY_CONTENT),
    })
    provides = frozenset({photo(Col.SEND_OFF_SCORE)})

    def applies_to(self, context: AlbumContext) -> bool:
        return bool(context.facts.is_wedding)

    def execute(self, context: AlbumContext) -> AlbumContext:
        photos = context.photos
        logger = context.logger

        # Scored for every photo, tagged for none of them yet: the column is
        # part of the contract and downstream reads it whether or not a
        # send-off was found.
        photos[Col.SEND_OFF_SCORE] = tl.concept_scores(photos, CONFIGS['send_off_concept'])
        context.photos = photos

        frame = tl.ordered(photos)
        ceremony = tl.ceremony_timeline(frame, CONFIGS['send_off_min_photos'])
        if ceremony is None:
            if logger:
                logger.info("No ceremony block: neither kiss nor send-off can be anchored")
            return context

        if logger:
            logger.info(
                f"Ceremony anchor at position {ceremony.anchor} "
                f"(core {ceremony.core_start}-{ceremony.core_end}, "
                f"{len(ceremony.climax_positions)} climax frames)")

        kissed = self._tag_kiss(context, ceremony)
        self._tag_send_off(context, ceremony, exclude=kissed)
        return context

    # -- the kiss: anchor as centre -----------------------------------------

    def _tag_kiss(self, context: AlbumContext, ceremony: tl.CeremonyTimeline) -> List:
        radius = CONFIGS['kiss_radius']
        start, end = ceremony.window(before=radius, after=radius)

        candidates = tl.eligible(ceremony.frame, CONFIGS['kiss_eligible_labels'], start, end)
        candidates = candidates[candidates[Col.IMAGE_SUBQUERY_CONTENT].isin(KISS_QUERIES)]

        if candidates.empty:
            if context.logger:
                context.logger.info(
                    f"No kiss: no kiss-subquery frame within {radius} positions of the anchor")
            return []

        # A kiss is one moment, not a scatter across the ceremony: keep the run
        # closest to the anchor rather than every kiss-like frame in range.
        runs = tl.group_adjacent(candidates, CONFIGS['kiss_max_gap'])
        best = min(runs, key=lambda r: abs(ceremony.frame.loc[r, tl.POSITION].median() - ceremony.anchor))
        best = best[:CONFIGS['kiss_max_photos']]

        context.photos.loc[best, Col.CLUSTER_CONTEXT] = MAY_KISS_BRIDE
        if context.logger:
            offset = int(ceremony.frame.loc[best, tl.POSITION].median() - ceremony.anchor)
            context.logger.info(
                f"Kiss detected: {len(best)} photos, {offset:+d} positions from the anchor")
        return best

    # -- the send-off: anchor as lower bound ---------------------------------

    def _tag_send_off(self, context: AlbumContext, ceremony: tl.CeremonyTimeline,
                      exclude: List) -> None:
        logger = context.logger
        start = max(0, ceremony.anchor - CONFIGS['send_off_back_slack'])
        end = ceremony.core_end + CONFIGS['send_off_horizon']

        candidates = tl.eligible(ceremony.frame, CONFIGS['send_off_eligible_labels'], start, end)
        candidates = candidates.drop(index=[i for i in exclude if i in candidates.index])
        candidates = candidates[
            ceremony.frame.loc[candidates.index, Col.SEND_OFF_SCORE]
            >= CONFIGS['send_off_photo_floor']]

        if candidates.empty:
            if logger:
                logger.info(f"No send-off: nothing over {CONFIGS['send_off_photo_floor']} "
                            f"in positions {start}-{end}")
            return

        bursts = [b for b in tl.group_adjacent(candidates, CONFIGS['send_off_max_gap'])
                  if len(b) >= CONFIGS['send_off_min_photos']]
        if not bursts:
            if logger:
                logger.info(f"No send-off: no burst of at least "
                            f"{CONFIGS['send_off_min_photos']} photos")
            return

        best = max(bursts, key=lambda b: ceremony.frame.loc[b, Col.SEND_OFF_SCORE].mean())
        mean = ceremony.frame.loc[best, Col.SEND_OFF_SCORE].mean()
        if mean < CONFIGS['send_off_burst_floor']:
            if logger:
                logger.info(f"No send-off: best burst scores {mean:.3f}, "
                            f"under {CONFIGS['send_off_burst_floor']}")
            return

        context.photos.loc[best, Col.CLUSTER_CONTEXT] = SEND_OFF
        if logger:
            scores = ceremony.frame.loc[best, Col.SEND_OFF_SCORE]
            logger.info(f"Send-off detected: {len(best)} photos, "
                        f"score mean={scores.mean():.3f} max={scores.max():.3f}")
