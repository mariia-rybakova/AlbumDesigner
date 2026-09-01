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

``bride walking the aisle`` / ``groom walking the aisle``
    The anchor as an **upper bound** — the processional happens before the
    ceremony. Identity is mandatory here and does the discriminating; the
    subquery and concept signals only rank. That split is forced by the data:
    the ``walking the aisle`` label covers just 1, 9, 5 and 24 photos on the
    validation galleries, and the query bank has no phrase for the groom
    walking in at all.

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
BRIDE_AISLE = "bride walking the aisle"
GROOM_AISLE = "groom walking the aisle"

#: Subqueries that identify a kiss frame.
KISS_QUERIES = (
    "wedding kiss at ceremony",
    "bride and groom kissing romantically",
)

#: Subqueries that hint at a processional. Only a ranking bonus — the groom has
#: none of his own, and the bride's do not cover every gallery.
AISLE_QUERIES = {
    "bride": ("bride walking down aisle with father", "bride walking aisle with parents"),
    "groom": ("groom waiting for bride at the aisle",),
}


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
    provides = frozenset({photo(Col.SEND_OFF_SCORE), photo(Col.AISLE_SCORE)})

    def applies_to(self, context: AlbumContext) -> bool:
        return bool(context.facts.is_wedding)

    def execute(self, context: AlbumContext) -> AlbumContext:
        photos = context.photos
        logger = context.logger

        # Scored for every photo, tagged for none of them yet: the column is
        # part of the contract and downstream reads it whether or not a
        # send-off was found.
        photos[Col.SEND_OFF_SCORE] = tl.concept_scores(photos, CONFIGS['send_off_concept'])
        aisle = CONFIGS['aisle_concepts']
        bride_score = tl.concept_scores(photos, aisle['bride'])
        groom_score = tl.concept_scores(photos, aisle['groom'])
        photos[Col.AISLE_SCORE] = np.maximum(bride_score, groom_score)
        self._aisle_scores = {'bride': bride_score, 'groom': groom_score}
        context.photos = photos

        self._model_version = tl.model_version_of(photos)
        if self._model_version != 2 and logger:
            logger.warning(
                f"Image model version {self._model_version}: concept-gated detection is not "
                f"calibrated for this embedding space, so the send-off and processional will "
                f"not fire. See CONFIGS['send_off_photo_floor'].")
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

        claimed = list(self._tag_kiss(context, ceremony))
        claimed += self._tag_aisle(context, ceremony, exclude=claimed)
        self._tag_send_off(context, ceremony, exclude=claimed)
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

    # -- the processional: anchor as upper bound -----------------------------

    def _tag_aisle(self, context: AlbumContext, ceremony: tl.CeremonyTimeline,
                   exclude: List) -> List:
        """Tag the bride's and the groom's walk in, separately."""
        start = max(0, ceremony.core_start - CONFIGS['aisle_lead_in'])
        end = ceremony.core_start + CONFIGS['aisle_upper_overlap']
        window = tl.eligible(ceremony.frame, CONFIGS['aisle_eligible_labels'], start, end)
        window = window.drop(index=[i for i in exclude if i in window.index])
        if window.empty:
            return []

        bride_id = context.facts.bride_id
        groom_id = context.facts.groom_id
        if bride_id is None and not window.empty:
            bride_id = _first(window, Col.BRIDE_ID)
            groom_id = _first(window, Col.GROOM_ID)

        claimed: List = []
        for who, identity, other, tag in (("bride", bride_id, groom_id, BRIDE_AISLE),
                                          ("groom", groom_id, bride_id, GROOM_AISLE)):
            picked = self._tag_one_walk(context, ceremony, window, who, identity, other, tag,
                                        exclude=exclude + claimed)
            claimed += picked
        return claimed

    def _tag_one_walk(self, context, ceremony, window, who, identity, other, tag, exclude):
        logger = context.logger
        if identity is None or (isinstance(identity, float) and np.isnan(identity)):
            return []

        # Identity is the mandatory signal. Solo: the couple walking in together
        # is not either of them walking in.
        solo = window[window[Col.PERSONS_IDS].apply(
            lambda ids: identity in ids and other not in ids)]
        solo = solo.drop(index=[i for i in exclude if i in solo.index])
        if solo.empty:
            if logger:
                logger.info(f"No {who} processional: no solo-{who} frame before the ceremony")
            return []

        runs = [r for r in tl.group_adjacent(solo, CONFIGS['aisle_max_gap'])
                if len(r) >= CONFIGS['aisle_min_photos']]
        if not runs:
            if logger:
                logger.info(f"No {who} processional: no run of at least "
                            f"{CONFIGS['aisle_min_photos']} solo-{who} frames")
            return []

        concept = pd.Series(self._aisle_scores[who], index=ceremony.frame.index)
        best = max(runs, key=lambda r: self._rank_run(ceremony, solo, concept, who, r))
        best = self._extend_with_singletons(ceremony, solo, best)

        # Cap to the frames nearest the ceremony start, so the tag stays one
        # coherent moment rather than the highest-scoring scatter.
        positions = ceremony.frame.loc[best, tl.POSITION]
        best = list(positions.sub(ceremony.core_start).abs()
                    .sort_values().index[:CONFIGS['aisle_max_photos']])

        context.photos.loc[best, Col.CLUSTER_CONTEXT] = tag
        if logger:
            hits = int(solo.reindex(best)[Col.IMAGE_SUBQUERY_CONTENT]
                       .isin(AISLE_QUERIES[who]).sum())
            span = sorted(int(p) for p in ceremony.frame.loc[best, tl.POSITION])
            logger.info(f"{who.capitalize()} processional: {len(best)} photos at "
                        f"{span[0]}-{span[-1]}, concept={concept.loc[best].mean():.3f}, "
                        f"{hits} with a matching subquery")
        return list(best)

    @staticmethod
    def _rank_run(ceremony, solo, concept, who, run) -> float:
        """Rank a candidate run. See CONFIGS['aisle_rank_weights'].

        The concept term is the raw mean, not normalised across runs: raw, it
        contributes in proportion to how much the embedding space actually
        separates the concept, which keeps this working in both CLIP spaces.
        """
        weights = CONFIGS['aisle_rank_weights']
        distance = min(abs(int(p) - ceremony.core_start)
                       for p in ceremony.frame.loc[run, tl.POSITION])
        subquery_rate = solo.loc[run, Col.IMAGE_SUBQUERY_CONTENT].isin(AISLE_QUERIES[who]).mean()
        return (weights['subquery'] * subquery_rate
                + weights['proximity'] / (1 + distance / CONFIGS['aisle_proximity_half'])
                + weights['concept'] * concept.loc[run].mean())

    @staticmethod
    def _extend_with_singletons(ceremony, solo, run) -> List:
        """Pull in solo frames just off the ends of the winning run."""
        positions = ceremony.frame.loc[run, tl.POSITION]
        low, high = int(positions.min()), int(positions.max())
        reach = CONFIGS['aisle_extend_gap']
        nearby = solo[ceremony.frame.loc[solo.index, tl.POSITION]
                      .between(low - reach, high + reach)]
        return list(dict.fromkeys(list(run) + list(nearby.index)))

    # -- the send-off: anchor as lower bound ---------------------------------

    def _tag_send_off(self, context: AlbumContext, ceremony: tl.CeremonyTimeline,
                      exclude: List) -> None:
        logger = context.logger
        start = max(0, ceremony.anchor - CONFIGS['send_off_back_slack'])
        end = ceremony.core_end + CONFIGS['send_off_horizon']

        candidates = tl.eligible(ceremony.frame, CONFIGS['send_off_eligible_labels'], start, end)
        candidates = candidates.drop(index=[i for i in exclude if i in candidates.index])
        photo_floor = tl.floor_for(CONFIGS['send_off_photo_floor'], self._model_version)
        candidates = candidates[
            ceremony.frame.loc[candidates.index, Col.SEND_OFF_SCORE] >= photo_floor]

        if candidates.empty:
            if logger:
                logger.info(f"No send-off: nothing over {photo_floor:.3f} "
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
        if mean < tl.floor_for(CONFIGS['send_off_burst_floor'], self._model_version):
            if logger:
                logger.info(f"No send-off: best burst scores {mean:.3f}, "
                            f"under {tl.floor_for(CONFIGS['send_off_burst_floor'], self._model_version)}")
            return

        context.photos.loc[best, Col.CLUSTER_CONTEXT] = SEND_OFF
        if logger:
            scores = ceremony.frame.loc[best, Col.SEND_OFF_SCORE]
            logger.info(f"Send-off detected: {len(best)} photos, "
                        f"score mean={scores.mean():.3f} max={scores.max():.3f}")


def _first(frame: pd.DataFrame, column: str):
    """First non-null value of a gallery-constant column, or None."""
    if column not in frame.columns:
        return None
    values = frame[column].dropna()
    return values.iloc[0] if len(values) else None
