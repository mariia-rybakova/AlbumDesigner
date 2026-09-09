"""Who the parents are -- by identity, or not at all.

The old rule (`identify_parents`, still reachable via
``CONFIGS['parents']['by_identity'] = False``) classified *photos*: a portrait
with 3-4 people including a partner, whose two others were of different
genders, were "in a social circle", and were roughly ``couple_age + 15``.
Every one of those tests turned out to be wrong or vacuous -- the age offset
selects the sibling band and rejects real parents, and the circle test
collapses to "either of them appears in any circle at all". See
`docs/parent_identification.md`.

This resolves **identities** instead, and its output is deliberately
three-valued: the bride's parents, the groom's parents, or *inconclusive*. A
wrong name is worse than no name -- an unmarked parent costs a category that
would have been budgeted anyway, while a marked stranger puts a stranger on
the family spread -- so every gate here fails toward silence.

Once the identities are named the photo labelling is exact and needs no
guessing: a portrait's `parent_category` follows from which named parents are
in it. That also removes the old rule's structural blind spots for free -- one
parent, three parents, and two parents of the same gender all work, because
nothing counts heads any more.

**What separates a parent from a bridesmaid** is the question that matters.
Almost every indicator the two share: both are in the prep scenes, both are at
the ceremony, both are in the posed portraits. Three things do separate them,
measured on the validation galleries:

*The party classes.* On 47981912 the wedding party sat at 32-42 photos in
`bride party` / `groom party` / `full party` while every parent sat at 0-3. On
53459898, 50-60 against 2. This is the strongest single signal and it is a
*negative* one, which is why it is weighted as heavily as any positive.

*Side skew.* Family are photographed with their own child during prep and the
portraits. On 47981912 the six candidates split cleanly: 21/1, 15/0, 10/2
bride-side against 0/13, 3/23, 0/10 groom-side. This is also how the side is
assigned -- not from `main_persons` order, which the old rule used and which
says nothing about who is whose.

*Age, as a rank.* Never as an offset. Face-age estimators regress toward the
mean, so a 60-year-old reads as ~50 and a 30-year-old as ~32: the *gap*
compresses but the *ordering* survives. Same lesson as `enrich.timeline` --
position is trustworthy, measured distance is not.

Two rarer signals are near-conclusive when they fire and absent otherwise, so
they enter as bonuses rather than requirements: a **two-person dance frame**
holding exactly one partner and one candidate (the father-daughter or
mother-son dance -- id 16 on 53459898, the groom's mother, has two), and a
**small social circle** shared with another old candidate, which is how the
two parents of one side corroborate each other. The circles really do carry
this: 53459898's are `[19,21]`, `[8,27]`, `[8,28,39]`, `[17,25,33]` -- pairs
and family triples, exactly the structure the old rule flattened away.

**The queries augment age, they do not replace it.** `parents_of_couple` minus
`wedding_party_member`, scored only on photos where the candidate is one of at
most three identities so the image-level cosine is actually about them. On
47981912 that delta orders parents (+0.04..+0.09) above the party (-0.06)
correctly, but on 53459898 the raw delta is dominated by identities with three
or four attributable photos -- the known bride's mother, with 31, ranked
sixteenth. So it is shrunk toward zero by sample size and capped at a modest
weight. It is a second opinion on age, not a detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.pipeline.contracts import Col
from src.pipeline.enrich.timeline import LABEL, POSITION, concept_scores, ordered
from utils.configs import CONFIGS

# -- the classes each indicator reads --------------------------------------

#: Getting-ready classes, per side. The bride's side is well served by the
#: content model; the groom's prep has no class of its own beyond `suit`, which
#: is why the groom-side prep count is systematically the weaker of the two.
BRIDE_PREP = ("bride getting dressed", "getting hair-makeup")
GROOM_PREP = ("suit",)

AISLE = ("walking the aisle",)
CEREMONY_CLASS = ("ceremony",)
DANCE = ("dancing", "first dance")

#: The confusion class. Membership here is the strongest evidence *against*
#: being a parent.
PARTY = ("bride party", "groom party", "full party")

#: Concepts, all built by `tools/build_concept_bin.py`.
PARENT_CONCEPT = "parents_of_couple"
PARTY_CONCEPT = "wedding_party_member"
DANCE_CONCEPT = "parent_dance"
CELEBRANT_CONCEPT = "celebrant"


def settings() -> dict:
    return CONFIGS["parents"]


# -- evidence ---------------------------------------------------------------


@dataclass(frozen=True)
class Indicators:
    """Everything measured about one candidate identity.

    Kept separate from the scoring so a gallery can be inspected without
    committing to any particular weighting -- `scratchpad/diag_parents.py`
    prints this table directly.
    """

    identity: int
    appearances: int
    age: Optional[float]
    age_rank: float
    gender: Optional[int]

    with_bride: int
    with_groom: int
    prep_bride: int
    prep_groom: int
    aisle_bride: int
    aisle_groom: int
    duo_dance_bride: int
    duo_dance_groom: int
    ceremony: int
    party: int

    attributable: int
    parent_query: float
    dance_query: float
    celebrant_query: float

    #: This candidate's party count as a share of the most party-heavy
    #: candidate in *this* gallery. Absolute counts do not travel: on 53459898
    #: the groom's father sits at 10 `groom party` frames -- in a suit he is
    #: not visually separable from the groomsmen -- while the actual party sits
    #: at 45-53. Ten is disqualifying on an absolute scale and obviously not
    #: on a relative one.
    party_share: float = 0.0
    circle_partners: Tuple[int, ...] = ()
    span: Tuple[float, float] = (0.0, 1.0)

    def side_skew(self) -> Optional[str]:
        """``'bride'``, ``'groom'`` or None when the two sides are too close.

        A candidate with no clear side is not a resolvable parent: even if they
        are one, we cannot say whose, and a parent attached to the wrong side
        is exactly the false mark this module exists to avoid.
        """
        total = self.with_bride + self.with_groom
        if total < settings()["min_side_frames"]:
            return None
        share = self.with_bride / total
        margin = settings()["min_side_share"]
        if share >= margin:
            return "bride"
        if (1.0 - share) >= margin:
            return "groom"
        return None

    def leaning_side(self) -> Optional[str]:
        """The side a candidate tilts to, for when `side_skew` will not commit.

        Same measurement, a lower bar (``min_side_share_lean``). Still requires
        `min_side_frames` of evidence and still returns None at a genuine tie,
        so this widens the gate rather than removing it -- `resolve` pairs it
        with a raised score floor.
        """
        total = self.with_bride + self.with_groom
        if total < settings()["min_side_frames"]:
            return None
        share = self.with_bride / total
        margin = settings()["min_side_share_lean"]
        if share >= margin:
            return "bride"
        if (1.0 - share) >= margin:
            return "groom"
        return None

    def own_side(self, side: str) -> int:
        """Frames alone with that partner -- the couple's other half absent.

        Family are photographed with their own child. A guest is photographed
        with neither, so this is a positive term and not only the gate that
        `side_skew` uses it as. The wedding party shares it, which is what the
        party penalty is there to answer.
        """
        return self.with_bride if side == "bride" else self.with_groom

    def prep(self, side: str) -> int:
        return self.prep_bride if side == "bride" else self.prep_groom

    def aisle(self, side: str) -> int:
        return self.aisle_bride if side == "bride" else self.aisle_groom

    def duo_dance(self, side: str) -> int:
        return self.duo_dance_bride if side == "bride" else self.duo_dance_groom


@dataclass
class Resolution:
    """The three-valued answer, plus why."""

    bride_parents: Tuple[int, ...] = ()
    groom_parents: Tuple[int, ...] = ()
    #: Per-identity explanation, for the log and for `diagnostics`.
    reasons: Dict[int, str] = field(default_factory=dict)
    #: Per-side note when a side could not be resolved.
    inconclusive: Dict[str, str] = field(default_factory=dict)

    def resolved(self) -> bool:
        return bool(self.bride_parents or self.groom_parents)

    def all_parents(self) -> Tuple[int, ...]:
        return tuple(self.bride_parents) + tuple(self.groom_parents)

    def side_of(self, identity: int) -> Optional[str]:
        if identity in self.bride_parents:
            return "bride"
        if identity in self.groom_parents:
            return "groom"
        return None


# -- measurement ------------------------------------------------------------


def _saturate(count: int, full: int) -> float:
    """Diminishing returns: ``full`` occurrences score 1.0, more score 1.0.

    Raw counts would let one heavily-photographed candidate dominate every
    term at once, which is how the most-photographed bridesmaid wins.
    """
    if full <= 0:
        return 0.0
    return min(1.0, count / float(full))


def _shrink(value: float, sample: int, prior: int) -> float:
    """Pull a per-identity mean toward zero when it rests on few photos.

    Without this the query term ranks a candidate with three attributable
    photos above one with thirty; see the module docstring.
    """
    if sample <= 0:
        return 0.0
    return value * (sample / float(sample + prior))


def _small_circles(social_circles: Optional[pd.DataFrame]) -> List[frozenset]:
    """Circles small enough to be a household rather than a guest list."""
    if social_circles is None or social_circles.empty:
        return []
    limit = settings()["max_circle_size"]
    circles = []
    for ids in social_circles.get("identity_ids", []):
        members = frozenset(int(i) for i in (ids or []))
        if 2 <= len(members) <= limit:
            circles.append(members)
    return circles


def measure(
    photos: pd.DataFrame,
    person_details: Optional[pd.DataFrame],
    social_circles: Optional[pd.DataFrame],
    bride_id,
    groom_id,
) -> List[Indicators]:
    """Measure every candidate identity in the gallery.

    Returns an empty list when the gallery cannot support the question at all
    -- no couple resolved, or no per-identity ages to rank.
    """
    if photos is None or photos.empty:
        return []
    if bride_id is None or groom_id is None or pd.isna(bride_id) or pd.isna(groom_id):
        return []
    if person_details is None or person_details.empty:
        return []

    config = settings()
    frame = ordered(photos)
    total = len(frame)

    ages = (
        person_details.dropna(subset=["age"])
        .set_index("identity_id")["age"]
        .astype(float)
        .to_dict()
    )
    genders = person_details.set_index("identity_id")["gender"].to_dict()
    if not ages:
        return []

    # Age as a percentile among the gallery's own identities. Absolute years
    # are the untrustworthy part; the ordering is what survives.
    ranked = sorted(ages.values())
    def age_rank(value: Optional[float]) -> float:
        if value is None:
            return 0.0
        below = sum(1 for other in ranked if other < value)
        return below / float(max(1, len(ranked) - 1))

    scores = {}
    for concept in (PARENT_CONCEPT, PARTY_CONCEPT, DANCE_CONCEPT, CELEBRANT_CONCEPT):
        try:
            scores[concept] = concept_scores(frame, concept)
        except Exception:
            # A missing bin must not take the gallery down; the structural
            # indicators stand on their own and the query term simply
            # contributes nothing.
            scores[concept] = np.zeros(len(frame), dtype=np.float32)

    people = frame[Col.PERSONS_IDS].apply(lambda x: set(x or []))
    crowd = people.apply(len)
    labels = frame[LABEL]
    positions = frame[POSITION]
    attributable_mask = crowd <= config["max_ids_for_attribution"]

    everyone = {p for group in people for p in group}
    out: List[Indicators] = []
    circles = _small_circles(social_circles)

    for identity in sorted(everyone):
        if identity in (bride_id, groom_id):
            continue
        here = people.apply(lambda group: identity in group)
        appearances = int(here.sum())
        if appearances < config["min_appearances"]:
            continue

        with_bride = int((here & people.apply(
            lambda g: bride_id in g and groom_id not in g)).sum())
        with_groom = int((here & people.apply(
            lambda g: groom_id in g and bride_id not in g)).sum())

        def duo(partner) -> int:
            """Dance frames whose people are exactly this candidate and one
            partner -- the parent dance, and nothing else looks like it."""
            pair = {identity, partner}
            return int((here & labels.isin(DANCE) & people.apply(
                lambda g: g == pair)).sum())

        attr = here & attributable_mask
        n_attr = int(attr.sum())
        parent_delta = 0.0
        dance_query = celebrant_query = 0.0
        if n_attr:
            picked = attr.values
            parent_delta = float(
                scores[PARENT_CONCEPT][picked].mean()
                - scores[PARTY_CONCEPT][picked].mean())
            dance_query = float(scores[DANCE_CONCEPT][picked].max())
            celebrant_query = float(scores[CELEBRANT_CONCEPT][picked].mean())

        mine = positions[here]
        out.append(Indicators(
            identity=identity,
            appearances=appearances,
            age=ages.get(identity),
            age_rank=age_rank(ages.get(identity)),
            gender=genders.get(identity),
            with_bride=with_bride,
            with_groom=with_groom,
            prep_bride=int((here & labels.isin(BRIDE_PREP)).sum()),
            prep_groom=int((here & labels.isin(GROOM_PREP)).sum()),
            aisle_bride=int((here & labels.isin(AISLE) & people.apply(
                lambda g: bride_id in g)).sum()),
            aisle_groom=int((here & labels.isin(AISLE) & people.apply(
                lambda g: groom_id in g)).sum()),
            duo_dance_bride=duo(bride_id),
            duo_dance_groom=duo(groom_id),
            ceremony=int((here & labels.isin(CEREMONY_CLASS)).sum()),
            party=int((here & labels.isin(PARTY)).sum()),
            attributable=n_attr,
            parent_query=_shrink(parent_delta, n_attr, config["query_prior"]),
            dance_query=dance_query,
            celebrant_query=celebrant_query,
            circle_partners=(),
            span=((mine.min() / total, mine.max() / total) if total else (0.0, 1.0)),
        ))

    return _attach_circles(_attach_party_share(out), circles)


def _attach_party_share(candidates: List[Indicators]) -> List[Indicators]:
    """Scale each candidate's party count against the gallery's own maximum.

    A gallery with no party coverage at all -- 52894932 has none -- gives every
    candidate a share of zero rather than amplifying two or three frames into a
    full penalty.
    """
    reference = max((c.party for c in candidates), default=0)
    if reference < settings()["min_party_reference"]:
        return candidates
    return [Indicators(**{**c.__dict__, "party_share": c.party / float(reference)})
            for c in candidates]


def _attach_circles(candidates: List[Indicators], circles: Sequence[frozenset]):
    """Record, per candidate, which *other old candidates* share a small circle.

    Two parents of one side corroborate each other: a household circle holding
    two candidates who are both in the older part of the gallery is far better
    evidence than either alone. A circle shared with a young candidate says
    nothing -- that is a bridesmaid and her partner.
    """
    floor = settings()["min_age_rank"]
    old = {c.identity for c in candidates if c.age_rank >= floor}
    by_id = {c.identity: c for c in candidates}
    updated = []
    for candidate in candidates:
        partners = set()
        for circle in circles:
            if candidate.identity in circle:
                partners |= {i for i in circle
                             if i != candidate.identity and i in old and i in by_id}
        updated.append(Indicators(**{**candidate.__dict__,
                                     "circle_partners": tuple(sorted(partners))}))
    return updated


# -- scoring ----------------------------------------------------------------


def score(candidate: Indicators, side: str) -> float:
    """How much this candidate looks like a parent *on this side*.

    Positive terms are capped; the party count is the one term allowed to sink
    a candidate outright, because it is the only indicator that is nearly
    exclusive to the confusion class.
    """
    config = settings()
    weights = config["weights"]

    value = weights["age_rank"] * candidate.age_rank
    value += weights["own_side"] * _saturate(candidate.own_side(side),
                                             config["own_side_full"])
    value += weights["prep"] * _saturate(candidate.prep(side), config["prep_full"])
    value += weights["aisle"] * _saturate(candidate.aisle(side), config["aisle_full"])
    value += weights["duo_dance"] * (1.0 if candidate.duo_dance(side) else 0.0)
    value += weights["circle"] * (1.0 if candidate.circle_partners else 0.0)
    value += weights["query"] * float(
        np.clip(candidate.parent_query / config["query_full"], -1.0, 1.0))

    value -= weights["party"] * candidate.party_share

    # The officiant: old, at the ceremony, on neither side, and present in a
    # narrow band of the day. Ranks well on age alone otherwise -- id 26 on
    # 53459898 is 61 and appears in 28 ceremony frames inside 8% of the day.
    width = candidate.span[1] - candidate.span[0]
    if width <= config["officiant_span"] and candidate.ceremony >= config["officiant_ceremony"]:
        value -= weights["officiant"]

    return value


# -- the decision -----------------------------------------------------------


def resolve(candidates: Sequence[Indicators]) -> Resolution:
    """Name the parents, or decline to.

    Each side is decided on its own, so a gallery that resolves the bride's
    mother and cannot tell the groom's parents apart keeps the half it knows.
    """
    config = settings()
    resolution = Resolution()

    by_side: Dict[str, List[Indicators]] = {"bride": [], "groom": []}
    ambiguous: List[Indicators] = []
    for candidate in candidates:
        side = candidate.side_skew()
        if side is not None:
            by_side[side].append(candidate)
        else:
            ambiguous.append(candidate)

    # A parent whose side is not clean enough to assert is still a parent. The
    # mother of the bride on 49995684 sits at 10 frames alone with her daughter
    # against 7 alone with the groom -- 58.8%, under the 70% the strict gate
    # wants -- so she was dropped from both pools and never ranked, despite
    # being the second-strongest bride-side candidate in the gallery at 0.66.
    #
    # That is not a harmless miss. `label` requires a portrait to hold nobody
    # outside the named family, so an unnamed parent does not merely go
    # uncredited: she invalidates every family portrait she stands in. All seven
    # `[bride, her, her husband]` portraits failed on her alone, and the gallery
    # ended with a resolved father and zero parent portraits.
    #
    # So the lean is allowed to decide when it is still a lean, and the price is
    # a higher score bar than the strict path pays -- being obviously a parent is
    # what earns the weaker side evidence. Candidates with no lean at all remain
    # unplaceable, which is the case the strict gate was really written for.
    for candidate in ambiguous:
        side = candidate.leaning_side()
        if side is None:
            continue
        if score(candidate, side) < config["min_score_ambiguous_side"]:
            continue
        by_side[side].append(candidate)

    for side, pool in by_side.items():
        chosen, note = _resolve_side(side, pool, config)
        if chosen:
            if side == "bride":
                resolution.bride_parents = tuple(c.identity for c in chosen)
            else:
                resolution.groom_parents = tuple(c.identity for c in chosen)
            for candidate in chosen:
                resolution.reasons[candidate.identity] = _explain(candidate, side)
        else:
            resolution.inconclusive[side] = note

    return resolution


def _resolve_side(side: str, pool: List[Indicators], config: dict):
    """Return ``(chosen, note)`` for one side."""
    if not pool:
        return [], "no candidate is clearly on this side"

    ranked = sorted(pool, key=lambda c: -score(c, side))

    # A parent is in the older part of the gallery. This is the one hard
    # requirement -- everything else trades off, but a candidate younger than
    # the couple's own cohort is not their parent whatever else fires.
    eligible = [c for c in ranked
                if c.age_rank >= config["min_age_rank"]
                and score(c, side) >= config["min_score"]]
    if not eligible:
        best = score(ranked[0], side)
        return [], (f"best candidate {ranked[0].identity} scores {best:.2f} "
                    f"(age rank {ranked[0].age_rank:.2f}); "
                    f"floor is {config['min_score']:.2f} at rank "
                    f"{config['min_age_rank']:.2f}")

    chosen = eligible[: config["max_per_side"]]

    # The separation test. If the next candidate is as good as the last one
    # taken, we cannot tell them apart -- so give back the ambiguous tail
    # rather than guess which of them is the parent.
    rest = [c for c in ranked if c not in chosen]
    while chosen and rest:
        gap = score(chosen[-1], side) - score(rest[0], side)
        if gap >= config["min_margin"]:
            break
        chosen = chosen[:-1]
    if not chosen:
        return [], (f"candidates {[c.identity for c in eligible[:3]]} are within "
                    f"{config['min_margin']:.2f} of each other; cannot separate them")

    return chosen, ""


def _explain(candidate: Indicators, side: str) -> str:
    bits = [f"side={side}", f"age={candidate.age:.0f}" if candidate.age else "age=?",
            f"rank={candidate.age_rank:.2f}"]
    if candidate.prep(side):
        bits.append(f"prep={candidate.prep(side)}")
    if candidate.aisle(side):
        bits.append(f"aisle={candidate.aisle(side)}")
    if candidate.duo_dance(side):
        bits.append(f"parent-dance={candidate.duo_dance(side)}")
    if candidate.circle_partners:
        bits.append(f"circle={list(candidate.circle_partners)}")
    if candidate.party:
        bits.append(f"party={candidate.party}")
    bits.append(f"query={candidate.parent_query:+.3f}")
    bits.append(f"score={score(candidate, side):.2f}")
    return " ".join(bits)


# -- labelling --------------------------------------------------------------

BOTH_PARENTS = "bride and groom with parents"
BRIDE_PARENTS = "bride with her parents"
GROOM_PARENTS = "groom with his parents"
PARENTS_PORTRAIT = "parents portrait"


def label(photos: pd.DataFrame, resolution: Resolution, bride_id, groom_id):
    """Re-label the portraits that are *of* the parents and the couple.

    Exact, now that the parents have names: no age rule, no gender rule. A
    portrait qualifies if it holds at least one partner, at least one named
    parent, **and nobody else**.

    That last clause is the whole point of the spread. Requiring only "a
    partner and a parent" admits any photo those two happen to appear in --
    which in practice means the big posed group shots, where the bride and her
    mother stand among twenty guests. Those are `very large group` photos that
    the content model happened to file under `portrait`, and putting one on the
    family spread is not what the spread is for.

    Two tests, because `persons_ids` only lists people the identity model
    actually recognised:

    * every *identified* face belongs to the couple or the named parents, give
      or take `max_extra_people`; and
    * the frame does not hold materially more faces than that -- a crowd where
      only three people were recognised still reads as a crowd, and
      `persons_ids` alone cannot see it.
    """
    photos = photos.copy()
    if Col.PARENT_CATEGORY not in photos.columns:
        photos[Col.PARENT_CATEGORY] = pd.Series(
            [None] * len(photos), index=photos.index, dtype="object")

    if not resolution.resolved():
        return photos, 0

    config = settings()
    slack = int(config.get('max_extra_people', 0))
    face_slack = int(config.get('max_unidentified_faces', 1))

    bride_set = set(resolution.bride_parents)
    groom_set = set(resolution.groom_parents)
    family = bride_set | groom_set | {i for i in (bride_id, groom_id)
                                      if i is not None and not pd.isna(i)}
    portraits = photos[Col.CLUSTER_CONTEXT] == "portrait"

    def classify(row) -> Optional[str]:
        group = set(row[Col.PERSONS_IDS] or [])
        has_bride, has_groom = bride_id in group, groom_id in group
        if not (has_bride or has_groom):
            return None
        if not (group & (bride_set | groom_set)):
            return None

        # Nobody but the family, and not a crowd of strangers around them.
        if len(group - family) > slack:
            return None
        if _crowd_size(row) > len(group) + face_slack:
            return None

        if has_bride and has_groom:
            return BOTH_PARENTS
        if has_bride:
            return BRIDE_PARENTS if group & bride_set else None
        return GROOM_PARENTS if group & groom_set else None

    category = photos.loc[portraits].apply(classify, axis=1)         if int(portraits.sum()) else pd.Series(dtype="object")
    hit = category.notna()
    index = category[hit].index
    photos.loc[index, Col.PARENT_CATEGORY] = category[hit].astype("object")
    photos.loc[index, Col.CLUSTER_CONTEXT] = PARENTS_PORTRAIT
    return photos, int(hit.sum())


def _crowd_size(row) -> int:
    """How many people are in frame, identified or not.

    `n_faces` and `number_bodies` disagree often enough that neither alone is
    trustworthy -- a turned head has a body and no face -- so take the larger.
    A row carrying neither returns 0, which lets the identity test decide on
    its own rather than rejecting the photo on missing data.
    """
    counts = []
    for column in (Col.N_FACES, Col.NUMBER_BODIES):
        value = row.get(column) if hasattr(row, 'get') else None
        if value is not None and not pd.isna(value):
            counts.append(int(value))
    return max(counts) if counts else 0
