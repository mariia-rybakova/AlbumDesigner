"""Who a category is about, and preferring the photos that hold them.

`person_score` already asks "does this photo contain one of the people the
request named". That is a different question from "is the couple in the cake
photo", and it answers the second one only by accident -- on 49995684 the
request named ``[58, 100, 71]``, so *no* signal in the pipeline preferred a
cake photo with the bride and groom in it. Twelve of the gallery's thirteen
cake photos held one of them; the album took the one that did not.

So the subject is stated per category instead of inferred from the request:
`cake cutting` is about the couple, `groom walking the aisle` is about the
groom, and a candidate that does not contain them is a worse answer for that
category than one that does, however it scores.

**A preference, not a filter.** If nothing in the category holds the subject
the category is offered back untouched, because a spread built from imperfect
candidates beats a spread that silently disappeared. That is also what keeps
this safe on galleries where the identity model found little: the worst case is
the behaviour we had before.

Identity is the only test. Not `n_faces`, which counts faces without saying
whose, and not the subquery, which is a caption rather than an observation --
the frame the album used for `groom walking the aisle` was captioned "guests
watching ceremony".
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from src.pipeline.contracts import Col
from utils.configs import CONFIGS

#: The two roles a category can be about, mapped onto the resolved identities.
BRIDE = "bride"
GROOM = "groom"


def settings() -> dict:
    return CONFIGS.get("subject", {}) or {}


def enabled() -> bool:
    return bool(settings().get("enabled", False))


# -- who a class is about ----------------------------------------------------
#
# One table, three readers. `select.cpsat` scores these as a bonus and excludes
# contradictions outright; `select.pick` and `select.preselect` read them
# through `prefer_subject` below. It lived in `cpsat` and so applied only when
# CP-SAT was the picker -- the loop had no notion of a class's subject at all,
# and neither did the `yes` categories, which is where `may kiss bride` is
# settled.


def _is_missing(value) -> bool:
    """True for None and for a pandas null, which `bride_id` can both be."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _people_of(value) -> set:
    return set(value) if isinstance(value, (list, tuple, set)) else set()


def _solo(who):
    """Only that person in frame -- `persons_ids == [id]` in the loop."""
    def rule(people, bride, groom):
        target = bride if who == 'bride' else groom
        return target is not None and people == {target}
    return rule


def _couple_alone(people, bride, groom) -> bool:
    """Both of them and nobody else.

    The loop reaches the same place from the other side, pairing "has both"
    with `n_faces == 2 or number_bodies == 2`; stating it as the identity set
    says it once and does not depend on the face count being right.
    """
    if bride is None or groom is None:
        return False
    return people == {bride, groom}


def _includes(who):
    def rule(people, bride, groom):
        target = bride if who == 'bride' else groom
        return target is not None and target in people
    return rule


def _includes_either(people, bride, groom) -> bool:
    return bool(people & {i for i in (bride, groom) if i is not None})


#: Class -> ``(rule, subjects, exclusive)``. The rule is what a photo of that
#: class should hold; `subjects` names who it is about, which is what makes a
#: *wrong* identity distinguishable from a *missing* one. Absent from the table
#: means the class is not about a particular person and rank decides on its own.
IDENTITY_RULES = {
    # Exclusive: the class is *only* about its subject, so another face in
    # frame with the subject absent is the wrong photo.
    'bride': (_solo('bride'), ('bride',), True),
    'groom': (_solo('groom'), ('groom',), True),
    'bride and groom': (_couple_alone, ('bride', 'groom'), True),
    'bride getting dressed': (_includes('bride'), ('bride',), True),
    'getting hair-makeup': (_includes('bride'), ('bride',), True),
    'bride walking the aisle': (_includes('bride'), ('bride',), True),
    'groom walking the aisle': (_includes('groom'), ('groom',), True),

    # Not exclusive: other people belong in these. Parents and flower girls
    # walk the aisle, and the party classes are about a group -- so a frame
    # naming someone other than the couple is not a wrong photo, it is just
    # not the *preferred* one. Penalising it emptied `walking the aisle`
    # outright on 53459898, where the couple is detected in none of its
    # frames, losing a scripted moment of the wedding to a detection gap.
    'bride party': (_includes('bride'), ('bride',), False),
    'groom party': (_includes('groom'), ('groom',), False),
    'walking the aisle': (_includes_either, ('bride', 'groom'), False),

    # The couple's own moments. Absent from this table until now, which is why
    # nothing preferred a cake photo with the couple in it: on 49995684 twelve
    # of thirteen `cake cutting` frames hold one of them and the album took the
    # thirteenth. Not exclusive -- guests crowd round the cake, the send-off is
    # a corridor of them, and the kiss is often shot over a shoulder.
    'couple': (_includes_either, ('bride', 'groom'), False),
    'kiss': (_includes_either, ('bride', 'groom'), False),
    'may kiss bride': (_includes_either, ('bride', 'groom'), False),
    'first dance': (_includes_either, ('bride', 'groom'), False),
    'cake cutting': (_includes_either, ('bride', 'groom'), False),
    'send off': (_includes_either, ('bride', 'groom'), False),
}


#: Classes that mean *the two of them*, so one of them is not enough.
COUPLE_ALONE_CLASSES = tuple(
    category for category, (rule, _subjects, _exclusive) in IDENTITY_RULES.items()
    if rule is _couple_alone)


def one_half_with_an_outsider(persons_ids: Any, bride_id, groom_id) -> bool:
    """One of the couple, a third person, and not the other half.

    The distinction that matters for a class meaning *the two of them*. Three
    things look alike in `persons_ids` and are not alike at all:

    * ``[bride, groom]`` -- the photo the class is for;
    * ``[bride]`` -- almost always the couple with one face unrecognised, which
      is ordinary and common (29 of 120 on 49995684). Demoting these would cost
      a quarter of the class to a detection gap;
    * ``[bride, someone]`` -- the bride standing with a groomsman. The model saw
      two people, recognised both, and neither is the other half.

    Only the third is a wrong photo, and it is rare: 9 of 120.
    """
    people = _people_of(persons_ids)
    couple = {i for i in (bride_id, groom_id) if i is not None}
    if len(couple) < 2 or not people:
        return False
    return bool(people & couple) and not couple <= people and bool(people - couple)


def refile_misfiled_couple(photos, bride_id, groom_id, logger=None):
    """Move "one of them plus a stranger" out of the couple classes.

    Returns ``(photos, moved)``. A relabel rather than a preference because it
    has to reach the photos `select.preselect` commits unconditionally: on
    49995684 the couple spread carried a bride-and-groomsman shot that was one
    of the request's own ``aiMetadata.photoIds``, and a preference cannot
    displace a hand pick. Re-filing keeps the photo the user asked for and only
    corrects the class the content model gave it -- they picked a photo, not a
    category.
    """
    destination = settings().get("misfiled_couple_class")
    if not enabled() or not destination or photos is None or photos.empty:
        return photos, 0
    if Col.PERSONS_IDS not in photos.columns or Col.CLUSTER_CONTEXT not in photos.columns:
        return photos, 0

    misfiled = photos[Col.CLUSTER_CONTEXT].isin(COUPLE_ALONE_CLASSES) & \
        photos[Col.PERSONS_IDS].apply(
            lambda ids: one_half_with_an_outsider(ids, bride_id, groom_id))

    moved = int(misfiled.sum())
    if moved:
        photos.loc[misfiled, Col.CLUSTER_CONTEXT] = destination
        if logger:
            logger.info(
                f"subject: re-filed {moved} photo(s) from the couple classes to "
                f"'{destination}' -- they hold one of the couple and a third "
                f"person, not the two of them")
    return photos, moved


def _scene_runs(photos: pd.DataFrame, max_gap: float) -> List[List[Any]]:
    """Index labels of each time-contiguous run, in time order.

    ``general_time`` rather than ``image_time``: on a gallery whose EXIF is
    unusable the first is a synthetic monotonic day rebuilt from scene order
    and the second is the unusable original, and `image_order` is not a time
    axis at all -- it is a quality rank, so splitting on it interleaves photos
    from opposite ends of the day.
    """
    if photos.empty or Col.GENERAL_TIME not in photos.columns:
        return []
    ordered = photos.assign(__t=pd.to_numeric(photos[Col.GENERAL_TIME], errors="coerce")) \
                    .dropna(subset=["__t"]) \
                    .sort_values("__t", kind="stable")
    runs, current, previous = [], [], None
    for label, moment in zip(ordered.index, ordered["__t"]):
        if previous is not None and moment - previous > max_gap:
            runs.append(current)
            current = []
        current.append(label)
        previous = moment
    if current:
        runs.append(current)
    return runs


def lopsided_couple_scenes(photos: pd.DataFrame, bride_id, groom_id) -> Optional[pd.Series]:
    """Couple-class scenes that hold one of them and never the other.

    The frame-by-frame test cannot see this. On 49996919 the album opened on
    the bride being kissed by her **father** -- a first look with dad, shot
    against the same wall as the couple portraits, and filed `bride and groom`
    by a content model that sees a tuxedo and a wedding dress. The father is
    never recognised in any frame of it, and the bride is named in some and not
    others, so every frame reads as the ordinary and common "the couple with
    one face missed":

        11548148002  persons_ids []     11548148006  persons_ids [9]
        11548148022  persons_ids [9]    11548148023  persons_ids [9]

    `one_half_with_an_outsider` needs the third person recognised; the parents
    relabel needs one of the couple *and* a named parent; `_presence` grades
    each frame alone. All three are blind to it, and because the scene spans
    three different presence grades, demoting any one of them merely promotes
    another frame of the same scene -- measured across a sweep of the presence
    weight, no setting picks a correct opening without also picking the least
    affectionate photo in the gallery.

    Aggregated over the scene it is unambiguous. A real couple scene names the
    groom in *some* of its frames; this one never does. On 49996919 exactly one
    run of the 183 couple-class frames is lopsided -- all seven frames of the
    first look, catching all four of the ones that reached the cover rule --
    and **none** of the 106 frames naming both is touched.

    Returns a boolean mask, or ``None`` when the question cannot be asked.
    """
    if photos is None or photos.empty:
        return None
    if Col.PERSONS_IDS not in photos.columns or Col.CLUSTER_CONTEXT not in photos.columns:
        return None

    couple = {i for i in (bride_id, groom_id) if i is not None and not _is_missing(i)}
    if len(couple) < 2:
        return None

    config = settings()
    classes = tuple(config.get("scene_classes", ()))
    if not classes:
        return None
    max_gap = float(config.get("scene_max_gap", 120))
    min_named = int(config.get("scene_min_named", 2))
    min_partner_frames = int(config.get("scene_min_partner_frames", 5))

    # Absence only means something if the identity model can find that person
    # at all. On a gallery where the groom is never recognised -- a detection
    # failure, or the identities never resolved -- every couple scene would
    # look lopsided and the rule would empty the class.
    seen = photos[Col.PERSONS_IDS].apply(_people_of)
    reach = {i: int(seen.apply(lambda people, who=i: who in people).sum()) for i in couple}
    if min(reach.values()) < min_partner_frames:
        return None

    mask = pd.Series(False, index=photos.index)
    candidates = photos[photos[Col.CLUSTER_CONTEXT].isin(classes)]
    for run in _scene_runs(candidates, max_gap):
        people = [seen.loc[label] for label in run]
        counts = {i: sum(1 for p in people if i in p) for i in couple}
        present = [i for i, n in counts.items() if n >= min_named]
        absent = [i for i, n in counts.items() if n == 0]
        # One of them recurs across the scene and the other never appears.
        if len(present) == 1 and len(absent) == 1:
            mask.loc[run] = True
    return mask


def refile_lopsided_couple_scenes(photos, bride_id, groom_id, logger=None):
    """Move whole one-sided scenes out of the couple classes.

    Returns ``(photos, moved)``. A relabel for the same reason
    `refile_misfiled_couple` is one: it has to reach the photos
    `select.preselect` commits unconditionally, and on 49996919 the misread
    scene supplied one of the request's own ``aiMetadata.photoIds``.
    """
    destination = settings().get("misfiled_couple_class")
    if not enabled() or not destination or photos is None or photos.empty:
        return photos, 0

    mask = lopsided_couple_scenes(photos, bride_id, groom_id)
    if mask is None:
        return photos, 0

    moved = int(mask.sum())
    if moved:
        photos.loc[mask, Col.CLUSTER_CONTEXT] = destination
        if logger:
            logger.info(
                f"subject: re-filed {moved} photo(s) from the couple classes to "
                f"'{destination}' -- whole scenes holding one of the couple "
                f"repeatedly and the other never")
    return photos, moved


def required_roles(category: Any) -> Tuple[str, ...]:
    """Which of the couple this category is about, empty when it is nobody."""
    if not enabled():
        return ()
    entry = IDENTITY_RULES.get(category)
    return tuple(entry[1]) if entry else ()


def is_unknown_category(category: Any) -> bool:
    """True for a class that names no real content (`other`, `None`).

    Both the string ``'None'`` and a genuine null occur in `cluster_context`:
    the focus CSV carries a row literally named ``None'``, and photos the
    content model could not place carry a missing value.
    """
    if category is None:
        return True
    try:
        if pd.isna(category):
            return True
    except (TypeError, ValueError):
        pass
    return str(category) in tuple(settings().get("unknown_categories", ()))


def _identities(roles: Sequence[str], bride_id, groom_id) -> Tuple[Any, ...]:
    """The role names resolved to identities, dropping any that is unknown."""
    wanted = []
    for role in roles:
        value = bride_id if role == BRIDE else groom_id
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        wanted.append(value)
    return tuple(wanted)


def holds_any(persons_ids: Any, identities: Iterable[Any]) -> bool:
    """Whether a row's ``persons_ids`` contains any of ``identities``."""
    if not isinstance(persons_ids, (list, tuple, set, pd.Series)):
        return False
    present = set(persons_ids)
    return any(identity in present for identity in identities)


def subject_mask(frame: pd.DataFrame, category: Any, bride_id, groom_id
                 ) -> Optional[pd.Series]:
    """Rows of ``frame`` that hold the category's subject.

    ``None`` when the question does not apply: the feature is off, the category
    is about nobody, the identities were never resolved, or the frame has no
    ``persons_ids`` to read.
    """
    identities = _identities(required_roles(category), bride_id, groom_id)
    if not identities or frame is None or frame.empty:
        return None
    if Col.PERSONS_IDS not in frame.columns:
        return None
    return frame[Col.PERSONS_IDS].apply(lambda ids: holds_any(ids, identities))


def prefer_subject(frame: pd.DataFrame, category: Any, bride_id, groom_id,
                   logger=None) -> pd.DataFrame:
    """Narrow a category's candidates to those holding its subject.

    Returns ``frame`` unchanged whenever narrowing would empty it, or when the
    category is about nobody.
    """
    mask = subject_mask(frame, category, bride_id, groom_id)
    if mask is None:
        return frame

    kept = int(mask.sum())
    if kept == 0 or kept == len(frame):
        if kept == 0 and logger:
            logger.info(
                f"subject: no photo of '{category}' holds the "
                f"{'/'.join(required_roles(category))}; offering all "
                f"{len(frame)} rather than none")
        return frame

    if logger:
        logger.info(
            f"subject: '{category}' narrowed {len(frame)} -> {kept} photos that "
            f"hold the {'/'.join(required_roles(category))}")
    return frame[mask]


def identity_tiers(frame: pd.DataFrame) -> Sequence[pd.DataFrame]:
    """A frame split into preference tiers for covering a named identity.

    The identity guarantee exists so a person the user named appears at all, and
    any photo of them satisfies it -- which is why it used to be settled on
    score alone, and why on 49995684 two of the album's `other` photos are there
    to cover identities 58 and 71. `other` and `None` are budgeted at 0% in the
    focus CSV precisely because they carry no content worth a spread, so taking
    one to cover a person puts a photo in the album that nothing else would
    have kept.

    Tiers, best first:

    1. a real class that is not about the couple -- the guest is the subject;
    2. a real class about the couple -- they are in the couple's photo, which is
       weaker but still a photo of them somewhere meaningful;
    3. whatever is left, i.e. `other` and `None`.

    Tier 1 comes before tier 2 because a named guest standing in a couple
    portrait is incidental to it, and that spread would have existed anyway.
    """
    if frame is None or frame.empty or Col.CLUSTER_CONTEXT not in frame.columns:
        return (frame,) if frame is not None else ()

    category = frame[Col.CLUSTER_CONTEXT]
    unknown = category.apply(is_unknown_category)
    couple = category.apply(lambda value: bool(required_roles(value)))

    return (frame[~unknown & ~couple], frame[~unknown & couple], frame[unknown])
