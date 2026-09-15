import random
import numpy as np
import pandas as pd
from utils.configs import CONFIGS


def _find_single_box_layout(layouts_df, orientation):
    """Find a single-box layout matching the image orientation.

    Prefers large boxes, falls back to any single-box layout.
    """
    if orientation == "landscape":
        # Prefer large landscape or large square
        candidates = [key for key, layout in layouts_df.iterrows() if layout["max landscapes"] == 1 and (
            layout['right_large_landscape'] == 1 or layout["left_large_landscape"] == 1 or
            layout["left_large_square"] == 1 or layout["right_large_square"] == 1)]
        if not candidates:
            # Fall back: any single-box layout that accepts a landscape
            candidates = [key for key, layout in layouts_df.iterrows() if layout["max landscapes"] >= 1
                          and layout["number of boxes"] == 1]
    else:
        # Prefer large portrait or large square
        candidates = [key for key, layout in layouts_df.iterrows() if layout["max portraits"] == 1 and (
            layout['left_large_portrait'] == 1 or layout["right_large_portrait"] == 1 or
            layout["left_large_square"] == 1 or layout["right_large_square"] == 1)]
        if not candidates:
            # Fall back: any single-box layout that accepts a portrait
            candidates = [key for key, layout in layouts_df.iterrows() if layout["max portraits"] >= 1
                          and layout["number of boxes"] == 1]
    # Last resort: any single-box layout
    if not candidates:
        candidates = [key for key, layout in layouts_df.iterrows() if layout["number of boxes"] == 1]
    return candidates


def get_design_id(layout_df, number_of_boxes, logger):
    # Get all layout keys where "number of boxes" == 1
    img_layouts = [key for key, layout in layout_df.iterrows() if layout["number of boxes"] == number_of_boxes]

    # Ensure there are at least 2 valid layouts
    if len(img_layouts) < 1:
        logger.error("Not enough layouts with one image to select two distinct ones.")
        return None

    return img_layouts[0]


def _pick_time_cluster(df, position="first"):
    if df.empty:
        return None
    return df["time_cluster"].min() if position == "first" else df["time_cluster"].max()


#: Time axes to draw the cover windows along, best first.
#:
#: `general_time` before `image_time`: the two are the same seconds whenever the
#: EXIF is trustworthy, but when it is not, `general_time` has been rebuilt from
#: scene order into a synthetic monotonic day while `image_time` still holds the
#: unusable original. Two of the four validation galleries carry **2 distinct
#: `image_time` values across 528 and 582 photos**, so sorting those by
#: `image_time` does not order the day at all.
_TIME_AXES = ("general_time", "image_time")

#: What share of the candidates each cover is drawn from, in time order -- the
#: opening from the earliest quarter of them, the closing from the latest.
COVER_FRACTION = 0.25


def _pick_cover_subset(df, position="first", window_size=10, fraction=COVER_FRACTION):
    """Candidates for one cover: the earliest or latest `fraction` of them.

    A quarter of the candidates **by count**, taken in time order -- not a
    quarter of the elapsed time, and not the first or last `time_cluster`.

    All three have now been tried and the first two both failed on real albums:

    - `time_cluster` min/max put both covers in one cluster, because a wedding's
      couple frames are often bunched into one part of the day. On one album all
      ten landscape couple photos sat in cluster 1 of 2.
    - A quarter of the elapsed *time* breaks whenever the gallery is not one
      continuous session. Gallery 52894932 holds two shoots **five days apart**
      -- a single 114.7-hour gap between consecutive photos -- so the first
      quarter of its time span contained **88% of the photos**, the window was
      effectively the whole gallery, and the opening cover came from 79% of the
      way through the day.

    Counting is immune to both. It is also the rule the rest of the pipeline
    already follows: `src/pipeline/enrich/timeline.py` works in *positions*
    rather than wall-clock minutes, for exactly this reason -- ordering by time
    is trustworthy, measuring distances along it is not.

    Nothing is trimmed by anything but time order here: which frame in the
    quarter is *good* is for the ranking below to decide.
    """
    if df.empty:
        return df

    axis = next((column for column in _TIME_AXES if column in df.columns), None)
    if axis is None:
        # No time at all: the caller's order is the only signal there is.
        return (df.head(window_size) if position == "first" else df.tail(window_size)).copy()

    frame = df.copy()
    frame["__t"] = pd.to_numeric(frame[axis], errors="coerce")
    # Rows with no time are dropped rather than sorted to one end, where they
    # would fill a quarter with photos of unknown place in the day.
    frame = frame.dropna(subset=["__t"]).sort_values("__t", ascending=True, kind="stable")
    if frame.empty:
        return frame.drop(columns="__t")

    take = max(1, int(round(len(frame) * fraction)))
    quarter = frame.head(take) if position == "first" else frame.tail(take)
    return quarter.drop(columns="__t")


def _select_by_priority_from_subset(df_subset, queries_primary, queries_fallback):
    """
    Try in order:
      1) landscape + primary
      2) landscape + fallback
      3) portrait  + primary
      4) portrait  + fallback
    Returns ordered list of image_ids.
    """
    if df_subset.empty:
        return []

    def pick(orientation, queries):
        sub = df_subset[
            (df_subset["image_orientation"] == orientation) &
            (df_subset["image_subquery_content"].isin(queries))
            ].copy()
        if sub.empty:
            return []
        sub["__q"] = pd.Categorical(sub["image_subquery_content"], categories=queries, ordered=True)
        # Break ties within the same subquery by image_order, best first.
        # `image_order` is the content model's `selectionOrder`, a rank where
        # **0 is best** -- `update_photos_ranks` sets a hand-picked photo to 0,
        # and the selection stage sorts it ascending for the same reason. This
        # sorted descending, so it was picking the worst-ranked frame of every
        # tie it broke.
        if "image_order" in sub.columns:
            sub = sub.sort_values(["__q", "image_order"], ascending=[True, True])
        else:
            sub = sub.sort_values("__q")
        return sub["image_id"].tolist()

    for orientation, queries in [
        ("landscape", queries_primary),
        ("landscape", queries_fallback),
        ("portrait", queries_primary),
        ("portrait", queries_fallback),
    ]:
        ids = pick(orientation, queries)
        if ids:
            return ids

    picked = df_subset[
        (df_subset["image_subquery_content"].isin(["unknown_bride_and_groom"]))
    ].copy()

    if picked.empty:
        return []
    else:
        return picked["image_id"].tolist()


#: Subqueries each cover prefers, best first. Affinity is graded by position in
#: the list rather than used as a filter, so a frame tagged something else stays
#: a candidate and competes on quality.
FIRST_COVER_QUERIES = (
    'bride and groom kissing romantically',
    'bride and groom kissing',
    'bride and groom hugging or kissing',
    'bride and groom smiling at each other',
    'bride and groom not formal pose',
    'bride and groom walking together',
    'bride and groom during the ceremony',
    # Last on purpose. A posed portrait is the two of them *presenting*, and it
    # is the commonest couple subquery by a wide margin -- 147 of 211 on
    # 53227528 -- so ranking it near the top made the cover the album's most
    # generic frame rather than its most particular one.
    'bride and groom posing for a portrait',
)
LAST_COVER_QUERIES = (
    'bride and groom kissing romantically',
    'bride and groom kissing',
    'bride and groom hugging or kissing',
    'bride and groom dancing',
    'bride and groom smiling at each other',
    'bride and groom not formal pose',
    'bride and groom walking together',
    'bride and groom during the ceremony',
    'bride and groom posing for a portrait',
)


def settings() -> dict:
    return CONFIGS['covers']


def _concept_score(frame, concepts, logger, aggregate, label):
    """Cosine against a bank of concepts, combined and normalised over `frame`.

    ``aggregate`` is the whole difference between the two terms that use this.
    Quality takes the **mean**: a cover should be a good photograph on every
    count at once, so strong on one and weak on the rest is not the same
    thing. Affection takes the **max**: a frame shows one kind of it -- a kiss,
    or held hands, or a look -- and averaging across the others would punish a
    photo for being emphatically one thing, which is the photo we are after.

    Zeros when the concepts cannot be scored -- no embeddings, no
    `model_version`, a bin missing for this model version. A gallery that
    cannot be projected loses the term and is decided by subquery and rank,
    which is what the rule did before these terms existed.
    """
    if frame.empty or not concepts:
        return np.zeros(len(frame), dtype=float)

    # Imported here: `src.core` is reached by ProcessStage, which must not gain
    # a hard dependency on the pipeline package for an optional score term.
    from src.pipeline.enrich.timeline import concept_scores

    scored = []
    for concept in concepts:
        try:
            scored.append(np.asarray(concept_scores(frame, concept), dtype=float))
        except Exception as exc:  # noqa: BLE001 - an absent bin costs a term, not the covers
            logger.info(f"cover {label}: {concept} unavailable "
                        f"({type(exc).__name__}: {exc})")
    if not scored:
        return np.zeros(len(frame), dtype=float)

    combined = np.max(scored, axis=0) if aggregate == 'max' else np.mean(scored, axis=0)
    return np.asarray(_minmax_normalize(list(combined)), dtype=float)


def _quality(frame, logger):
    """How good a photograph it is, averaged over the quality concepts."""
    return _concept_score(frame, tuple(settings().get('quality_concepts', ())),
                          logger, 'mean', 'quality')


def _affection(frame, logger):
    """How much the frame shows the two of them meaning it.

    The term this module was missing. A cover has to be *particular* -- a look,
    a touch, a candid moment that reads as mutual -- and nothing here measured
    that: `affection` and `romance` were two of six concepts averaged into
    `quality`, so a frame that was emphatically affectionate and unremarkable
    on the rest scored no better than a well-lit posed portrait.
    """
    return _concept_score(frame, tuple(settings().get('affection_concepts', ())),
                          logger, 'max', 'affection')


def _presence(frame, bride_id, groom_id):
    """How surely both of them are in the frame, graded rather than required.

    This replaces a hard filter, and the filter is why the best frames were
    unreachable. `persons_ids` is built from face clusters, so a photo with no
    detected face carries no identity at all -- and an embrace with faces
    turned away, buried in a shoulder, or shot from behind is exactly that. On
    53227528, 53 of the 204 `bride and groom` frames have no face and so no
    identity, and every one was dropped before it could be scored. Requiring a
    face is close to requiring them to look at the camera.

    So presence became a preference. A frame naming both of them still wins
    every tie; a faceless one has to earn the cover on affection and quality,
    against a handicap.

    **A missing identity and a wrong one are not the same evidence.** The
    grades above were blind to the difference: a frame in which nobody was
    recognised and a frame in which somebody was recognised and it is neither
    of them both scored `faces_only`. The first may well be the couple -- that
    is the case this grade exists to protect. The second cannot be, so `others`
    grades it apart.

    It is worth being exact about what this does *not* fix, because the gallery
    that prompted it is not an instance of it. On 49996919 the album opened on
    the bride being kissed by her father, and that frame carries
    ``persons_ids == [9]`` -- the **bride**, recognised from her body with
    ``n_faces == 0``. The father is not recognised in it or in any other frame
    of the scene, so there is no wrong identity to see: the frame grades `one`,
    exactly like the ordinary and common "the couple with one face missed".
    `enrich.couple_scenes` is what answers that, by reading the whole scene
    instead of the frame. This grade is for the case where the identity model
    did see the other person and said who they were.
    """
    grades = settings().get('presence_grades', {})
    both = float(grades.get('both', 1.0))
    one = float(grades.get('one', 0.6))
    faces_only = float(grades.get('faces_only', 0.35))
    neither = float(grades.get('neither', 0.2))
    others = float(grades.get('others', 0.05))

    def grade(row):
        people = row['persons_ids']
        people = set(people) if isinstance(people, (list, tuple, set)) else set()
        named = sum(1 for i in (bride_id, groom_id) if i is not None and i in people)
        if named >= 2:
            return both
        if named == 1:
            return one
        # Neither of them named. Recognising *somebody* here is positive
        # evidence against the frame, not the absence of evidence `faces_only`
        # stands for.
        if people:
            return others
        faces = row.get('n_faces')
        return faces_only if faces is not None and faces == faces and faces > 0 else neither

    return frame.apply(grade, axis=1).astype(float).values


def _subquery_affinity(frame, queries):
    """1.0 for the first listed subquery, falling to 0 for anything unlisted."""
    order = {name: i for i, name in enumerate(queries)}
    span = max(1, len(queries))
    return frame['image_subquery_content'].map(
        lambda name: (span - order[name]) / span if name in order else 0.0
    ).astype(float).values


def _crowd_penalty(frame):
    """How far past the couple the face count runs, as a 0-1 share.

    A cover is of the two of them. `n_faces` at two is the couple; the confetti
    frame that opened 53507032 carried six.
    """
    slack = int(settings().get('crowd_slack', 1))
    allowed = int(settings().get('min_faces', 2)) + slack
    excess = (frame['n_faces'].astype(float) - allowed).clip(lower=0)
    return (excess / excess.max()).fillna(0.0).values if excess.max() > 0 else excess.values


def _score_covers(frame, queries, logger, bride_id=None, groom_id=None):
    """Score every candidate for one cover. Higher is better.

    Affection can be **gated on presence** rather than added beside it, by
    setting `affection_presence_floor` below 1.0. The idea is that `affection`
    otherwise scores whatever feeling the frame shows between whoever happens
    to be in it, and multiplying by presence would make it mean "affection
    between *the two of them*".

    **It ships off, because it was measured and it does not do that.** What the
    multiplier actually does is discount the frames carrying the *least*
    identity hardest -- and the misread frame it was written for, the bride and
    her father on 49996919, names the bride and so is discounted *less* than a
    genuine faceless embrace of the couple. Swept over that gallery's real 46
    candidates it never changes whether a father frame opens the album, only
    which one does. `enrich.couple_scenes` addresses that case instead, by
    taking the scene out of the couple classes rather than out-weighing it.

    Kept as a lever, at the value that leaves the score the plain sum it has
    always been.
    """
    weights = settings().get('weights', {})
    preferred = settings().get('preferred_orientation', 'landscape')
    floor = float(settings().get('affection_presence_floor', 1.0))

    rank = np.asarray(_minmax_normalize(
        [float(v) if v == v else 0.0 for v in frame['image_order']]), dtype=float)
    orientation = (frame['image_orientation'] == preferred).astype(float).values
    presence = _presence(frame, bride_id, groom_id)
    # Confidence that the affection on show is *theirs*, never below `floor`.
    affection_gate = floor + (1.0 - floor) * presence

    return (
        weights.get('affection', 0.0) * _affection(frame, logger) * affection_gate
        + weights.get('quality', 0.0) * _quality(frame, logger)
        + weights.get('subquery', 0.0) * _subquery_affinity(frame, queries)
        + weights.get('presence', 0.0) * presence
        # `image_order` is a rank where 0 is best, so the *low* end is rewarded.
        + weights.get('rank', 0.0) * (1.0 - rank)
        + weights.get('orientation', 0.0) * orientation
        - weights.get('crowd', 0.0) * _crowd_penalty(frame)
    )


def _ranked_ids(frame, queries, logger, bride_id=None, groom_id=None):
    """Candidate ids for one cover, best first."""
    if frame.empty:
        return []
    scored = frame.copy()
    scored['__score'] = _score_covers(scored, queries, logger, bride_id, groom_id)
    scored = scored.sort_values('__score', ascending=False, kind='stable')
    return scored['image_id'].tolist()


def _candidate_base(chosen_df, bride_id, groom_id, logger):
    """Every couple frame, with presence left to the scoring.

    The class is the gate and nothing else is. This used to require both
    identities *and* two faces, and drop everything else before scoring --
    which quietly made "a cover shows their faces" the rule. It cannot be:
    `persons_ids` comes from face clusters, so a frame with no detected face
    has no identity either, and the embrace shot with faces turned away fails
    both halves at once. On 49995684 that removed 18 of 117 couple frames and
    on 53227528 53 of 204, none of which could be looked at.

    `_presence` grades the same evidence instead, so a faceless frame is
    handicapped rather than absent and has to be a better photograph to win.
    Set `require_identities` to restore the old gate.
    """
    classes = tuple(settings().get('cover_classes', ('bride and groom',)))
    couple = chosen_df[chosen_df["cluster_context"].isin(classes)].copy()

    if not settings().get('require_identities', False):
        return couple

    named = couple[couple["persons_ids"].apply(
        lambda x: isinstance(x, list) and bride_id in x and groom_id in x)]
    if not named.empty:
        return named
    if not couple.empty:
        logger.info(f"cover candidates: no frame names both of them, "
                    f"falling back to {len(couple)} couple frames")
    return couple


def _positions(frame):
    """Each row's place in time order, as a 0-1 share of the candidates."""
    axis = next((column for column in _TIME_AXES if column in frame.columns), None)
    if axis is None or frame.empty:
        return {}
    ordered = frame.assign(__t=pd.to_numeric(frame[axis], errors='coerce')) \
                   .dropna(subset=['__t']) \
                   .sort_values('__t', kind='stable')
    span = max(1, len(ordered) - 1)
    return {row: i / span for i, row in enumerate(ordered['image_id'])}


def _separated(opening, candidates, frame):
    """First candidate that is a different photo, far enough from `opening`.

    Distinctness is absolute -- opening and closing are never the same frame,
    which is the fault this replaces. The separation is a preference: if no
    candidate clears it, the nearest distinct one is still better than a
    repeat.
    """
    distinct = [c for c in candidates if c != opening]
    if not distinct:
        return None

    minimum = float(settings().get('min_separation', 0.0))
    if minimum <= 0:
        return distinct[0]

    places = _positions(frame)
    here = places.get(opening)
    if here is None:
        return distinct[0]

    far = [c for c in distinct
           if c in places and abs(places[c] - here) >= minimum]
    return far[0] if far else distinct[0]


def get_important_imgs(data_df, bride_groom_df, logger):
    try:
        if bride_groom_df is not None:
            chosen_df = data_df.copy()
            if not bride_groom_df.empty:
                chosen_df = bride_groom_df.copy()
        else:
            chosen_df = data_df.copy()

        bride_id = chosen_df["bride_id"].values[0]
        groom_id = chosen_df["groom_id"].values[0]

        base = _candidate_base(chosen_df, bride_id, groom_id, logger)

        subset_first = _pick_cover_subset(base, position="first", window_size=10)
        first_page_ids = _ranked_ids(subset_first, FIRST_COVER_QUERIES, logger,
                                     bride_id, groom_id)

        subset_last = _pick_cover_subset(base, position="last", window_size=10)
        last_page_ids = _ranked_ids(subset_last, LAST_COVER_QUERIES, logger,
                                    bride_id, groom_id)

        # Fall back only for the cover that has no candidate of its own, and
        # rank **ascending** -- `image_order` is a rank where 0 is best, so
        # sorting it descending handed back the worst photo in the gallery.
        if not first_page_ids or not last_page_ids:
            logger.warning("No ideal cover images found, falling back to highest ranked images.")
            pool = data_df
            if bride_groom_df is not None and not bride_groom_df.empty:
                pool = bride_groom_df
            all_image_ids = pool.sort_values("image_order", ascending=True)["image_id"].tolist()

            if not all_image_ids:
                logger.error("No images available in the album.")
                return [], []

            # One photo per side, and never the photo the other side holds.
            # Taking the whole ranked list for the missing side left the other
            # side nothing to be distinct from, which is how the collision this
            # replaces survived into the fallback.
            def _best_other_than(taken):
                return next((i for i in all_image_ids if i not in taken),
                            all_image_ids[0])

            if not first_page_ids:
                first_page_ids = [_best_other_than(last_page_ids)]
            if not last_page_ids:
                last_page_ids = [_best_other_than(first_page_ids)]

        return first_page_ids, last_page_ids

    except Exception as e:
        logger.error(f"Error inside the function get_important_imgs {e}")
        return None, None


def _minmax_normalize(values):
    """Min-max normalize to [0, 1]. Returns zeros if empty or constant."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _select_cover_image_ids(pool_df, pool_bg, logger):
    """Return (first_page_ids, last_page_ids), one photo each and never the same.

    Both lists arrive score-ordered from `get_important_imgs`, so the opening
    is simply its best candidate. The closing takes the best candidate of its
    own that is a different photo and far enough away in the day, rather than
    the one most *dissimilar* to the opening: dissimilarity blended distance
    with rank and never looked at whether the frame was any good.
    """
    first_candidates, last_candidates = get_important_imgs(pool_df, pool_bg, logger)

    # Degenerate case: at least one candidate list is missing/empty.
    # Forward whatever we got, normalizing None to [] so the caller always gets (list, list).
    if not first_candidates or not last_candidates:
        first_page_ids = first_candidates if first_candidates else []
        last_page_ids = last_candidates if last_candidates else []
        return first_page_ids, last_page_ids

    first_id = first_candidates[0]
    last_id = _separated(first_id, last_candidates, pool_df)
    last_page_ids = [last_id] if last_id is not None else []
    return [first_id], last_page_ids


def _drop_cover_duplicates(df, cover_rows, logger, manual_selection=False):
    """Drop body photos that repeat a cover, unless the user chose them.

    Never fires on the manual route. There the photo list *is* the user's
    decision -- they picked each one and expect to find it in the album, so a
    near-duplicate of the cover among them is a choice to honour, not a
    mistake to correct. Dropping even one silently is what made 44573310 look
    broken: 55 photos chosen, 16 placed. The rule is for the AI route, where
    the pool is the pipeline's own doing and a repeated cover is its own error.

    Taking the cover out of the body was never enough. The frame beside it in
    the burst is a different `image_id` and survives, so the album closes on a
    photograph the reader has already turned past: on 53227528 the back cover
    and a body photo are the same forehead kiss one step wider, cosine 0.840,
    consecutive ids.

    CP-SAT's own near-duplicate exclusion cannot catch this and should not be
    asked to. It fires at 0.97 -- burst-identical frames -- and it runs during
    *selection*, before anyone knows which photo will be promoted to a cover.
    That is decided later, here, over the pool selection returned. So the check
    belongs at the point the covers are finally known.

    Only against the covers, and only a handful of them: two similar photos
    inside the body are an ordinary editing choice, while a repeated cover
    reads as a mistake. The threshold has measured room -- 0.840 for the real
    pair on 53227528 against 0.719 for the next nearest, and a maximum of
    0.567 anywhere on 49995684, which loses nothing at any threshold to 0.75.
    """
    if manual_selection:
        # Logged, because the alternative is someone finding the cover shot
        # twice in a manual album and having nothing to explain it.
        if logger:
            logger.info("covers: manual selection -- keeping every chosen photo, "
                        "cover duplicates not pruned")
        return df

    threshold = float(settings().get('cover_duplicate_similarity', 0.0))
    if threshold <= 0 or df.empty or cover_rows is None or len(cover_rows) == 0:
        return df
    if 'embedding' not in df.columns or 'embedding' not in cover_rows.columns:
        return df

    def unit(vector):
        try:
            vector = np.asarray(vector, dtype=np.float32)
        except (TypeError, ValueError):
            return None
        if vector.ndim != 1 or vector.size == 0:
            return None
        norm = float(np.linalg.norm(vector))
        return None if norm == 0 else vector / norm

    reference = [v for v in (unit(e) for e in cover_rows['embedding']) if v is not None]
    if not reference:
        return df

    similarity = {}
    for image_id, embedding in zip(df['image_id'], df['embedding']):
        candidate = unit(embedding)
        if candidate is None or candidate.shape != reference[0].shape:
            continue
        similarity[image_id] = max(float(r @ candidate) for r in reference)

    over = sorted((score, image_id) for image_id, score in similarity.items()
                  if score >= threshold)
    cap = int(settings().get('cover_duplicate_max_drop', 3))
    dropped = [image_id for _score, image_id in over[-cap:]] if cap > 0 else []
    if not dropped:
        return df

    if logger:
        detail = ", ".join(f"{i} ({similarity[i]:.3f})" for i in dropped)
        logger.info(f"covers: dropped {len(dropped)} body photo(s) repeating a cover "
                    f"above {threshold}: {detail}")
    return df[~df['image_id'].isin(dropped)]


def choose_good_wedding_images(df, bride_groom_df, logger, prune_duplicates=True,
                               manual_selection=False):
    # Orientation is a score term, not a pre-filter. Filtering on it first cost
    # 53507032 its covers: the gallery had 42 landscapes and 48 couple frames
    # showing both faces, but only *one* frame in both sets, so the candidate
    # base collapsed to that single photo and it opened and closed the album.
    first_page_ids, last_page_ids = _select_cover_image_ids(df, bride_groom_df, logger)

    if bride_groom_df is not None:
        if not bride_groom_df.empty and bride_groom_df['image_id'].isin(first_page_ids).any() and bride_groom_df['image_id'].isin(last_page_ids).any():
            first_cover_image_df = bride_groom_df[bride_groom_df['image_id'].isin(first_page_ids)]
            last_cover_image_df = bride_groom_df[bride_groom_df['image_id'].isin(last_page_ids)]
        else:
            first_cover_image_df = df[df['image_id'].isin(first_page_ids)]
            last_cover_image_df = df[df['image_id'].isin(last_page_ids)]
    else:
        # Get rows corresponding to selected images
        first_cover_image_df = df[df['image_id'].isin(first_page_ids)]
        last_cover_image_df = df[df['image_id'].isin(last_page_ids)]

    # Remove selected images from main dataframe
    df = df[~df['image_id'].isin(first_page_ids + last_page_ids)]
    # Only for the caller that keeps the pruned frame. `enrich.key_pages` calls
    # this for the two ids alone and discards the body, so pruning there is
    # both wasted over the whole gallery and a second, misleading log line.
    if prune_duplicates:
        df = _drop_cover_duplicates(
            df, pd.concat([first_cover_image_df, last_cover_image_df]), logger,
            manual_selection=manual_selection)

    return df, first_page_ids, first_cover_image_df, last_page_ids, last_cover_image_df


def choose_good_non_wedding_images(df, number_of_images, logger):
    # Validate input DataFrame
    required_columns = {'persons_ids', 'image_order', 'image_id'}

    if not required_columns.issubset(df.columns):
        missing_cols = required_columns - set(df.columns)
        logger.error(f"Error: DataFrame is missing required columns: {missing_cols}")
        return df, None, None, None, None

    # Collect all unique people IDs in the dataset
    unique_people_ids = set()
    for _, row in df.iterrows():
        if isinstance(row['persons_ids'], list):  # Ensure it's a list
            unique_people_ids.update(row['persons_ids'])

    if not unique_people_ids:
        logger.warning("Warning: No unique people IDs found in dataset.")
        selected_images_df = df.nlargest(number_of_images, 'image_order')

    # Find images that contain all unique people IDs
    selected_images_df = df[
        df['persons_ids'].apply(lambda x: set(x).issuperset(unique_people_ids) if isinstance(x, list) else False)]

    if selected_images_df.empty:
        logger.warning("No images matched. Selecting top by n_faces.")
        # Take top-N by number of faces
        selected_images_df = df.nlargest(number_of_images, 'n_faces')
    else:
        # Keep your current priority by image_order first
        selected_images_df = selected_images_df.nlargest(number_of_images, 'image_order')

        # If still fewer than required, fill the remaining by highest n_faces from the whole df (excluding already chosen)
        if len(selected_images_df) < number_of_images:
            remaining = number_of_images - len(selected_images_df)
            chosen_ids = set(selected_images_df['image_id'].tolist())
            fill_pool = df[~df['image_id'].isin(chosen_ids)]
            fill_df = fill_pool.nlargest(remaining, 'n_faces')

            selected_images_df = pd.concat([selected_images_df, fill_df], ignore_index=False).drop_duplicates(
                subset='image_id').head(number_of_images)

    if selected_images_df.empty:
        logger.warning("Warning: No images selected based on image_order.")
        return df, None, None, None, None

    # Extract image IDs
    selected_image_ids = selected_images_df['image_id'].tolist()
    mid_index = len(selected_image_ids) // 2
    first_image_id = selected_image_ids[:mid_index]
    last_image_id = selected_image_ids[mid_index:]

    first_image_df = df[df['image_id'].isin(first_image_id)]
    last_image_df = df[df['image_id'].isin(last_image_id)]

    # Remove selected images from the original DataFrame
    df_without_selected = df[~df['image_id'].isin(selected_image_ids)]

    logger.info(f"Selected cover images: {selected_image_ids}")

    return df_without_selected, first_image_id, last_image_id, first_image_df, last_image_df


def generate_first_last_pages(message, df, logger):
    first_last_pages_data_dict = dict()

    if message.pagesInfo.get("firstPage"):
        if message.content.get('is_wedding', True):
            # The same flag ProcessStage reads a few lines later: on the manual
            # route the chosen photos are the user's, and none of them is the
            # pipeline's to discard.
            manual_selection = message.content.get('manual_selection', False)
            df, first_images_ids, first_imgs_df, last_images_ids, last_imgs_df = choose_good_wedding_images(df,
                                                                                                            message.content.get(
                                                                                                                'bride and groom'),
                                                                                                            logger,
                                                                                                            manual_selection=manual_selection)
        else:
            df, first_images_ids, last_images_ids, first_imgs_df, last_imgs_df = choose_good_non_wedding_images(df, 1,
                                                                                                                logger)

        if message.pagesInfo.get("firstPage"):
            layouts_df = message.designsInfo[f"firstPage_layouts_df"]
            if not first_imgs_df.empty:
                cover_layouts = _find_single_box_layout(layouts_df, first_imgs_df["image_orientation"].values[0])
                if cover_layouts:
                    first_last_pages_data_dict["firstPage"] = {
                        'design_id': cover_layouts[0],
                        'first_images_ids': first_images_ids,
                        'first_images_df': first_imgs_df,
                    }
                else:
                    logger.warning("No matching single-box layout found for firstPage")
        else:
            logger.warning("For this album theres no first page cover image")

        if message.pagesInfo.get('lastPage'):
            layouts_df = message.designsInfo[f"lastPage_layouts_df"]
            if not last_imgs_df.empty:
                cover_layouts = _find_single_box_layout(layouts_df, last_imgs_df["image_orientation"].values[0])
                if cover_layouts:
                    first_last_pages_data_dict['lastPage'] = {
                        'design_id': cover_layouts[0],
                        'last_images_ids': last_images_ids,
                        'last_images_df': last_imgs_df,
                    }
                else:
                    logger.warning("No matching single-box layout found for lastPage")
        else:
            logger.warning("For this album theres no last page cover image")

    return df, first_last_pages_data_dict
