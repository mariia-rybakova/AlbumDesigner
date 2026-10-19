"""Scoring a photograph as a cover.

How good a candidate is, as a number. Every term lives here: what the concept
bank says about it (`_concept_score`), its technical quality, how much
affection it shows, whether the people in it are the album's subjects, and
the penalties that push a shot down -- a cut-off subject, a crowd, a table, a
detail shot, tears. `rank_general_covers` is the whole model applied to a
frame; `settings` is the `CONFIGS['covers']` block they all read.

Nothing here chooses anything. It ranks, and `choosing` decides.
"""

import numpy as np
import pandas as pd

from utils.configs import CONFIGS


def settings() -> dict:
    return CONFIGS['covers']


def _concept_score(frame, concepts, logger, aggregate, label, discounted=None):
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

    ``discounted`` maps a concept to a boolean row mask: on those rows the
    concept is held at its lowest score over `frame`, so it neither lifts nor
    sinks them.
    """
    if frame.empty or not concepts:
        return np.zeros(len(frame), dtype=float)

    # Imported here: `src.core` is reached by ProcessStage, which must not gain
    # a hard dependency on the pipeline package for an optional score term.
    from src.pipeline.enrich.timeline import concept_scores

    discounted = discounted or {}
    scored = []
    for concept in concepts:
        try:
            values = np.asarray(concept_scores(frame, concept), dtype=float)
        except Exception as exc:  # noqa: BLE001 - an absent bin costs a term, not the covers
            logger.info(f"cover {label}: {concept} unavailable "
                        f"({type(exc).__name__}: {exc})")
            continue
        mask = discounted.get(concept)
        if mask is not None and np.any(mask):
            values = np.where(mask, values.min(), values)
        scored.append(values)
    if not scored:
        return np.zeros(len(frame), dtype=float)

    combined = np.max(scored, axis=0) if aggregate == 'max' else np.mean(scored, axis=0)
    return np.asarray(_minmax_normalize(list(combined)), dtype=float)


def _quality(frame, logger):
    """How good a *photograph* it is, as against what it is of.

    The positive concepts (`quality_concepts`) less the negative ones
    (`quality_negative_concepts`), each standardised over the candidates first
    so that no concept leads on the scale of its cosines alone, then
    normalised 0-1 like every other term.

    This used to average `portrait`, `smiling`, `affection`, `romance`,
    `softlight` and `happiness` -- five of six about faces and feeling, so it
    measured "close and smiling" and counted affection a second time. On
    53951381 it put the photographer's chosen opening, a wide kiss at a garden
    gate, at 0.35 of the close portraits around it. Every photo in a delivered
    gallery already passed the photographer's edit; what is left to tell apart
    is the background, the light, the frame.
    """
    positive = tuple(settings().get('quality_concepts', ()))
    negative = tuple(settings().get('quality_negative_concepts', ()))
    if frame.empty or not (positive or negative):
        return np.zeros(len(frame), dtype=float)

    from src.pipeline.enrich.timeline import concept_scores

    total, counted = np.zeros(len(frame), dtype=float), 0
    for concepts, sign in ((positive, 1.0), (negative, -1.0)):
        for concept in concepts:
            try:
                values = np.asarray(concept_scores(frame, concept), dtype=float)
            except Exception as exc:  # noqa: BLE001 - an absent bin costs a concept, not the covers
                logger.info(f"cover quality: {concept} unavailable "
                            f"({type(exc).__name__}: {exc})")
                continue
            spread = values.std()
            if spread > 0:
                total += sign * (values - values.mean()) / spread
            counted += 1
    if not counted:
        return np.zeros(len(frame), dtype=float)
    return np.asarray(_minmax_normalize(list(total / counted)), dtype=float)


def _affection(frame, logger):
    """How much the frame shows the two of them meaning it.

    The term this module was missing. A cover has to be *particular* -- a look,
    a touch, a candid moment that reads as mutual -- and nothing here measured
    that: `affection` and `romance` were two of six concepts averaged into
    `quality`, so a frame that was emphatically affectionate and unremarkable
    on the rest scored no better than a well-lit posed portrait.

    The `affection` bin also scores a moved face: 49994361 closed on the two
    of them each wiping away tears at the speeches, not touching and not
    looking at each other. `_tears_penalty` answers that, not this term.

    Held hands are a sign of affection between two people in the frame, not a
    subject of their own. In a frame with no person detected -- a close-up of
    their hands -- `context_concepts` do not count: 49995684 closed on exactly
    that, carried by the gallery's highest holding-hands score.
    """
    context_only = tuple(settings().get('context_concepts', ()))
    empty = ~_shows_people(frame)
    return _concept_score(frame, tuple(settings().get('affection_concepts', ())),
                          logger, 'max', 'affection',
                          discounted={concept: empty for concept in context_only})


def _shows_people(frame):
    """Rows with at least one detected face or body.

    A frame with neither is a detail -- hands, rings, the dress -- whatever its
    tags say. Without body detections nothing can be called a detail: an
    embrace with the faces turned away has no face either, and it is the frame
    `_presence` exists to protect.
    """
    if 'number_bodies' not in frame.columns:
        return np.ones(len(frame), dtype=bool)

    def count(column):
        if column not in frame.columns:
            return np.zeros(len(frame), dtype=float)
        return pd.to_numeric(frame[column], errors='coerce').fillna(0).values

    return (count('n_faces') > 0) | (count('number_bodies') > 0)


def _detail_penalty(frame):
    """1 for a frame that shows no person at all, else 0."""
    return (~_shows_people(frame)).astype(float)


def _significance(frame, subjects=2):
    """0-1: how much of the photo's subject the couple is.

    The couple's two largest body boxes against the foreground blob -- the
    largest salient region BgSegmentation found. Below 1 the foreground runs
    well past the people: a table, flowers, a room, with the two of them small
    in it. On 49995684 the closing that survived the tears bin was the couple
    far off at their own table dabbing their eyes, 0.78 of its foreground; the
    frames worth keeping on three galleries sat at 1.23 to 1.36.

    Saturating on purpose, so a close-up gains nothing over a frame where they
    are simply the subject: rises linearly across `significance_range` and is
    flat above it. `blobDiameter` is measured on BgSegmentation's square mask,
    so the foreground's share of the frame is pi*(d/2)^2 whatever the
    orientation. Frames with no body boxes or no blob read 1 -- nothing to
    measure is not evidence against them, and a detail is `_detail_penalty`'s.

    ``subjects`` is how many bodies make up the people the photo is of: two
    for a couple, and the gallery's main people for a family. At two, a
    family of three is measured on two of its bodies and the frame showing
    all of them reads smaller than the one showing a pair -- 54117560's
    best trio frames graded 0.66 and 0.76 against the pairs' 1.0.
    """
    low, high = settings().get('significance_range', (0.7, 1.2))
    subjects = max(1, int(subjects))

    def ratio(row):
        bodies = _listed(row.get('bodies_info'))
        diameter = row.get('diameter')
        try:
            diameter = float(diameter)
        except (TypeError, ValueError):
            return None
        if not bodies or not diameter > 0:
            return None
        areas = sorted((max(0.0, b.bbox.x2 - b.bbox.x1) * max(0.0, b.bbox.y2 - b.bbox.y1)
                        for b in bodies), reverse=True)
        return sum(areas[:subjects]) / (np.pi * (diameter / 2) ** 2)

    def grade(row):
        r = ratio(row)
        if r is None:
            return 1.0
        return float(np.clip((r - low) / max(high - low, 1e-6), 0.0, 1.0))

    if frame.empty:
        return np.zeros(0, dtype=float)
    return frame.apply(grade, axis=1).astype(float).values


def _tears_penalty(frame, logger):
    """0-1: how surely someone in the frame is crying.

    Tears are a real moment of the day and the wrong one for a cover, and the
    `affection` bin cannot tell them from affection: 49994361 closed on the
    couple each wiping away tears at the speeches. Graded on the raw cosine
    between the two ends of `tears_range`, so the penalty means the same in
    every window rather than always biting on somebody. Zeros when the bin is
    missing.
    """
    from src.pipeline.enrich.timeline import concept_scores

    if frame.empty:
        return np.zeros(0, dtype=float)
    try:
        raw = np.asarray(concept_scores(frame, 'tears'), dtype=float)
    except Exception as exc:  # noqa: BLE001 - a missing bin costs the term, not the covers
        logger.info(f"cover tears: unavailable ({type(exc).__name__}: {exc})")
        return np.zeros(len(frame), dtype=float)
    low, high = settings().get('tears_range', (0.30, 0.42))
    return np.clip((raw - low) / max(high - low, 1e-6), 0.0, 1.0)


def _table_penalty(frame, logger):
    """0-1: how surely the frame is people seated at a table.

    The reception is mostly dinner, and a cover of the two of them sitting
    behind glasses and plates is the album's least particular frame: 49994361
    closed on the bride laughing at the head table, the groom beside her and
    half out of the picture. It matters more now the closing draws on the
    toasts, which are usually drunk sitting down -- `speech` is admitted only
    for its toast subqueries, and this is what keeps a seated toast from
    winning on those alone.

    Graded on the raw cosine across `table_range`, like `_tears_penalty`, so it
    means the same in every window. Zeros when the `seated_at_table` bin is
    missing, which is how it ships until the bin is built and uploaded.
    """
    from src.pipeline.enrich.timeline import concept_scores

    if frame.empty:
        return np.zeros(0, dtype=float)
    try:
        raw = np.asarray(concept_scores(frame, 'seated_at_table'), dtype=float)
    except Exception as exc:  # noqa: BLE001 - a missing bin costs the term, not the covers
        logger.info(f"cover table: unavailable ({type(exc).__name__}: {exc})")
        return np.zeros(len(frame), dtype=float)
    low, high = settings().get('table_range', (0.30, 0.42))
    return np.clip((raw - low) / max(high - low, 1e-6), 0.0, 1.0)


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
    one_and_body = float(grades.get('one_and_body', one))
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
            return one_and_body if _faceless_subject_body(row) else one
        # Neither of them named. Recognising *somebody* here is positive
        # evidence against the frame, not the absence of evidence `faces_only`
        # stands for.
        if people:
            return others
        faces = row.get('n_faces')
        return faces_only if faces is not None and faces == faces and faces > 0 else neither

    return frame.apply(grade, axis=1).astype(float).values


def _listed(value):
    """`value` as a list, or [] -- protobuf containers included, which list fine
    but have no `__iter__` attribute."""
    if value is None or isinstance(value, (str, float)):
        return []
    try:
        return list(value)
    except TypeError:
        return []


def _faceless_subject_body(row):
    """Whether the frame holds a subject-sized body with no face on it.

    That body is a person the detector saw but could not name: in a frame
    naming one of the couple, most likely the other one, face turned away or
    hidden. Subject-sized means at least `presence_body_min_share` of the
    tallest body in the frame, so a guest in the background does not count.

    **Height, not area.** A partner half hidden behind the other keeps their
    height and loses their width: in 53951381's kiss at the garden gate the
    groom's box is 0.21 of the bride's area and 0.84 of her height.

    **A face accounts for one body, not every body it falls in.** The couple
    overlap whenever they embrace, so the visible face sits inside both boxes,
    and testing each body for "holds a face" found no faceless body in the very
    frames this exists for -- the same kiss, with the bride's back to the
    camera, graded `one`. So the bodies are counted against the faces on them.

    Identities placed by body (`faceless_persons_ids`) occupy faceless bodies
    of their own, so only the faceless bodies beyond them are unnamed: a frame
    of the bride alone, recognised from her body, is not also the groom.
    """
    bodies = _listed(row.get('bodies_info'))
    if not bodies:
        return False
    faces = [f for f in _listed(row.get('faces_info')) if getattr(f, 'blurLevel', 0) >= 0]

    def height(b):
        return max(0.0, b.bbox.y2 - b.bbox.y1)

    def inside(f, b):
        return (b.bbox.x1 <= (f.bbox.x1 + f.bbox.x2) / 2 <= b.bbox.x2
                and b.bbox.y1 <= (f.bbox.y1 + f.bbox.y2) / 2 <= b.bbox.y2)

    tallest = max(height(b) for b in bodies)
    share = float(settings().get('presence_body_min_share', 0.5))
    subjects = [b for b in bodies if height(b) >= share * tallest]
    faced = sum(1 for f in faces if any(inside(f, b) for b in subjects))
    unnamed = len(subjects) - min(faced, len(subjects))
    return unnamed > len(_listed(row.get('faceless_persons_ids')))


def _cut_subject_penalty(frame):
    """1 for a frame where one of its main faces runs off the side of the photo.

    The photographer's framing, not the crop: on 49994361 the album closed on
    the bride laughing at the table with the groom half out of the picture, his
    face cut by the right border of the original. No crop brings that back, and
    nothing else in the score saw it -- he is named and his face is detected, so
    presence grades the frame as both of them.

    Only faces count. A body running off the frame is ordinary framing -- a
    half-length portrait cuts both of them at the waist -- while a face at the
    border is a person the reader sees half of. Only the sides and the bottom:
    a face touching the top edge is a close-up cropped at the forehead, which
    is a style. And only faces of subject size, at least `cut_face_min_share`
    of the largest face, so a guest at the edge of a wide shot does not count.
    """
    margin = float(settings().get('cut_edge_margin', 0.01))
    share = float(settings().get('cut_face_min_share', 0.5))

    def grade(row):
        faces = [f for f in _listed(row.get('faces_info')) if getattr(f, 'blurLevel', 0) >= 0]
        if not faces:
            return 0.0
        area = [max(0.0, f.bbox.x2 - f.bbox.x1) * max(0.0, f.bbox.y2 - f.bbox.y1) for f in faces]
        largest = max(area) or 1.0
        for face, size in zip(faces, area):
            if size < share * largest:
                continue
            b = face.bbox
            if b.x1 <= margin or b.x2 >= 1.0 - margin or b.y2 >= 1.0 - margin:
                return 1.0
        return 0.0

    if frame.empty:
        return np.zeros(0, dtype=float)
    return frame.apply(grade, axis=1).astype(float).values


def _subquery_grades(queries):
    """Each listed subquery's grade: 1.0 for the first, falling by position."""
    span = max(1, len(queries))
    return {name: (span - i) / span for i, name in enumerate(queries)}


def _subquery_affinity(frame, queries):
    """How well the frame matches the cover's subqueries, graded by their order.

    **Soft, not the label.** The label is one argmax over the photo's class's
    phrases, and it can turn on nothing: 53951381's kiss at the garden gate
    read "posing for a portrait" at 0.276 against "kissing" at 0.271, and
    since posing is listed last that 0.005 cost it 0.45 of the cover score.
    So the frame's similarity to every listed phrase is turned into a softmax
    at CLIP's own temperature (`subquery_temperature`), and the affinity is the
    expected grade under it: a frame that is nearly as much a kiss as a pose
    scores nearly as a kiss does.

    The phrase vectors are the tagger's own (`queries_file_v2/_v3`), so this
    scores against exactly what the label was argmax'd over. Falls back to the
    label when the frame has no embeddings or no listed phrase has a vector.
    """
    grades = _subquery_grades(queries)
    soft = _soft_subquery_affinity(frame, queries, grades)
    if soft is not None:
        return soft
    return frame['image_subquery_content'].map(
        lambda name: grades.get(name, 0.0)).astype(float).values


def _soft_subquery_affinity(frame, queries, grades):
    if frame.empty or 'embedding' not in frame.columns or 'model_version' not in frame.columns:
        return None
    try:
        names, vectors = _subquery_vectors(tuple(queries), int(frame['model_version'].iloc[0]))
        if not names:
            return None
        matrix = np.vstack([np.asarray(e, dtype=np.float32).ravel() for e in frame['embedding']])
        if matrix.shape[1] != vectors.shape[1]:
            return None
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    except Exception:  # noqa: BLE001 - the label is still there to fall back on
        return None
    temperature = float(settings().get('subquery_temperature', 0.01))
    logits = (matrix @ vectors.T) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ np.array([grades[name] for name in names], dtype=float)


_SUBQUERY_VECTORS = {}


def _subquery_vectors(queries, model_version):
    """(names, unit vectors) of the listed subqueries the tagger's bank holds."""
    key = (queries, model_version)
    if key not in _SUBQUERY_VECTORS:
        from src.readers.query_bank import read_pkl_file
        bank = read_pkl_file(CONFIGS['queries_file_v2' if model_version == 1 else 'queries_file_v3'])
        found = {}
        for phrases in bank.values():
            for name, vector in phrases.items():
                if name in queries and name not in found:
                    found[name] = np.atleast_2d(np.asarray(vector, dtype=np.float32))[0]
        names = [name for name in queries if name in found]
        vectors = (np.vstack([found[name] for name in names]) if names
                   else np.zeros((0, 1), dtype=np.float32))
        if names:
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        _SUBQUERY_VECTORS[key] = (names, vectors)
    return _SUBQUERY_VECTORS[key]


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
    floor = float(settings().get('affection_presence_floor', 1.0))

    rank = np.asarray(_minmax_normalize(
        [float(v) if v == v else 0.0 for v in frame['image_order']]), dtype=float)
    presence = _presence(frame, bride_id, groom_id)
    # Confidence that the affection on show is *theirs*, never below `floor`.
    affection_gate = floor + (1.0 - floor) * presence

    return (
        weights.get('affection', 0.0) * _affection(frame, logger) * affection_gate
        + weights.get('quality', 0.0) * _quality(frame, logger)
        + weights.get('subquery', 0.0) * _subquery_affinity(frame, queries)
        + weights.get('presence', 0.0) * presence
        + weights.get('significance', 0.0) * _significance(frame)
        # `image_order` is a rank where 0 is best, so the *low* end is rewarded.
        + weights.get('rank', 0.0) * (1.0 - rank)
        + _orientation_preference(frame)
        - weights.get('crowd', 0.0) * _crowd_penalty(frame)
        - weights.get('detail', 0.0) * _detail_penalty(frame)
        - weights.get('tears', 0.0) * _tears_penalty(frame, logger)
        - weights.get('cut_subject', 0.0) * _cut_subject_penalty(frame)
        - weights.get('table', 0.0) * _table_penalty(frame, logger)
    )


def _orientation_preference(frame):
    """The cover orientation term, shared by both routes.

    A cover box is a single large box, so a frame that fills it is preferred --
    a preference, never a filter. Filtering on orientation is what cost 53507032
    its covers (see `choose_good_wedding_images`), and `find_single_box_layout`
    already falls back to any single-box layout, so a portrait that wins on
    merit still gets a box to sit in.

    Returns zeros rather than raising when the frame has no orientation column,
    so a caller that never needed one keeps working.
    """
    weight = float(settings().get('weights', {}).get('orientation', 0.0))
    if not weight or 'image_orientation' not in frame.columns:
        return np.zeros(len(frame), dtype=float)
    preferred = settings().get('preferred_orientation', 'landscape')
    return weight * (frame['image_orientation'] == preferred).astype(float).values


def _ranked_ids(frame, queries, logger, bride_id=None, groom_id=None):
    """Candidate ids for one cover, best first."""
    if frame.empty:
        return []
    scored = frame.copy()
    scored['__score'] = _score_covers(scored, queries, logger, bride_id, groom_id)
    scored = scored.sort_values('__score', ascending=False, kind='stable')
    ids = scored['image_id'].tolist()
    return _preferred_orientation_first(scored, ids, logger)


def rank_general_covers(frame, logger, presence=None, subjects=2):
    """Cover candidates of a gallery that is not a wedding, best first.

    The wedding score's terms that are about the *photograph* and not about
    the bride and groom: quality (`_quality`, background, light, frame),
    the content model's rank (weighted 0 by `general_weights`: the model
    reads every gallery as a wedding), how much of the frame its people are
    (`significance`), and the penalties for a frame with nobody in it and for
    a subject cut off by the border. Orientation is the same preference, and
    the same floor (`_preferred_orientation_first`) puts a good-enough
    landscape on the cover.

    ``presence`` is the caller's 0-1 answer to "are these the people this
    album is about" -- the overlap with the gallery's main people. Without one
    the term is left out rather than guessed. ``subjects`` is how many people
    those are, which is how many bodies `_significance` measures.
    """
    if frame.empty:
        return []
    weights = {**settings().get('weights', {}), **settings().get('general_weights', {})}
    scored = frame.copy()
    rank = np.asarray(_minmax_normalize(
        [float(v) if v == v else 0.0 for v in scored['image_order']]), dtype=float) \
        if 'image_order' in scored.columns else np.zeros(len(scored))
    score = (weights.get('quality', 0.0) * _quality(scored, logger)
             + weights.get('rank', 0.0) * (1.0 - rank)
             + weights.get('significance', 0.0) * _significance(scored, subjects)
             + _orientation_preference(scored)
             - weights.get('detail', 0.0) * _detail_penalty(scored)
             - weights.get('cut_subject', 0.0) * _cut_subject_penalty(scored))
    if presence is not None:
        score = score + weights.get('presence', 0.0) * np.asarray(presence, dtype=float)
    scored['__score'] = score
    scored = scored.sort_values('__score', ascending=False, kind='stable')
    return _preferred_orientation_first(scored, scored['image_id'].tolist(), logger)


def _preferred_orientation_first(scored, ids, logger):
    """The best frame in `preferred_orientation` first, when it is good enough.

    A cover is one large box, and photographers do not want a portrait there
    when the session has a landscape that will do: 53951381 opened on a close
    portrait over the landscape kiss its photographer chose. The orientation
    term could not say this -- the scores favour close frames over wide ones,
    and in a gallery whose wide shots are its landscapes that gap was 1.27,
    more than the affection weight. A bigger orientation weight would just
    override good portraits on galleries without the gap.

    So it is a rule with a floor: the best landscape leads when it scores at
    least `orientation_floor` of the best frame. A weak landscape still loses
    to a strong portrait, which keeps the failure of the old hard filter away
    -- 53507032, where the only landscape opened and closed the album.
    """
    floor = float(settings().get('orientation_floor', 0.0))
    if not floor or 'image_orientation' not in scored.columns or len(ids) < 2:
        return ids
    preferred = settings().get('preferred_orientation', 'landscape')
    best = scored.iloc[0]
    if best['image_orientation'] == preferred or best['__score'] <= 0:
        return ids
    fitting = scored[scored['image_orientation'] == preferred]
    if fitting.empty:
        return ids
    challenger = fitting.iloc[0]
    if challenger['__score'] < floor * best['__score']:
        return ids
    logger.info(f"covers: {preferred} {challenger['image_id']} leads over "
                f"{best['image_orientation']} {best['image_id']} "
                f"({challenger['__score']:.2f} of {best['__score']:.2f}, "
                f"floor {floor:.0%})")
    # The list's own element, not the row's: a row hands back a numpy scalar,
    # and the ids leave this module as plain values.
    lead = next(i for i in ids if i == challenger['image_id'])
    return [lead] + [i for i in ids if i != lead]


def _minmax_normalize(values):
    """Min-max normalize to [0, 1]. Returns zeros if empty or constant."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]
