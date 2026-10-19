"""Choosing which photographs open and close the album.

Narrows the gallery to the candidates worth scoring -- the opening and
closing slice of the day, filtered by the cover queries -- asks `scoring` to
rank them, and settles the result: one photo for the front, one for the back,
far enough apart, in the orientation the design wants, with near-duplicates
of the chosen covers pruned from the album body.

`choose_good_wedding_images` is the wedding path and
`choose_good_non_wedding_images` the other one.
"""

import numpy as np
import pandas as pd

from src.covers.scoring import (_minmax_normalize, _orientation_preference, _ranked_ids, settings)


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


def opening_fraction(gallery_size):
    """The opening's window as a share of the candidates, by gallery size.

    A small gallery has few couple frames, and a quarter of them is a handful
    from the first minutes of the shoot: on 53951381 (199 photos) the window
    held 16 of 66 couple frames and closed ten frames before the landscape kiss
    at the garden gate the photographer wanted, in the same session. So the
    window widens as the gallery shrinks -- `opening_window`'s `small` share at
    or below its photo count, `large` at or above its, linear in between.
    """
    config = settings().get('opening_window') or {}
    small_at, small = config.get('small', (200, COVER_FRACTION))
    large_at, large = config.get('large', (500, COVER_FRACTION))
    if not gallery_size or gallery_size <= small_at:
        return float(small)
    if gallery_size >= large_at:
        return float(large)
    share = (gallery_size - small_at) / float(large_at - small_at)
    return float(small) + share * (float(large) - float(small))


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
    'romantic couple moment',
    'bride and groom dancing',
    # The evening's moments, which close a day and do not open one. Spelled as
    # the query bank spells them: four entries here named phrases the bank does
    # not hold ('bride and groom cutting cake', '... making toasts for the
    # speech', ...), so no photo could ever carry them and the closing gave
    # the cake and the toasts no credit at all.
    'bride and groom feeding each other cake',
    'bride and groom cutting cake together',
    'guests raising glasses for toast',
    'bride and groom smiling at each other',
    'bride and groom not formal pose',
    'bride and groom walking together',
    'bride and groom during the ceremony',
    'bride and groom posing for a portrait',
)


def _candidate_base(chosen_df, bride_id, groom_id, logger, classes=None, subqueries=None):
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
    if classes is None:
        classes = tuple(settings().get('cover_classes', ('bride and groom',)))
    couple = chosen_df[chosen_df["cluster_context"].isin(classes)].copy()
    # A class that holds the moment only in some of its shots -- `speech`, whose
    # toasts are a closing and whose speeches are not -- is narrowed to them.
    for category, allowed in (subqueries or {}).items():
        if allowed and 'image_subquery_content' in couple.columns:
            wrong = (couple["cluster_context"] == category) & \
                    ~couple["image_subquery_content"].isin(tuple(allowed))
            couple = couple[~wrong]

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

        # Candidates by class from the whole pool. `bride_groom_df` holds the
        # `bride and groom` class alone, so drawing on it made every other
        # entry of `cover_classes` unreachable -- `couple` and `kiss` were
        # listed and never once a candidate.
        source = data_df if "cluster_context" in data_df.columns else chosen_df
        base = _candidate_base(source, bride_id, groom_id, logger)
        # The closing has moments of its own: the day ends on the cake, the
        # toasts, a kiss. The opening does not open on them.
        closing_base = _candidate_base(
            source, bride_id, groom_id, logger,
            classes=tuple(settings().get('closing_cover_classes',
                                         settings().get('cover_classes', ('bride and groom',)))),
            subqueries=settings().get('closing_class_subqueries', {}))

        subset_first = _pick_cover_subset(base, position="first", window_size=10,
                                          fraction=opening_fraction(len(data_df)))
        first_page_ids = _ranked_ids(subset_first, FIRST_COVER_QUERIES, logger,
                                     bride_id, groom_id)

        subset_last = _pick_cover_subset(closing_base, position="last", window_size=10)
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


def _kept(preferred, pool_ids, taken=()):
    """The first of `preferred` still in the pool and not already taken."""
    return next((i for i in preferred or () if i in pool_ids and i not in taken), None)


def _select_cover_image_ids(pool_df, pool_bg, logger, preferred=None):
    """Return (first_page_ids, last_page_ids), one photo each and never the same.

    **`preferred` wins where it can.** It is what `enrich.key_pages` chose over
    the whole gallery, and `select.preselect` commits it, so it is normally in
    the pool. Choosing again here, over the few selected couple frames, judged
    a different field: the latest quarter of ~15 photos rather than of ~130, so
    on 49994361 the album closed on a frame the whole-gallery rule had not
    picked -- one with the groom half out of the picture. Each side falls back
    to the rule below only when its preferred photo is gone.

    Both lists arrive score-ordered from `get_important_imgs`, so the opening
    is simply its best candidate. The closing takes the best candidate of its
    own that is a different photo and far enough away in the day, rather than
    the one most *dissimilar* to the opening: dissimilarity blended distance
    with rank and never looked at whether the frame was any good.
    """
    first_candidates, last_candidates = get_important_imgs(pool_df, pool_bg, logger)

    if preferred is not None and settings().get('reuse_key_pages', True):
        pool_ids = set(pool_df['image_id'])
        opening = _kept(preferred.opening, pool_ids)
        closing = _kept(preferred.closing, pool_ids, taken={opening})
        if opening is not None or closing is not None:
            if opening is None:
                opening = next((i for i in first_candidates or () if i != closing), None)
            if closing is None and opening is not None:
                closing = _separated(opening, last_candidates or [], pool_df)
            logger.info(f"covers: key pages kept -- opening {opening}"
                        f"{'' if opening in (preferred.opening or ()) else ' (re-chosen)'}, "
                        f"closing {closing}"
                        f"{'' if closing in (preferred.closing or ()) else ' (re-chosen)'}")
            return ([opening] if opening is not None else [],
                    [closing] if closing is not None else [])

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


def drop_cover_duplicates(df, cover_rows, logger, manual_selection=False):
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
                               manual_selection=False, preferred=None):
    # Orientation is a score term, not a pre-filter. Filtering on it first cost
    # 53507032 its covers: the gallery had 42 landscapes and 48 couple frames
    # showing both faces, but only *one* frame in both sets, so the candidate
    # base collapsed to that single photo and it opened and closed the album.
    first_page_ids, last_page_ids = _select_cover_image_ids(df, bride_groom_df, logger,
                                                            preferred=preferred)

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
        df = drop_cover_duplicates(
            df, pd.concat([first_cover_image_df, last_cover_image_df]), logger,
            manual_selection=manual_selection)

    return df, first_page_ids, first_cover_image_df, last_page_ids, last_cover_image_df


def _rank_non_wedding_covers(frame, number_of_images):
    """The best cover candidates of a non-wedding gallery, best first.

    Mirrors `_score_covers` on the two terms such a gallery can supply. It is
    ranked rather than filtered for the same reason the wedding covers are:
    `image_order` is the content model's rank where **0 is best**, so
    `nlargest` on it was handing back the worst photographs in the gallery,
    and orientation is a preference that a better photograph may outweigh.
    """
    if frame.empty:
        return frame
    scored = frame.copy()
    rank = np.asarray(_minmax_normalize(
        [float(v) if v == v else 0.0 for v in scored['image_order']]), dtype=float)
    weights = settings().get('weights', {})
    # `image_order` is a rank where 0 is best, so the *low* end is rewarded.
    scored['__score'] = (weights.get('rank', 0.0) * (1.0 - rank)
                         + _orientation_preference(scored))
    scored = scored.sort_values('__score', ascending=False, kind='stable')
    return scored.drop(columns='__score').head(number_of_images)


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
        selected_images_df = _rank_non_wedding_covers(selected_images_df, number_of_images)

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
    # One photo per side, opening first. `len // 2` handed the opening
    # `ids[:0] == []` whenever a single photo was asked for -- and the caller
    # drops a cover whose frame is empty -- so every album on this route opened
    # on an empty box while the closing took the only photo chosen.
    split = max(1, len(selected_image_ids) // 2)
    first_image_id = selected_image_ids[:split]
    last_image_id = selected_image_ids[split:]

    first_image_df = df[df['image_id'].isin(first_image_id)]
    last_image_df = df[df['image_id'].isin(last_image_id)]

    # Remove selected images from the original DataFrame
    df_without_selected = df[~df['image_id'].isin(selected_image_ids)]

    logger.info(f"Selected cover images: {selected_image_ids}")

    return df_without_selected, first_image_id, last_image_id, first_image_df, last_image_df
