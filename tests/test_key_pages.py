"""Tests for ``enrich.key_pages``.

The rule itself is `src/core/key_pages.py`, which ProcessStage has always used
and which this substage calls unchanged. What is worth guarding is the wiring:
that the substage runs over the whole gallery without disturbing it, that its
answer survives the message boundary, and that a gallery it cannot serve loses
its covers rather than its album.

    python -m pytest tests/test_key_pages.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import ENRICH, AlbumContext, Col, KeyPages  # noqa: E402
from src.pipeline.contracts import DesignSpec, GalleryFacts  # noqa: E402
from src.core.key_pages import (  # noqa: E402
    COVER_FRACTION,
    _pick_cover_subset,
    _select_by_priority_from_subset,
)
from src.pipeline.enrich.key_pages import CLOSING, OPENING  # noqa: E402
from src.pipeline.registry import get  # noqa: E402
from utils.configs import CONFIGS  # noqa: E402

BRIDE, GROOM = 101, 202

#: `get_important_imgs` logs from inside its own try/except, so the direct-call
#: tests need a logger rather than None.
import logging as _logging
_QUIET_LOG = _logging.getLogger("test_key_pages")
_QUIET_LOG.addHandler(_logging.NullHandler())


def couple_gallery(n=24, orientation="landscape"):
    """A gallery whose couple photos span the day, earliest first.

    Half carry an opening-worthy subquery and half a closing-worthy one, so the
    priority ladder has something to find at both ends.
    """
    openers = ["bride and groom smiling at each other", "bride and groom posing for a portrait"]
    closers = ["bride and groom dancing", "bride and groom smiling at each other"]
    rows = []
    for i in range(n):
        subquery = (openers if i < n // 2 else closers)[i % 2]
        rows.append({
            Col.IMAGE_ID: 1000 + i,
            Col.IMAGE_ORDER: float(i % 7),
            Col.IMAGE_ORIENTATION: orientation,
            Col.IMAGE_TIME: 1_700_000_000 + i * 60,
            Col.GENERAL_TIME: i * 60,
            Col.PERSONS_IDS: [BRIDE, GROOM],
            Col.N_FACES: 2,
            Col.CLUSTER_CONTEXT: "bride and groom",
            Col.IMAGE_SUBQUERY_CONTENT: subquery,
            Col.BRIDE_ID: BRIDE,
            Col.GROOM_ID: GROOM,
            Col.EMBEDDING: np.linspace(i, i + 1, 8).astype(np.float32),
        })
    return pd.DataFrame(rows)


def run(photos, is_wedding=True, pages=None):
    context = AlbumContext(
        photos=photos,
        facts=GalleryFacts(is_wedding=is_wedding),
        designs=DesignSpec(pages=pages if pages is not None else {}),
    )
    return get("enrich.key_pages")()(context)


# -- composition -----------------------------------------------------------


def test_registered_in_enrich():
    assert "enrich.key_pages" in ENRICH
    assert get("enrich.key_pages").name == "enrich.key_pages"


def test_runs_after_the_enrichments_it_reads():
    """It needs `cluster_context`, the identities and the subquery tags, so it
    has to come after the substages that derive them."""
    order = list(ENRICH)
    for earlier in ("enrich.content_class", "enrich.identities", "enrich.semantic_tags"):
        assert order.index(earlier) < order.index("enrich.key_pages")


# -- the wedding path ------------------------------------------------------


def test_tags_one_opening_and_one_closing():
    context = run(couple_gallery())

    assert not context.failed
    assert len(context.key_pages.opening) == 1
    assert len(context.key_pages.closing) == 1
    assert context.key_pages.opening != context.key_pages.closing

    tagged = context.photos[context.photos[Col.KEY_PAGE] != ""]
    assert sorted(tagged[Col.KEY_PAGE]) == [CLOSING, OPENING]


def test_the_tag_matches_the_sidecar():
    context = run(couple_gallery())
    photos = context.photos

    opening = photos.loc[photos[Col.KEY_PAGE] == OPENING, Col.IMAGE_ID].tolist()
    closing = photos.loc[photos[Col.KEY_PAGE] == CLOSING, Col.IMAGE_ID].tolist()

    assert opening == context.key_pages.opening
    assert closing == context.key_pages.closing


def test_every_row_gets_the_column():
    """Empty string rather than NaN, so consumers can compare without a null
    check and the `provides` contract holds across the whole frame."""
    context = run(couple_gallery())
    assert Col.KEY_PAGE in context.photos.columns
    assert not context.photos[Col.KEY_PAGE].isna().any()


def test_the_gallery_is_left_intact():
    """`choose_good_wedding_images` returns the frame with the covers removed --
    that pruning is for ProcessStage's benefit and must not reach enrich, or
    selection would never see the two best couple photos."""
    photos = couple_gallery()
    before = list(photos[Col.IMAGE_ID])

    context = run(photos)

    assert list(context.photos[Col.IMAGE_ID]) == before


def test_portrait_only_gallery_still_gets_covers():
    """Landscape is a preference, not a requirement."""
    context = run(couple_gallery(orientation="portrait"))
    assert context.key_pages.opening and context.key_pages.closing


# -- where in the day each cover comes from --------------------------------
#
# `_pick_cover_subset` is shared with ProcessStage, so these guard both callers.


def axis_frame():
    """`image_time` unusable, `general_time` the rebuilt sequence.

    This is what an artificial-time gallery looks like: the EXIF collapsed onto
    a couple of values, and a synthetic monotonic day beside it. The two orders
    disagree completely -- `image_time` ascending is 3, 1, 2.
    """
    return pd.DataFrame({
        Col.IMAGE_ID: [1, 2, 3],
        Col.IMAGE_TIME: [900, 900, 100],
        Col.GENERAL_TIME: [0, 1800, 3600],
    })


def spread_frame(n=12):
    """Candidates evenly spread across an hour."""
    return pd.DataFrame({
        Col.IMAGE_ID: list(range(1, n + 1)),
        Col.GENERAL_TIME: [i * 300 for i in range(n)],
        Col.IMAGE_ORDER: [float(i) for i in range(n)],
    })


def test_the_two_covers_come_from_opposite_ends():
    frame = spread_frame()

    first = _pick_cover_subset(frame, "first")
    last = _pick_cover_subset(frame, "last")

    assert set(first[Col.IMAGE_ID]).isdisjoint(set(last[Col.IMAGE_ID]))
    assert max(first[Col.GENERAL_TIME]) < min(last[Col.GENERAL_TIME])


def test_each_cover_comes_from_a_quarter_of_the_candidates():
    """A quarter **by count**, in time order."""
    frame = spread_frame(n=12)

    first = _pick_cover_subset(frame, "first")
    last = _pick_cover_subset(frame, "last")

    assert len(first) == 3 and len(last) == 3
    assert list(first[Col.IMAGE_ID]) == [1, 2, 3]
    assert list(last[Col.IMAGE_ID]) == [10, 11, 12]


def test_a_gap_in_the_gallery_does_not_widen_the_window():
    """Counting rather than measuring elapsed time. Gallery 52894932 holds two
    shoots five days apart -- one 114.7-hour gap between consecutive photos --
    and a quarter of its *time span* contained 88% of the photos, so the
    opening cover came from 79% of the way through the day.
    """
    frame = pd.DataFrame({
        Col.IMAGE_ID: list(range(1, 13)),
        # eleven photos minutes apart, then one five days later
        Col.GENERAL_TIME: [i * 300 for i in range(11)] + [11 * 300 + 5 * 86400],
        Col.IMAGE_ORDER: [float(i) for i in range(12)],
    })

    first = _pick_cover_subset(frame, "first")

    assert len(first) == 3, "the window is three of twelve however long the gap is"
    assert list(first[Col.IMAGE_ID]) == [1, 2, 3]


def test_a_single_candidate_still_yields_one():
    frame = spread_frame(n=2)

    assert len(_pick_cover_subset(frame, "first")) == 1
    assert len(_pick_cover_subset(frame, "last")) == 1


def test_a_quarter_is_more_than_one_photo_so_quality_can_decide():
    """The whole point of a window rather than an edge: the ranking below gets
    a choice, instead of being handed the single earliest frame."""
    first = _pick_cover_subset(spread_frame(n=12), "first")

    assert len(first) > 1


def test_one_time_cluster_no_longer_collapses_the_two_covers():
    """The bug this replaced. `time_cluster` used to decide the windows, taking
    all rows at its min for the opening and its max for the closing -- so a
    gallery whose couple frames are bunched into a single cluster opened and
    closed on two shots of the same moment. On the reviewed album all ten
    landscape couple photos sat in cluster 1 of 2.
    """
    frame = spread_frame()
    frame["time_cluster"] = 1

    first = _pick_cover_subset(frame, "first")
    last = _pick_cover_subset(frame, "last")

    assert set(first[Col.IMAGE_ID]).isdisjoint(set(last[Col.IMAGE_ID]))
    assert max(first[Col.GENERAL_TIME]) < min(last[Col.GENERAL_TIME])


def test_general_time_beats_image_time():
    """On an artificial-time gallery the EXIF order is meaningless, and picking
    along it put one validation gallery's closing photo earlier in the day than
    its opening one."""
    frame = axis_frame()

    assert list(_pick_cover_subset(frame, "first")[Col.IMAGE_ID]) == [1]
    assert list(_pick_cover_subset(frame, "last")[Col.IMAGE_ID]) == [3]


def test_photos_with_no_time_are_dropped_not_sorted_to_one_end():
    """Otherwise a quarter fills up with photos of unknown place in the day."""
    frame = pd.DataFrame({
        Col.IMAGE_ID: [1, 2, 3],
        Col.GENERAL_TIME: [0, 1800, None],
    })

    assert list(_pick_cover_subset(frame, "last")[Col.IMAGE_ID]) == [2]


def test_one_timestamp_across_every_candidate_keeps_the_given_order():
    """Counting needs no special case for ties: a stable sort leaves the
    caller's order alone, so the two ends still come from opposite ends."""
    frame = pd.DataFrame({
        Col.IMAGE_ID: [1, 2, 3, 4],
        Col.GENERAL_TIME: [900, 900, 900, 900],
    })

    assert list(_pick_cover_subset(frame, "first")[Col.IMAGE_ID]) == [1]
    assert list(_pick_cover_subset(frame, "last")[Col.IMAGE_ID]) == [4]


def test_falls_back_to_image_time_when_there_is_no_general_time():
    frame = axis_frame().drop(columns=[Col.GENERAL_TIME])

    assert list(_pick_cover_subset(frame, "first")[Col.IMAGE_ID]) == [3]


# -- which photo, inside the window ---------------------------------------


def test_a_lower_image_order_is_the_better_photo():
    """`image_order` is the content model's `selectionOrder`, a rank where 0 is
    best -- `update_photos_ranks` sets a hand-picked photo to 0, and the
    selection stage sorts it ascending. Both cover rules sorted it descending,
    so they preferred the worst-ranked candidate of every tie they broke.
    """
    subset = pd.DataFrame({
        Col.IMAGE_ID: [1, 2, 3],
        Col.IMAGE_ORDER: [90.0, 2.0, 40.0],
        Col.IMAGE_ORIENTATION: ['landscape'] * 3,
        Col.IMAGE_SUBQUERY_CONTENT: ['bride and groom smiling at each other'] * 3,
    })

    ranked = _select_by_priority_from_subset(
        subset,
        ['bride and groom smiling at each other'],
        ['bride and groom during the ceremony'],
    )

    assert ranked[0] == 2, f"best rank should lead, got {ranked}"


# -- gating and failure ----------------------------------------------------


def test_skipped_on_a_non_wedding_gallery():
    """The rule is built around the couple and reads `cluster_context`, which
    non-wedding galleries never get. ProcessStage keeps handling those."""
    context = run(couple_gallery(), is_wedding=False)

    assert context.key_pages is None
    assert Col.KEY_PAGE not in context.photos.columns
    assert context.diagnostics[-1].note == "skipped"


def test_skipped_when_the_album_has_no_first_page():
    context = run(couple_gallery(), pages={"lastPage": True})

    assert context.key_pages is None
    assert Col.KEY_PAGE not in context.photos.columns
    assert context.diagnostics[-1].note == "skipped"


def test_runs_when_there_is_no_design_data_at_all():
    """An offline run or a test has no `pagesInfo`; the covers are still worth
    computing there."""
    context = run(couple_gallery(), pages={})
    assert context.key_pages is not None


def test_a_gallery_it_cannot_serve_loses_the_covers_not_the_album():
    """Missing `cluster_context` on a wedding is a broken gallery, but a cover
    is not worth failing an album over -- the substage is optional."""
    photos = couple_gallery().drop(columns=[Col.CLUSTER_CONTEXT])

    context = run(photos)

    assert not context.failed
    assert not context.diagnostics[-1].ok


# -- the message boundary --------------------------------------------------


class FakeMessage:
    def __init__(self, content=None):
        self.content = content if content is not None else {"base_url": "x://y", "projectId": 1}
        self.pagesInfo = {}
        self.designsInfo = {}


def test_survives_a_round_trip_through_the_message():
    message = FakeMessage()
    context = AlbumContext.from_message(message)
    context.photos = couple_gallery()
    context.key_pages = KeyPages(opening=[7], closing=[9])
    context.sync_to_message()

    assert message.content["key_pages"] == {"opening": [7], "closing": [9]}

    rebuilt = AlbumContext.for_message(FakeMessage(dict(message.content)))

    assert rebuilt.key_pages.opening == [7]
    assert rebuilt.key_pages.closing == [9]


def test_absent_key_pages_stay_absent():
    """A message from before this substage existed must not grow an empty
    KeyPages that a consumer would read as 'no cover found'."""
    message = FakeMessage({"base_url": "x://y", "projectId": 1,
                           "gallery_photos_info": couple_gallery()})

    rebuilt = AlbumContext.for_message(message)

    assert rebuilt.key_pages is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# -- the covers must be two different photos ------------------------------
#
# Regression tests for 53507032, which opened *and* closed on one frame: a
# confetti-line shot with six faces, the groom looking away and the bride's
# face cut off at the edge.


def mixed_orientation_gallery(n=24, landscapes=1):
    """A gallery whose couple frames are portrait but for `landscapes` of them.

    This is the shape that broke the rule. Landscape was filtered *before* the
    couple test, so a 48-frame candidate base collapsed to the single landscape
    frame inside it -- and one frame cannot furnish two covers.
    """
    frame = couple_gallery(n=n, orientation="portrait")
    frame.loc[frame.index[:landscapes], Col.IMAGE_ORIENTATION] = "landscape"
    return frame


def test_one_landscape_among_portraits_still_gives_two_covers():
    context = run(mixed_orientation_gallery(landscapes=1))

    assert context.key_pages.opening, "a cover is still due"
    assert context.key_pages.closing
    assert context.key_pages.opening != context.key_pages.closing, (
        "the album opened and closed on the same photo")


def test_the_single_landscape_does_not_win_both_ends():
    """It is early in the day, so only the opening window holds it at all."""
    frame = mixed_orientation_gallery(landscapes=1)
    landscape = frame.loc[frame[Col.IMAGE_ORIENTATION] == "landscape", Col.IMAGE_ID].iloc[0]

    context = run(frame)

    assert context.key_pages.closing != [landscape]


def test_orientation_is_a_preference_not_a_filter():
    """With everything else equal the landscape wins its window; the portraits
    are still candidates, which is the difference that matters."""
    frame = couple_gallery(n=8, orientation="portrait")
    frame.loc[frame.index[1], Col.IMAGE_ORIENTATION] = "landscape"
    frame[Col.IMAGE_ORDER] = 5.0  # no rank signal
    frame[Col.IMAGE_SUBQUERY_CONTENT] = "bride and groom smiling at each other"

    context = run(frame)

    assert context.key_pages.opening == [frame[Col.IMAGE_ID].iloc[1]]


def test_covers_are_separated_across_the_day():
    context = run(couple_gallery(n=24))
    photos = context.photos

    def position(image_id):
        ordered = photos.sort_values(Col.GENERAL_TIME)[Col.IMAGE_ID].tolist()
        return ordered.index(image_id) / (len(ordered) - 1)

    apart = abs(position(context.key_pages.closing[0])
                - position(context.key_pages.opening[0]))
    assert apart >= 0.25, f"covers only {apart:.2f} of the day apart"


def test_a_crowd_loses_to_the_couple_alone():
    """A cover is of the two of them. The frame this replaced carried six faces."""
    frame = couple_gallery(n=8)
    frame[Col.IMAGE_ORDER] = 5.0
    frame[Col.IMAGE_SUBQUERY_CONTENT] = "bride and groom smiling at each other"
    frame.loc[frame.index[0], Col.N_FACES] = 8      # earliest, but a crowd
    frame.loc[frame.index[1], Col.N_FACES] = 2

    context = run(frame)

    assert context.key_pages.opening == [frame[Col.IMAGE_ID].iloc[1]]


def test_a_thinner_presence_loses_all_else_being_equal():
    """Presence is graded, not required. Both names and two faces still beats
    one clear face and a profile at the frame edge -- it just no longer
    excludes it before anything is scored."""
    frame = couple_gallery(n=8)
    frame[Col.IMAGE_ORDER] = 5.0
    frame[Col.IMAGE_SUBQUERY_CONTENT] = "bride and groom smiling at each other"
    frame.loc[frame.index[0], Col.N_FACES] = 1
    frame.at[frame.index[0], Col.PERSONS_IDS] = [BRIDE]

    context = run(frame)

    assert context.key_pages.opening != [frame[Col.IMAGE_ID].iloc[0]]


def test_a_faceless_embrace_is_a_candidate_at_all():
    """The point of relaxing the gate. `persons_ids` is built from face
    clusters, so an embrace with faces turned away carries no face *and* no
    identity, and the old base dropped it twice over -- 53 of 204 couple
    frames on 53227528. It must at least be reachable."""
    frame = couple_gallery(n=8)
    frame[Col.N_FACES] = 0
    frame[Col.PERSONS_IDS] = [[] for _ in range(len(frame))]

    context = run(frame)

    assert context.key_pages.opening, "a gallery of faceless couple frames still gets a cover"
    assert context.key_pages.opening != context.key_pages.closing


def test_the_old_gate_can_be_restored():
    frame = couple_gallery(n=8)
    frame[Col.N_FACES] = 0
    frame[Col.PERSONS_IDS] = [[] for _ in range(len(frame))]
    original = CONFIGS['covers']
    CONFIGS['covers'] = {**original, 'require_identities': True}
    try:
        context = run(frame)
    finally:
        CONFIGS['covers'] = original

    # Nothing names the couple, so the fallback hands back the couple frames
    # rather than nothing -- a worse cover beats no cover, as before.
    assert context.key_pages.opening


def test_a_gallery_of_single_face_frames_still_gets_covers():
    """Relaxed rather than enforced: a worse cover beats no cover."""
    frame = couple_gallery(n=8)
    frame[Col.N_FACES] = 1

    context = run(frame)

    assert context.key_pages.opening and context.key_pages.closing
    assert context.key_pages.opening != context.key_pages.closing


def test_the_rank_fallback_takes_the_best_photo_not_the_worst():
    """`image_order` is a rank where 0 is best. The fallback sorted it
    descending, so the one path that ran when no candidate was found handed
    back the worst-ranked photo in the gallery.
    """
    from src.core.key_pages import get_important_imgs

    # No couple frames at all, so both covers fall through to the fallback.
    frame = couple_gallery(n=6)
    frame[Col.CLUSTER_CONTEXT] = "other"
    frame[Col.IMAGE_ORDER] = [90.0, 2.0, 40.0, 70.0, 55.0, 80.0]

    first, last = get_important_imgs(frame, None, _QUIET_LOG)

    best = frame.loc[frame[Col.IMAGE_ORDER].idxmin(), Col.IMAGE_ID]
    assert first[0] == best, f"expected the best-ranked photo, got {first[0]}"
    assert last, "the closing cover must be filled too"
    assert first[0] != last[0], "the fallback handed both covers the same photo"
    worst = frame.loc[frame[Col.IMAGE_ORDER].idxmax(), Col.IMAGE_ID]
    assert first[0] != worst


def test_quality_decides_between_equal_candidates(monkeypatch):
    """The term the priority ladder had no way to express: two frames matching
    the same subquery, one of them a better photograph.
    """
    import src.core.key_pages as key_pages

    frame = couple_gallery(n=16)
    frame[Col.IMAGE_ORDER] = 5.0
    frame[Col.IMAGE_SUBQUERY_CONTENT] = "bride and groom smiling at each other"
    favoured = frame[Col.IMAGE_ID].iloc[2]

    def fake_scores(photos, concept):
        return np.where(photos[Col.IMAGE_ID].values == favoured, 0.9, 0.1)

    monkeypatch.setattr(key_pages, "_quality",
                        lambda f, log: fake_scores(f, None))

    context = run(frame)

    assert context.key_pages.opening == [favoured]


def test_covers_survive_concepts_being_unavailable():
    """No embeddings in this frame at all, so every concept projection raises.
    The covers are then decided by subquery and rank, as before the term
    existed -- a missing bin costs a score term, not the covers.
    """
    frame = couple_gallery(n=12).drop(columns=[Col.EMBEDDING])

    context = run(frame)

    assert context.key_pages.opening and context.key_pages.closing
    assert context.key_pages.opening != context.key_pages.closing


# -- affection: what makes a cover worth being one --------------------------


def _scored(frame, queries=None, bride=BRIDE, groom=GROOM):
    from src.core.key_pages import FIRST_COVER_QUERIES, _score_covers
    return _score_covers(frame, queries or FIRST_COVER_QUERIES, _QUIET_LOG, bride, groom)


def test_affection_takes_the_max_and_quality_the_mean():
    """A frame shows one kind of affection -- a kiss, or held hands, or a look
    -- so averaging across the bank punishes it for being emphatically one
    thing, which is the photo we are after. Quality is the opposite: a cover
    should be a good photograph on every count at once.

    Row 0 is emphatic on one concept and poor on the other (max 0.9, mean
    0.5); row 1 is evenly good (max 0.6, mean 0.6). Rows 2 and 3 fill out the
    candidates. Max ranks row 0 first, mean ranks row 1 first.
    """
    from src.core.key_pages import _concept_score
    import src.pipeline.enrich.timeline as tl

    def fake(frame, concept):
        return {'a': [0.9, 0.6, 0.0, 0.1], 'b': [0.1, 0.6, 0.0, 0.9]}[concept]

    original = tl.concept_scores
    tl.concept_scores = fake
    try:
        frame = couple_gallery(n=4)
        by_max = _concept_score(frame, ('a', 'b'), _QUIET_LOG, 'max', 'x')
        by_mean = _concept_score(frame, ('a', 'b'), _QUIET_LOG, 'mean', 'x')
    finally:
        tl.concept_scores = original

    assert by_max[0] > by_max[1], "max keeps the emphatic frame ahead"
    assert by_mean[1] > by_mean[0], "mean prefers the evenly good one"


def test_a_candid_beats_a_posed_portrait_on_subquery():
    """The commonest couple subquery by a wide margin is the posed portrait --
    147 of 211 on 53227528 -- and it used to rank second of four, so the cover
    was the album's most generic frame."""
    from src.core.key_pages import FIRST_COVER_QUERIES, _subquery_affinity

    frame = couple_gallery(n=2)
    frame.loc[frame.index[0], Col.IMAGE_SUBQUERY_CONTENT] = 'bride and groom kissing'
    frame.loc[frame.index[1], Col.IMAGE_SUBQUERY_CONTENT] = \
        'bride and groom posing for a portrait'

    affinity = _subquery_affinity(frame, FIRST_COVER_QUERIES)

    assert affinity[0] > affinity[1]


def test_presence_grades_rather_than_gates():
    from src.core.key_pages import _presence

    frame = couple_gallery(n=4)
    frame.at[frame.index[1], Col.PERSONS_IDS] = [BRIDE]
    frame.at[frame.index[2], Col.PERSONS_IDS] = []
    frame.at[frame.index[3], Col.PERSONS_IDS] = []
    frame.loc[frame.index[3], Col.N_FACES] = 0

    grades = _presence(frame, BRIDE, GROOM)

    assert grades[0] > grades[1] > grades[2] > grades[3]
    assert grades[3] > 0, "a faceless frame is handicapped, never excluded"


def _affection_frame(named_first=0.0, faceless=1.0):
    """Two identical frames -- one naming the couple, one with no face at all
    -- with the affection term stubbed so the trade-off is the only variable.
    """
    frame = couple_gallery(n=2)
    frame[Col.IMAGE_SUBQUERY_CONTENT] = 'bride and groom smiling at each other'
    frame[Col.IMAGE_ORDER] = 3.0
    frame.at[frame.index[1], Col.PERSONS_IDS] = []
    frame.loc[frame.index[1], Col.N_FACES] = 0
    # Faces turned away, but both of them are there: two bodies, not a detail.
    frame[Col.NUMBER_BODIES] = 2

    import src.pipeline.enrich.timeline as tl
    original = tl.concept_scores
    # Only the concepts affection does *not* share with quality may vary, or
    # the stub leaks into the quality term and carries the result there --
    # which is exactly what happened, and let the test pass with the affection
    # weight zeroed.
    exclusive = (set(CONFIGS['covers']['affection_concepts'])
                 - set(CONFIGS['covers']['quality_concepts']))
    assert exclusive, "affection needs at least one concept of its own"
    tl.concept_scores = lambda f, concept: (
        [named_first, faceless] if concept in exclusive else [0.5] * len(f))
    return frame, original, tl


def test_all_else_equal_the_frame_naming_both_wins():
    frame, original, tl = _affection_frame(named_first=0.5, faceless=0.5)
    try:
        scores = _scored(frame)
    finally:
        tl.concept_scores = original

    assert scores[0] > scores[1], "the handicap decides when affection does not"


def test_a_faceless_frame_wins_when_it_is_the_more_affectionate_photograph():
    """The whole point of relaxing the gate. An embrace with faces turned away
    carries no face and so no identity, and it must be able to beat a frame
    that merely names the couple -- by being the better photograph, against a
    handicap of 0.8 of the presence weight."""
    frame, original, tl = _affection_frame(named_first=0.0, faceless=1.0)
    try:
        scores = _scored(frame)
    finally:
        tl.concept_scores = original

    assert scores[1] > scores[0]


def test_the_couple_classes_are_all_reachable():
    """`kiss` and `couple` are couple moments by definition and were
    unreachable while the base was `bride and groom` alone."""
    for category in ('bride and groom', 'kiss', 'couple'):
        assert category in CONFIGS['covers']['cover_classes']


def test_a_kiss_frame_can_be_a_cover_now():
    from src.core.key_pages import _candidate_base

    frame = couple_gallery(n=4)
    frame.loc[frame.index[0], Col.CLUSTER_CONTEXT] = 'kiss'

    base = _candidate_base(frame, BRIDE, GROOM, _QUIET_LOG)

    assert frame[Col.IMAGE_ID].iloc[0] in set(base[Col.IMAGE_ID])


def test_the_affection_concepts_ship_for_both_model_versions():
    """No new bin, so no blob write -- the same constraint the quality term
    was built under."""
    from src.selection.auto_selection import load_pre_queries_embeddings
    from utils.configs import CONFIGS as C
    for concept in C['covers']['affection_concepts']:
        for version in (1, 2):
            assert len(load_pre_queries_embeddings(concept, version)) > 0, \
                f"{concept} missing for v{version}"


# -- a cover must not be repeated inside the album -------------------------


def _cover_and_body(similarity):
    """A body frame at a chosen cosine to the cover, and one far from it."""
    import numpy as _np
    cover = _np.array([1.0, 0.0], dtype=_np.float32)
    near = _np.array([similarity, (1 - similarity ** 2) ** 0.5], dtype=_np.float32)
    far = _np.array([0.0, 1.0], dtype=_np.float32)
    body = pd.DataFrame({
        Col.IMAGE_ID: [10, 11],
        'embedding': [near, far],
    })
    covers = pd.DataFrame({Col.IMAGE_ID: [1], 'embedding': [cover]})
    return body, covers


def test_a_body_photo_that_repeats_a_cover_is_dropped():
    """Taking the cover out of the body is not enough -- the next frame of the
    burst is a different `image_id`. On 53227528 the back cover and a body
    photo are the same forehead kiss one step wider, cosine 0.840."""
    from src.core.key_pages import _drop_cover_duplicates

    body, covers = _cover_and_body(0.84)

    kept = _drop_cover_duplicates(body, covers, _QUIET_LOG)

    assert list(kept[Col.IMAGE_ID]) == [11]


def test_a_merely_related_body_photo_is_kept():
    """The threshold has to leave the ordinary run of couple photos alone: the
    next-nearest on 53227528 is 0.719 and nothing on 49995684 exceeds 0.567."""
    from src.core.key_pages import _drop_cover_duplicates

    body, covers = _cover_and_body(0.72)

    kept = _drop_cover_duplicates(body, covers, _QUIET_LOG)

    assert list(kept[Col.IMAGE_ID]) == [10, 11]


def test_the_drop_is_capped():
    """A cover resembling half the gallery must not empty the body."""
    from src.core.key_pages import _drop_cover_duplicates
    import numpy as _np

    cover = _np.array([1.0, 0.0], dtype=_np.float32)
    body = pd.DataFrame({
        Col.IMAGE_ID: list(range(20, 30)),
        'embedding': [cover.copy() for _ in range(10)],
    })
    covers = pd.DataFrame({Col.IMAGE_ID: [1], 'embedding': [cover]})

    kept = _drop_cover_duplicates(body, covers, _QUIET_LOG)

    cap = CONFIGS['covers']['cover_duplicate_max_drop']
    assert len(kept) == 10 - cap


def test_no_embeddings_is_not_an_error():
    from src.core.key_pages import _drop_cover_duplicates

    body = pd.DataFrame({Col.IMAGE_ID: [10, 11]})
    covers = pd.DataFrame({Col.IMAGE_ID: [1]})

    assert len(_drop_cover_duplicates(body, covers, _QUIET_LOG)) == 2


def test_a_manual_selection_never_loses_a_photo_to_the_cover_check():
    """The user picked these. A duplicate of the cover among them is theirs.

    44573310 asked for 55 chosen photos; three were near-identical to the cover
    and were dropped without a word.
    """
    from src.core.key_pages import _drop_cover_duplicates

    body, covers = _cover_and_body(0.99)

    assert len(_drop_cover_duplicates(body, covers, _QUIET_LOG)) == 1
    kept = _drop_cover_duplicates(body, covers, _QUIET_LOG, manual_selection=True)

    assert len(kept) == 2
    assert set(kept[Col.IMAGE_ID]) == {10, 11}


def test_a_manual_selection_keeps_photos_identical_to_the_cover():
    """Not even the cap-filling case: no drop means none at all."""
    from src.core.key_pages import _drop_cover_duplicates
    import numpy as _np

    cover = _np.array([1.0, 0.0], dtype=_np.float32)
    body = pd.DataFrame({
        Col.IMAGE_ID: list(range(20, 30)),
        'embedding': [cover.copy() for _ in range(10)],
    })
    covers = pd.DataFrame({Col.IMAGE_ID: [1], 'embedding': [cover]})

    kept = _drop_cover_duplicates(body, covers, _QUIET_LOG, manual_selection=True)

    assert len(kept) == 10


def test_choose_good_wedding_images_passes_the_manual_flag_down():
    """The flag has to survive the call the layout stage actually makes."""
    from src.core import key_pages as kp

    seen = {}

    def _spy(df, cover_rows, logger, manual_selection=False):
        seen['manual_selection'] = manual_selection
        return df

    original = kp._drop_cover_duplicates
    kp._drop_cover_duplicates = _spy
    try:
        df = pd.DataFrame({
            Col.IMAGE_ID: [1, 2, 3, 4],
            'image_order': [1, 2, 3, 4],
            'persons_ids': [[1], [1], [1], [1]],
            'image_orientation': ['landscape'] * 4,
            'embedding': [None] * 4,
        })
        kp.choose_good_wedding_images(df, None, _QUIET_LOG, manual_selection=True)
    finally:
        kp._drop_cover_duplicates = original

    assert seen.get('manual_selection') is True


def test_the_cover_duplicate_check_can_be_switched_off():
    from src.core.key_pages import _drop_cover_duplicates

    body, covers = _cover_and_body(0.99)
    original = CONFIGS['covers']
    CONFIGS['covers'] = {**original, 'cover_duplicate_similarity': 0.0}
    try:
        kept = _drop_cover_duplicates(body, covers, _QUIET_LOG)
    finally:
        CONFIGS['covers'] = original

    assert len(kept) == 2


# -- a recognised face that is not theirs ------------------------------------
#
# 49996919 opened on the bride and her *father*: a cheek kiss scoring the top
# of the affection range, `persons_ids == [5]` -- a bride-side parent
# `enrich.parents` had already resolved at 0.95 -- and the bride's own face
# turned away and so unlisted. Three guards missed it (`refile_misfiled_couple`
# needs one of the couple present, the parents relabel was gated on `portrait`,
# and `_presence` scored "recognised nobody" and "recognised somebody else"
# alike) and it won the cover over frames naming them both.

#: A stand-in for the father: an identity that is neither of the couple.
OUTSIDER = 5


def test_a_recognised_outsider_scores_below_a_frame_that_recognised_nobody():
    """Absence of evidence and evidence of absence are not the same grade.

    A frame with no identity at all may still be the couple -- that is the case
    `_presence` became a grade to protect. A frame naming somebody else cannot
    be, and must not share its score.
    """
    from src.core.key_pages import _presence

    frame = couple_gallery(n=3)
    frame.at[frame.index[1], Col.PERSONS_IDS] = []          # faces, no identity
    frame.at[frame.index[2], Col.PERSONS_IDS] = [OUTSIDER]  # the father

    grades = _presence(frame, BRIDE, GROOM)

    assert grades[0] > grades[1] > grades[2]
    assert grades[2] > 0, "graded down, never excluded"


def test_a_recognised_outsider_is_a_real_handicap():
    """What the `others` grade is worth, stated as the thing it actually does.

    It is a handicap, not a veto: the same frame scores strictly lower for
    naming somebody who is neither of them than it would for naming them both.
    A veto is what `require_identities` was, and this module removed it on
    purpose.

    It is deliberately *not* asserted here that such a frame can never open an
    album -- with the affection gate off it still can, given enough affection,
    and on 49996919 it did. Scoring was the wrong place to fix that: the whole
    misread scene spans three presence grades, so demoting one only promotes
    another frame of the same scene. `enrich.couple_scenes` takes the scene out
    of the couple classes instead, and
    `test_a_one_sided_scene_is_refiled` is where that is guarded.
    """
    def score_with(persons_ids):
        frame = couple_gallery(n=2)
        frame[Col.IMAGE_SUBQUERY_CONTENT] = 'bride and groom smiling at each other'
        frame[Col.IMAGE_ORDER] = 3.0
        frame.at[frame.index[1], Col.PERSONS_IDS] = persons_ids

        import src.pipeline.enrich.timeline as tl
        original = tl.concept_scores
        exclusive = (set(CONFIGS['covers']['affection_concepts'])
                     - set(CONFIGS['covers']['quality_concepts']))
        tl.concept_scores = lambda f, concept: (
            [0.5, 0.5] if concept in exclusive else [0.5] * len(f))
        try:
            return _scored(frame)[1]
        finally:
            tl.concept_scores = original

    named_both = score_with([BRIDE, GROOM])
    outsider = score_with([OUTSIDER])
    nobody = score_with([])

    assert outsider < nobody < named_both,         "recognising the wrong person is worse evidence than recognising none"


def test_affection_is_undiluted_for_a_frame_naming_both():
    """The constraint the gate had to respect: affection keeps its full weight
    where the identities confirm the couple. Only frames whose presence is in
    doubt are discounted, so the gate is not a quiet cut to the weight."""
    from src.core.key_pages import _score_covers, FIRST_COVER_QUERIES

    frame = couple_gallery(n=2)
    frame[Col.IMAGE_SUBQUERY_CONTENT] = 'bride and groom smiling at each other'
    frame[Col.IMAGE_ORDER] = 3.0

    import src.pipeline.enrich.timeline as tl
    original = tl.concept_scores
    exclusive = (set(CONFIGS['covers']['affection_concepts'])
                 - set(CONFIGS['covers']['quality_concepts']))
    tl.concept_scores = lambda f, concept: (
        [0.0, 1.0] if concept in exclusive else [0.5] * len(f))
    try:
        scores = _score_covers(frame, FIRST_COVER_QUERIES, _QUIET_LOG, BRIDE, GROOM)
    finally:
        tl.concept_scores = original

    weight = CONFIGS['covers']['weights']['affection']
    assert scores[1] - scores[0] == pytest.approx(weight), \
        "both named: the full affection weight separates the two frames"


# -- a "single-box" cover layout that is not single-box -----------------------
#
# 53009168 failed its whole album on `single positional indexer is
# out-of-bounds`. The cover section is filled from the cover photos -- one, for
# a wedding -- but the loop walked every box of the chosen layout, and the
# chosen layout had more than one: the preferred branches of
# `_find_single_box_layout` tested only "max landscapes == 1", which a
# three-box design with one large landscape satisfies.

def _layout_row(**over):
    row = {
        "id": 1, "number of boxes": 1,
        "max landscapes": 0, "max portraits": 0,
        "left_large_landscape": 0, "right_large_landscape": 0,
        "left_large_portrait": 0, "right_large_portrait": 0,
        "left_large_square": 0, "right_large_square": 0,
    }
    row.update(over)
    return row


def test_a_multi_box_layout_is_never_offered_as_a_cover():
    from src.core.key_pages import _find_single_box_layout

    layouts = pd.DataFrame([
        # one large landscape, but three boxes: accepted before, fatal later
        _layout_row(id=10, **{"number of boxes": 3, "max landscapes": 1,
                              "left_large_landscape": 1}),
        # the honest single-box landscape
        _layout_row(id=11, **{"number of boxes": 1, "max landscapes": 1,
                              "right_large_landscape": 1}),
    ])

    chosen = _find_single_box_layout(layouts, "landscape")

    assert chosen, "a single-box landscape layout exists and must be found"
    assert all(layouts.loc[key]["number of boxes"] == 1 for key in chosen)
    assert 0 not in chosen, "the three-box layout must not be offered"


def test_a_multi_box_portrait_layout_is_never_offered_as_a_cover():
    from src.core.key_pages import _find_single_box_layout

    layouts = pd.DataFrame([
        _layout_row(id=20, **{"number of boxes": 2, "max portraits": 1,
                              "left_large_portrait": 1}),
        _layout_row(id=21, **{"number of boxes": 1, "max portraits": 1,
                              "right_large_portrait": 1}),
    ])

    chosen = _find_single_box_layout(layouts, "portrait")

    assert chosen
    assert all(layouts.loc[key]["number of boxes"] == 1 for key in chosen)


def test_the_cover_loop_leaves_extra_boxes_empty_rather_than_failing():
    """The second line: a mismatch must cost boxes, not the whole album."""
    from src.request_processing import _fillable_cover_boxes

    images_df = pd.DataFrame({Col.IMAGE_ID: [7]})          # one cover photo
    fillable = _fillable_cover_boxes([101, 102, 103], images_df, [7],
                                     'firstPage', _QUIET_LOG)

    assert fillable == [101]
    # and every box is filled when the counts agree
    two = pd.DataFrame({Col.IMAGE_ID: [7, 8]})
    assert _fillable_cover_boxes([101, 102], two, [7, 8], 'firstPage', _QUIET_LOG) == [101, 102]


# -- tears and details ---------------------------------------------------------
#
# 49994361 closed on the couple each wiping away tears at the dinner speeches,
# apart and not looking at each other: the `affection` bin reads a moved face as
# affection. 49995684 closed on a close-up of held hands, carried by the
# gallery's highest holding-hands score. Neither is a photo *of the two of them*.


def _with_tears(values):
    """Stub `concept_scores` for `tears` alone; every other bin scores zero."""
    import src.pipeline.enrich.timeline as tl

    original = tl.concept_scores

    def fake(frame, concept):
        return list(values) if concept == 'tears' else [0.0] * len(frame)

    tl.concept_scores = fake
    return original


def test_tears_are_penalised_on_the_raw_score_not_the_window():
    """Absolute: a window where nobody cries is not penalised at all."""
    import src.pipeline.enrich.timeline as tl
    from src.core.key_pages import _tears_penalty

    low, high = CONFIGS['covers']['tears_range']
    original = _with_tears([low - 0.05, (low + high) / 2, high + 0.05])
    try:
        penalty = _tears_penalty(couple_gallery(n=3), _QUIET_LOG)
    finally:
        tl.concept_scores = original

    assert penalty[0] == 0.0
    assert penalty[1] == pytest.approx(0.5)
    assert penalty[2] == 1.0


def test_a_crying_frame_loses_to_the_same_frame_dry():
    import src.pipeline.enrich.timeline as tl

    _, high = CONFIGS['covers']['tears_range']
    frame = couple_gallery(n=2)
    frame[Col.IMAGE_SUBQUERY_CONTENT] = 'bride and groom smiling at each other'
    frame[Col.IMAGE_ORDER] = 3.0
    original = _with_tears([high, 0.0])
    try:
        scores = _scored(frame)
    finally:
        tl.concept_scores = original

    assert scores[1] - scores[0] == pytest.approx(CONFIGS['covers']['weights']['tears'])


def test_a_missing_tears_bin_costs_nothing():
    import src.pipeline.enrich.timeline as tl
    from src.core.key_pages import _tears_penalty

    original = tl.concept_scores

    def missing(frame, concept):
        raise FileNotFoundError(concept)

    tl.concept_scores = missing
    try:
        penalty = _tears_penalty(couple_gallery(n=2), _QUIET_LOG)
    finally:
        tl.concept_scores = original

    assert list(penalty) == [0.0, 0.0]


def test_a_frame_with_nobody_in_it_is_a_detail():
    from src.core.key_pages import _detail_penalty

    frame = couple_gallery(n=3)
    frame[Col.NUMBER_BODIES] = [2, 1, 0]
    frame[Col.N_FACES] = [2, 0, 0]

    assert list(_detail_penalty(frame)) == [0.0, 0.0, 1.0],         "a body with its face turned away is still a person"


def test_held_hands_count_only_where_there_are_people():
    """In a close-up of hands, the hands are the subject, not a sign of it."""
    import src.pipeline.enrich.timeline as tl
    from src.core.key_pages import _affection

    frame = couple_gallery(n=3)
    frame[Col.NUMBER_BODIES] = [2, 2, 0]
    frame[Col.N_FACES] = [2, 2, 0]
    original = tl.concept_scores

    def fake(frame, concept):
        return {'holdinghands': [0.10, 0.20, 0.40]}.get(concept, [0.0, 0.0, 0.0])

    tl.concept_scores = fake
    try:
        affection = _affection(frame, _QUIET_LOG)
    finally:
        tl.concept_scores = original

    assert affection[1] > affection[0], "held hands still count between two people"
    assert affection[2] <= affection[0], "the detail gets nothing for them"


# -- presence: a faceless body beside the one who is named -------------------


def _bbox(x1, y1, x2, y2):
    import types
    return types.SimpleNamespace(bbox=types.SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2))


def _rated_face(x1, y1, x2, y2, blur=200.0):
    f = _bbox(x1, y1, x2, y2)
    f.blurLevel = blur
    return f


def _one_named(bodies, faces, faceless=()):
    frame = couple_gallery(n=1)
    frame.at[frame.index[0], Col.PERSONS_IDS] = [BRIDE]
    frame[Col.FACES_INFO] = [faces]
    frame[Col.BODIES_INFO] = [bodies]
    frame[Col.FACELESS_PERSONS_IDS] = [list(faceless)]
    return frame


BRIDE_FACE = _rated_face(0.28, 0.30, 0.35, 0.45)
BRIDE_BODY = _bbox(0.23, 0.27, 0.48, 0.85)
PARTNER_BODY = _bbox(0.52, 0.21, 0.88, 0.88)     # face turned away
GUEST_BODY = _bbox(0.90, 0.40, 0.98, 0.60)       # small, in the back


def _grade(frame):
    from src.core.key_pages import _presence
    return _presence(frame, BRIDE, GROOM)[0]


def test_a_faceless_body_beside_the_named_one_grades_above_one():
    grades = CONFIGS['covers']['presence_grades']

    got = _grade(_one_named([BRIDE_BODY, PARTNER_BODY], [BRIDE_FACE]))

    assert got == pytest.approx(grades['one_and_body'])
    assert grades['one'] < got < grades['both']


def test_without_such_a_body_it_is_just_one():
    grades = CONFIGS['covers']['presence_grades']

    assert _grade(_one_named([BRIDE_BODY], [BRIDE_FACE])) == pytest.approx(grades['one'])


def test_a_small_faceless_guest_is_not_the_partner():
    grades = CONFIGS['covers']['presence_grades']

    assert _grade(_one_named([BRIDE_BODY, GUEST_BODY], [BRIDE_FACE])) == pytest.approx(grades['one'])


def test_a_body_with_a_face_on_it_is_not_faceless():
    grades = CONFIGS['covers']['presence_grades']
    faces = [BRIDE_FACE, _rated_face(0.65, 0.25, 0.72, 0.40)]

    assert _grade(_one_named([BRIDE_BODY, PARTNER_BODY], faces)) == pytest.approx(grades['one'])


def test_the_named_one_recognised_by_body_is_not_also_the_partner():
    """The bride alone, face hidden, placed from her body: her own faceless body
    is accounted for by her identity, and nobody else is there."""
    grades = CONFIGS['covers']['presence_grades']

    got = _grade(_one_named([BRIDE_BODY], [], faceless=[BRIDE]))

    assert got == pytest.approx(grades['one'])


def test_no_body_data_changes_nothing():
    grades = CONFIGS['covers']['presence_grades']
    frame = couple_gallery(n=1)
    frame.at[frame.index[0], Col.PERSONS_IDS] = [BRIDE]

    assert _grade(frame) == pytest.approx(grades['one'])


def test_without_body_data_nothing_is_a_detail():
    """No body detections to consult: a faceless frame may be an embrace."""
    from src.core.key_pages import _detail_penalty

    frame = couple_gallery(n=2)
    frame[Col.N_FACES] = [0, 0]

    assert list(_detail_penalty(frame)) == [0.0, 0.0]


# -- significance: the couple as the subject, not small in a scene -----------
#
# 49995684's closing, once the tears bin had passed it, was the couple far off
# at their own table dabbing their eyes: their bodies 0.78 of the foreground
# blob, where the frames worth keeping on three galleries sat at 1.23-1.36.


def _significance_of(body_areas, diameter):
    """One frame whose bodies are squares of the given areas, and a blob."""
    from src.core.key_pages import _significance

    bodies = [_bbox(0.0, 0.0, a ** 0.5, a ** 0.5) for a in body_areas]
    frame = couple_gallery(n=1)
    frame[Col.BODIES_INFO] = [bodies]
    frame[Col.DIAMETER] = diameter
    return _significance(frame)[0]


def _diameter_for(fg_share):
    """The blob diameter whose square-mask area is `fg_share` of the frame."""
    return 2 * (fg_share / np.pi) ** 0.5


def test_a_couple_small_in_their_foreground_is_not_significant():
    low, _ = CONFIGS['covers']['significance_range']

    assert _significance_of([0.12, 0.12], _diameter_for(0.24 / (low - 0.1))) == 0.0


def test_significance_saturates_so_a_close_up_gains_nothing():
    _, high = CONFIGS['covers']['significance_range']

    subject = _significance_of([0.2, 0.2], _diameter_for(0.4 / high))
    close_up = _significance_of([0.45, 0.45], _diameter_for(0.4 / high))

    assert subject == pytest.approx(1.0) and close_up == pytest.approx(1.0)


def test_significance_rises_in_between():
    low, high = CONFIGS['covers']['significance_range']
    middle = (low + high) / 2

    assert _significance_of([0.1, 0.1], _diameter_for(0.2 / middle)) == pytest.approx(0.5)


def test_only_the_two_largest_bodies_are_the_couple():
    _, high = CONFIGS['covers']['significance_range']
    d = _diameter_for(0.4 / high)

    assert _significance_of([0.2, 0.2, 0.3], d) == _significance_of([0.3, 0.2], d)


def test_nothing_to_measure_is_not_evidence_against_a_frame():
    from src.core.key_pages import _significance

    frame = couple_gallery(n=2)
    frame[Col.BODIES_INFO] = [[], [_bbox(0.1, 0.1, 0.5, 0.9)]]
    frame[Col.DIAMETER] = [0.6, None]

    assert list(_significance(frame)) == [1.0, 1.0]


# -- a subject cut by the frame, reusing the key pages, closing classes ------


def _face(x1, y1, x2, y2, blur=1.0):
    from types import SimpleNamespace
    return SimpleNamespace(bbox=SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2), blurLevel=blur)


def test_a_face_cut_by_the_side_of_the_photo_is_penalised():
    """49994361 closed on the bride laughing with the groom half out of the
    picture, his face cut by the right border of the original."""
    from src.core.key_pages import _cut_subject_penalty
    frame = pd.DataFrame({'faces_info': [
        [_face(0.3, 0.3, 0.5, 0.5), _face(0.85, 0.3, 1.0, 0.5)],   # groom off the right
        [_face(0.3, 0.3, 0.5, 0.5), _face(0.6, 0.3, 0.8, 0.5)],    # both inside
        [_face(0.3, 0.0, 0.5, 0.2), _face(0.6, 0.0, 0.8, 0.2)],    # cropped at the forehead
        [_face(0.3, 0.3, 0.5, 0.5), _face(0.97, 0.3, 1.0, 0.33)],  # a small guest at the edge
        [],
    ]})

    assert list(_cut_subject_penalty(frame)) == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_the_cut_frame_loses_the_closing():
    gallery = couple_gallery(n=24)
    gallery['faces_info'] = [[_face(0.3, 0.3, 0.45, 0.5), _face(0.55, 0.3, 0.7, 0.5)]] * len(gallery)
    context = run(gallery.copy())
    winner = context.key_pages.closing[0]

    cut = gallery.copy()
    cut['faces_info'] = [
        [_face(0.3, 0.3, 0.45, 0.5), _face(0.85, 0.3, 1.0, 0.5)] if i == winner
        else [_face(0.3, 0.3, 0.45, 0.5), _face(0.55, 0.3, 0.7, 0.5)]
        for i in cut[Col.IMAGE_ID]]

    assert run(cut).key_pages.closing[0] != winner


def test_processing_keeps_the_key_pages_it_was_handed():
    """ProcessStage chose again over the few selected couple frames and closed
    on a different photo from the one chosen over the whole gallery."""
    from src.core.key_pages import choose_good_wedding_images
    gallery = couple_gallery(n=24)
    couple = gallery[gallery[Col.CLUSTER_CONTEXT] == 'bride and groom']
    preferred = KeyPages(opening=[1003], closing=[1020])

    _, first, _, last, _ = choose_good_wedding_images(
        gallery, couple, _QUIET_LOG, prune_duplicates=False, preferred=preferred)

    assert (first, last) == ([1003], [1020])


def test_a_key_page_missing_from_the_pool_is_chosen_again():
    from src.core.key_pages import choose_good_wedding_images
    gallery = couple_gallery(n=24)
    couple = gallery[gallery[Col.CLUSTER_CONTEXT] == 'bride and groom']
    preferred = KeyPages(opening=[1003], closing=[99999])

    _, first, _, last, _ = choose_good_wedding_images(
        gallery, couple, _QUIET_LOG, prune_duplicates=False, preferred=preferred)

    assert first == [1003]
    assert last and last[0] in set(gallery[Col.IMAGE_ID]) and last[0] != 1003


def test_the_closing_can_be_the_cake_or_a_toast_and_the_opening_cannot():
    gallery = couple_gallery(n=24)
    late = gallery[Col.GENERAL_TIME] >= gallery[Col.GENERAL_TIME].quantile(0.8)
    gallery.loc[late, Col.CLUSTER_CONTEXT] = 'cake cutting'
    gallery.loc[late, Col.IMAGE_SUBQUERY_CONTENT] = 'bride and groom cutting cake'
    early = gallery[Col.GENERAL_TIME] <= gallery[Col.GENERAL_TIME].quantile(0.2)
    gallery.loc[early, Col.CLUSTER_CONTEXT] = 'cake cutting'

    context = run(gallery)
    classes = gallery.set_index(Col.IMAGE_ID)[Col.CLUSTER_CONTEXT]

    assert classes[context.key_pages.closing[0]] == 'cake cutting'
    assert classes[context.key_pages.opening[0]] != 'cake cutting'


def test_a_speech_closes_only_when_it_is_a_toast():
    from src.core.key_pages import _candidate_base
    frame = pd.DataFrame({
        Col.CLUSTER_CONTEXT: ['speech', 'speech', 'bride and groom'],
        Col.IMAGE_SUBQUERY_CONTENT: ['person standing giving a speech',
                                     'guests and bride and groom making toasts for the speech',
                                     'bride and groom dancing'],
        Col.IMAGE_ID: [1, 2, 3],
    })
    covers = CONFIGS['covers']

    base = _candidate_base(frame, BRIDE, GROOM, _QUIET_LOG,
                           classes=covers['closing_cover_classes'],
                           subqueries=covers['closing_class_subqueries'])

    assert list(base[Col.IMAGE_ID]) == [2, 3]


def test_the_table_penalty_spares_standing_at_the_cake(monkeypatch):
    """Raw `seated_at_table` scores: seated frames read 0.45 and up, the couple
    standing at the cake table 0.28-0.35. Only the first is charged."""
    import src.pipeline.enrich.timeline as timeline
    from src.core.key_pages import _table_penalty
    frame = pd.DataFrame({Col.IMAGE_ID: [1, 2, 3, 4]})
    monkeypatch.setattr(timeline, 'concept_scores',
                        lambda f, concept: np.array([0.53, 0.42, 0.31, 0.01]))

    penalty = _table_penalty(frame, _QUIET_LOG)

    assert penalty[0] == 1.0, "seated at the head table"
    assert 0.0 < penalty[1] < 1.0, "a seated toast, part way"
    assert list(penalty[2:]) == [0.0, 0.0], "standing at the cake, and the field"


def test_the_table_penalty_is_inert_without_its_bin(monkeypatch):
    import src.pipeline.enrich.timeline as timeline
    from src.core.key_pages import _table_penalty

    def missing(frame, concept):
        raise FileNotFoundError(concept)
    monkeypatch.setattr(timeline, 'concept_scores', missing)

    assert list(_table_penalty(pd.DataFrame({Col.IMAGE_ID: [1, 2]}), _QUIET_LOG)) == [0.0, 0.0]
