"""What the keyword rules do to a whole album, not to one function.

`tests/test_album_keywords.py` pins the matchers and the substages one at a
time. These run a real selection over the synthetic wedding gallery the
equivalence tests use, and ask what came out the far end.

**They run the scored path**, which is the one production takes: a real request
carries `aiMetadata.subjects`, so `tags_features` is not empty and
`SelectionInputs.unscored` is False. The gallery's embeddings are 32-dimensional
and the real query bank is not, so a stand-in bank of the matching width is
installed for the duration -- without it the tag score collapses, the album
falls to six photos, and the run measures nothing. `select_unscored` keeps the
other path reachable, because the rules behave differently there and that is
worth saying out loud rather than discovering later.

    python -m pytest tests/test_album_keywords_acceptance.py -v
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.core.keywords as keywords  # noqa: E402
import src.pipeline.select.route as route  # noqa: E402
from src.pipeline import AlbumContext, build_select  # noqa: E402
from src.pipeline.contracts import AiHints, GalleryFacts  # noqa: E402
from src.pipeline.registry import get  # noqa: E402
from tests.test_selection_equivalence import CATEGORIES, make_gallery  # noqa: E402

LOGGER = logging.getLogger("album-keywords-acceptance")
LOGGER.addHandler(logging.NullHandler())
LOGGER.setLevel(logging.CRITICAL)

#: The gallery's embedding width. The stand-in bank has to match it.
EMBEDDING_DIM = 32

SUBJECTS = [category for category, _ in CATEGORIES]


@contextmanager
def a_query_bank(seed: int = 3):
    """A tag bank of the gallery's own width, so the scored path can run."""
    rng = np.random.default_rng(seed)

    def bins(tags, version, logger):
        bank = {}
        for tag in tags:
            vectors = rng.normal(size=(3, EMBEDDING_DIM))
            bank[tag] = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        return bank

    original = route.get_tags_bins
    route.get_tags_bins = bins
    try:
        yield
    finally:
        route.get_tags_bins = original


def marked(keywords_by_id=None, ratings_by_id=None, columns=True, seed=7):
    """The synthetic gallery, with the photographer's metadata written onto it."""
    gallery = make_gallery(seed)
    if not columns:
        return gallery

    ids = gallery['image_id']
    gallery['keywords'] = [list((keywords_by_id or {}).get(i, [])) for i in ids]
    gallery['metadata_rating'] = [float((ratings_by_id or {}).get(i, 0.0)) for i in ids]
    return gallery


def _run(gallery, subjects, density=3):
    context = AlbumContext(
        logger=LOGGER,
        request={'rating': []},
        photos=gallery.copy(),
        hints=AiHints(photo_ids=[], person_ids=[], focus=['brideAndGroom'],
                      subjects=subjects, density=density, present=True),
        facts=GalleryFacts(is_wedding=True, model_version=2),
    )
    context = get('enrich.album_keywords')()(context)
    context = build_select(logger=LOGGER).run(context)
    assert not context.failed, context.error
    return context


def select(gallery, density=3):
    """Exclusion, then a full selection down the path production takes."""
    with a_query_bank():
        context = _run(gallery, SUBJECTS, density)
    assert context.selection_inputs.unscored is False, "this is the scored path"
    return context


def select_unscored(gallery, density=3):
    """The same, for a request that names no people, no picks and no subjects."""
    context = _run(gallery, [''], density)
    assert context.selection_inputs.unscored is True
    return context


def how_many_of(ids, context):
    selected = set(context.selection.photo_ids)
    return sum(1 for i in ids if i in selected)


def every_fifteenth(gallery, count=30):
    """A priority set spread across the gallery rather than piled into one
    category, which would be capped by that category's budget alone."""
    return [int(i) for i in gallery['image_id'].iloc[::15][:count]]


# --------------------------------------------------- 1. the exclusion holds

def test_a_five_star_photo_tagged_not_for_album_never_appears():
    """The two marks pull hardest in opposite directions: five stars is the
    strongest lift there is, notforalbum is absolute, and absolute wins."""
    gallery = make_gallery(7)
    victim = int(gallery['image_id'].iloc[40])

    context = select(marked(
        keywords_by_id={victim: ['Not For Album']},
        ratings_by_id={victim: 5},
    ))

    assert victim not in set(context.selection.photo_ids), "not selected"
    assert victim not in set(context.photos['image_id']), "not on the photo table"
    assert victim not in set(context.all_photos['image_id']), \
        "not even in the pool the covers are chosen from"


@pytest.mark.parametrize('spelling', ['notforalbum', 'Not For Album',
                                      'not-for-album', 'NOT_FOR_ALBUM'])
def test_the_exclusion_holds_whatever_the_photographer_typed(spelling):
    """And it still holds when the same photo also asks to be in the album."""
    gallery = make_gallery(7)
    victim = int(gallery['image_id'].iloc[12])
    context = select(marked(keywords_by_id={victim: [spelling, 'album']},
                            ratings_by_id={victim: 5}))
    assert victim not in set(context.selection.photo_ids)


def test_the_exclusion_holds_on_the_unscored_path_too():
    """Unlike the priority lift below, which does nothing there."""
    gallery = make_gallery(7)
    victim = int(gallery['image_id'].iloc[40])
    context = select_unscored(marked(keywords_by_id={victim: ['notforalbum']}))
    assert victim not in set(context.selection.photo_ids)


# ------------------------------------------------ 2. the priority set is used

def test_a_thirty_photo_priority_set_is_preferred_but_not_guaranteed():
    """Thirty photographs keyworded album, spread across the gallery.

    **Seventeen of them reach the album, not thirty, and that is the chosen
    rule working rather than failing.** A lift raises a photograph within its
    category's ranking; it does not reach past the category strategy, which has
    the last word and spends its slots on variety. Three marked photographs of
    one moment are still three photographs of one moment, and `bride and groom`
    fills all nine of its slots without taking any of them.

    If every marked photograph has to appear, that is `select.preselect`, which
    commits photographs before any strategy runs -- not a larger bonus. See
    the test below.
    """
    gallery = make_gallery(7)
    priority = every_fifteenth(gallery)
    assert len(priority) == 30

    unmarked = select(marked())
    with_marks = select(marked(keywords_by_id={i: ['album'] for i in priority}))

    before = how_many_of(priority, unmarked)
    after = how_many_of(priority, with_marks)

    assert after > before, "the keyword has to be worth something"
    assert after >= 2 * before, "and worth roughly double, on this gallery"
    assert len(with_marks.selection.photo_ids) == len(unmarked.selection.photo_ids), \
        "a priority set does not grow the album"


def test_a_bigger_bonus_buys_nothing_once_the_photo_tops_its_category():
    """The measurement behind the test above, kept because it is the thing
    nobody expects: the strategies are the ceiling, not the score.

    Measured 2026-10-22: 7/30 with the rule off, then 17/30 at 0.25, at 0.60,
    at 1.00 and at 5.00 alike. Anyone reading a disappointing album should
    reach for `select.preselect` rather than for `FOR_ALBUM_BONUS`.
    """
    gallery = make_gallery(7)
    priority = every_fifteenth(gallery)
    marks = {i: ['album'] for i in priority}

    original = keywords.FOR_ALBUM_BONUS
    selected = {}
    try:
        for bonus in (0.25, 1.0, 5.0):
            keywords.FOR_ALBUM_BONUS = bonus
            selected[bonus] = how_many_of(
                priority, select(marked(keywords_by_id=marks)))
    finally:
        keywords.FOR_ALBUM_BONUS = original

    assert len(set(selected.values())) == 1, (
        f"a twentyfold bonus changed the album: {selected}")


def test_the_lift_does_nothing_on_the_unscored_path():
    """A known gap, pinned so it is a decision rather than a surprise.

    When a request names no people, no picks and no subjects, every ranking in
    the loop goes through `narrowing.order_index`, which reads `image_order`
    and never sees `total_score`. The lift is computed and then ignored.
    Closing it means teaching that path about the marks, which is a change to
    how unhinted requests are ordered and belongs in its own commit.
    """
    gallery = make_gallery(7)
    priority = every_fifteenth(gallery)

    unmarked = select_unscored(marked())
    with_marks = select_unscored(
        marked(keywords_by_id={i: ['album'] for i in priority}))

    assert (how_many_of(priority, with_marks)
            == how_many_of(priority, unmarked)), "still the gap, see docstring"


def test_a_priority_set_larger_than_its_category_is_capped_by_the_budget():
    """The other side of the same choice, recorded so it is not a surprise:
    every photograph in one category keyworded album cannot all appear, because
    the lift reorders a category and never enlarges it."""
    gallery = make_gallery(7)
    dancing = gallery.loc[gallery['cluster_context'] == 'dancing', 'image_id']
    priority = [int(i) for i in dancing]

    context = select(marked(keywords_by_id={i: ['album'] for i in priority}))
    used = how_many_of(priority, context)

    assert 0 < used < len(priority), "the budget still governs"


# ------------------------------- 3. a gallery with no metadata is as it was

def test_a_gallery_with_no_metadata_produces_the_same_result_as_today():
    """Three runs that must agree photo for photo:

    * the gallery as it arrives from a project that was never vectorised --
      no keyword column and no rating column at all;
    * the same gallery with both columns present and empty, which is what a
      vectorised project whose photographer marked nothing looks like;
    * the same gallery with both bonuses set to 0, which is the scorer exactly
      as it was before any of this was written.
    """
    without_columns = select(marked(columns=False))
    with_empty_columns = select(marked())

    original = (keywords.FOR_ALBUM_BONUS, keywords.METADATA_RATING_BONUS)
    keywords.FOR_ALBUM_BONUS = 0
    keywords.METADATA_RATING_BONUS = 0
    try:
        rule_switched_off = select(marked())
    finally:
        keywords.FOR_ALBUM_BONUS, keywords.METADATA_RATING_BONUS = original

    assert without_columns.selection.photo_ids == with_empty_columns.selection.photo_ids
    assert without_columns.selection.photo_ids == rule_switched_off.selection.photo_ids
    assert without_columns.selection.spreads == with_empty_columns.selection.spreads
    assert (without_columns.selection.min_total_spreads
            == with_empty_columns.selection.min_total_spreads)


def test_an_unmarked_photo_in_a_marked_gallery_scores_what_it_always_did():
    """The bisect property, measured on the scorer rather than argued: in a
    gallery where two photographs are marked, those two move and nothing else
    does.

    Compared photo by photo, not position by position -- `get_scores` hands
    back its pairs sorted by score, so two runs line up different photographs.
    """
    from src.selection.ai_wedding_selection import get_scores

    plain = marked()
    lifted = [int(plain['image_id'].iloc[5]), int(plain['image_id'].iloc[9])]
    gallery = marked(keywords_by_id={lifted[0]: ['album']},
                     ratings_by_id={lifted[1]: 5})

    before, _ = get_scores(plain.copy(), plain.iloc[:0], [], {}, LOGGER, {})
    after, _ = get_scores(gallery.copy(), gallery.iloc[:0], [], {}, LOGGER, {})

    before, after = dict(before), dict(after)
    moved = {image_id for image_id, score in before.items()
             if after[image_id] != score}
    assert moved == set(lifted)


# ----------------------------------------------- 4. the exact match is exact

def test_the_keyword_albums_does_not_match():
    """album is matched whole. A photographer who filed a shot under albums has
    described it, not asked for anything."""
    assert not keywords.is_for_album(['albums'])

    gallery = make_gallery(7)
    plural = int(gallery['image_id'].iloc[33])
    frame = marked(keywords_by_id={plural: ['albums']})

    assert keywords.priority_lift(frame).sum() == 0.0, "nothing was lifted"


def test_the_plural_changes_no_selection_at_all():
    gallery = make_gallery(7)
    plural = int(gallery['image_id'].iloc[33])

    neutral = select(marked())
    with_plural = select(marked(keywords_by_id={plural: ['albums']}))

    assert neutral.selection.photo_ids == with_plural.selection.photo_ids


@pytest.mark.parametrize('near_miss', ['albums', 'wedding album', 'album cover',
                                       'the album', 'al bum', 'notforalbum'])
def test_only_the_whole_word_asks_for_anything(near_miss):
    assert not keywords.is_for_album([near_miss])
