"""The two keywords a photographer addresses to the album.

`notforalbum` keeps a photograph out and is matched loosely, because it is
typed by hand: case folded, spaces, hyphens and underscores dropped. `album`
puts one forward and is matched exactly, so the word inside a longer keyword
does not count. These tests pin both matchers, and pin that the exclusion
happens early enough that nothing downstream can put the photograph back.

    python -m pytest tests/test_album_keywords.py -v
"""
import logging

import numpy as np
import pandas as pd
import pytest

from src.core.keywords import fold, is_for_album, is_not_for_album
from src.pipeline.contracts import AlbumContext, Col
from src.pipeline.registry import INGEST, ENRICH, get

LOGGER = logging.getLogger('test_album_keywords')


# ------------------------------------------------------------- the loose match

@pytest.mark.parametrize('spelling', [
    'notforalbum', 'NOTFORALBUM', 'Not For Album', 'not-for-album',
    'NOT_FOR_ALBUM', 'not for album', 'Not-For_Album', '  not for album  ',
    'Not  For   Album',
])
def test_every_spelling_of_the_exclusion_is_one_keyword(spelling):
    assert is_not_for_album([spelling])


@pytest.mark.parametrize('other', [
    'album', 'not for print', 'album not', 'notforalbums', 'forbidden album',
    'notforalbumcover', '', 'wedding',
])
def test_a_keyword_that_is_not_the_exclusion_does_not_exclude(other):
    assert not is_not_for_album([other])


def test_the_exclusion_is_found_among_other_keywords():
    assert is_not_for_album(['ישראל', 'Keyword 1', 'Not For Album', 'beach'])


def test_anything_that_is_not_a_list_of_strings_excludes_nothing():
    for value in ([], None, 'not for album', 42, [None, 7, {'a': 1}]):
        assert not is_not_for_album(value)


def test_folding_is_what_the_matcher_is_built_on():
    assert fold('Not-For_Album') == 'notforalbum'
    assert fold(None) == ''


# ------------------------------------------------------------- the exact match

@pytest.mark.parametrize('spelling', ['album', 'Album', 'ALBUM', '  Album  '])
def test_the_priority_keyword_is_case_insensitive(spelling):
    assert is_for_album([spelling])


@pytest.mark.parametrize('other', [
    'wedding album', 'albums', 'notforalbum', 'not for album', 'al bum',
    'album cover', 'the album',
])
def test_the_priority_keyword_is_the_whole_keyword_and_nothing_more(other):
    """Loose matching is reserved for the exclusion. A photographer who types
    `wedding album` has described the picture, not asked for anything."""
    assert not is_for_album([other])


def test_a_photo_can_carry_both_and_the_predicates_just_report_it():
    """Resolving the contest is the pipeline's job, not the matcher's."""
    both = ['Album', 'not-for-album']
    assert is_for_album(both) and is_not_for_album(both)


# ------------------------------------------------------------- the substage

def frame(keywords):
    return pd.DataFrame({
        'image_id': np.arange(len(keywords), dtype=np.int64),
        'keywords': keywords,
    })


def excluded_from(photos):
    context = AlbumContext(photos=photos, logger=LOGGER)
    return get('enrich.album_keywords')()(context)


def test_the_marked_photos_leave_the_table():
    context = excluded_from(frame([
        ['beach'], ['Not For Album'], [], ['album'], ['NOT_FOR_ALBUM', 'album'],
    ]))
    assert context.photos['image_id'].tolist() == [0, 2, 3]


def test_the_exclusion_beats_the_priority_keyword_on_the_same_photo():
    """`album` does not save a photograph marked `notforalbum` -- the exclusion
    is the top-priority decision, and it is taken first."""
    context = excluded_from(frame([['album', 'not-for-album'], ['album']]))
    assert context.photos['image_id'].tolist() == [1]


def test_a_gallery_nobody_marked_is_untouched():
    photos = frame([['beach'], ['album'], []])
    context = excluded_from(photos)
    assert len(context.photos) == 3


def test_a_gallery_with_no_keyword_column_is_skipped_silently():
    """Not vectorised, so no payload was read and no instruction was given."""
    photos = pd.DataFrame({'image_id': np.array([1, 2], dtype=np.int64)})
    context = excluded_from(photos)
    assert len(context.photos) == 2
    assert not context.failed


def test_a_gallery_that_is_entirely_marked_empties_rather_than_failing():
    """Refusing it is `enrich.require_cluster_data`'s job, further down, where
    the floor is sized against the design."""
    context = excluded_from(frame([['not for album'], ['NotForAlbum']]))
    assert context.photos.empty
    assert not context.failed


# ------------------------------------------------------------- the placement

def test_the_keywords_are_on_the_table_before_the_exclusion_reads_them():
    order = list(INGEST) + list(ENRICH)
    assert order.index('ingest.merge_photo_metadata') < order.index('enrich.album_keywords')


@pytest.mark.parametrize('later', [
    # Nothing may pair an excluded frame as a treatment twin and swap it back
    # onto a page, count it towards the gallery's type or its budget, or offer
    # it as a cover.
    'enrich.treatment_twins', 'enrich.duplicate_shots', 'enrich.gallery_type',
    'enrich.key_pages', 'enrich.variants',
])
def test_the_exclusion_runs_before_everything_that_could_reinstate_it(later):
    order = list(INGEST) + list(ENRICH)
    assert order.index('enrich.album_keywords') < order.index(later)


# --------------------------------------------------------------- the lift

def scored_frame(keywords=None, **columns):
    data = {'image_id': np.arange(len(next(iter(columns.values()))), dtype=np.int64)}
    data.update(columns)
    if keywords is not None:
        data['keywords'] = keywords
    return pd.DataFrame(data)


def test_a_marked_photo_is_lifted_and_an_unmarked_one_is_not():
    from src.core.keywords import FOR_ALBUM_BONUS, for_album_lift

    frame = scored_frame(keywords=[['album'], ['beach'], [], ['Album', 'sea']],
                         image_order=[0, 1, 2, 3])
    assert for_album_lift(frame).tolist() == [
        FOR_ALBUM_BONUS, 0.0, 0.0, FOR_ALBUM_BONUS]


def test_a_gallery_with_no_keyword_column_lifts_nothing():
    """Not vectorised, so no instruction was given and every score is what it
    was before this rule existed."""
    from src.core.keywords import for_album_lift

    frame = scored_frame(image_order=[0, 1, 2])
    assert for_album_lift(frame).tolist() == [0.0, 0.0, 0.0]


def test_the_lift_can_be_turned_off_without_removing_the_rule():
    from src.core.keywords import for_album_lift

    frame = scored_frame(keywords=[['album'], ['beach']], image_order=[0, 1])
    assert for_album_lift(frame, bonus=0).tolist() == [0.0, 0.0]


def test_the_lift_is_a_priority_and_not_a_guarantee():
    """A marked photo beats an unmarked one within the bonus of it, and loses
    to one that is better than that by more. Which is the whole difference
    between this and committing it in `select.preselect`."""
    from src.core.keywords import FOR_ALBUM_BONUS, for_album_lift

    frame = scored_frame(keywords=[['album'], ['beach'], ['sea']],
                         image_order=[0, 1, 2])
    blended = pd.Series([0.40, 0.55, 0.90])
    total = blended + for_album_lift(frame)

    assert total[0] > total[1], "beats the one it was 0.15 behind"
    assert total[0] < total[2], "still loses to the one it was 0.50 behind"
    assert FOR_ALBUM_BONUS == 0.25


def test_an_unmarked_photo_scores_exactly_what_it_scored_before():
    """The bisect property: the lift is added on top of the blend, never
    weighted into it, so a gallery nobody keyworded is bit-identical."""
    from src.core.keywords import for_album_lift

    frame = scored_frame(keywords=[['beach'], ['sea'], []], image_order=[0, 1, 2])
    blended = pd.Series([0.4123456789, 0.55, 0.9])
    assert (blended + for_album_lift(frame)).tolist() == blended.tolist()


def test_the_unscored_path_puts_marked_photos_first():
    """No person ids, no picks, no subjects -- `image_order` decides, and the
    keyword is an instruction rather than a signal to weigh, so it still
    sorts to the front without disturbing the order within each half."""
    from src.pipeline.select.scoring import CandidateGate, Scorer

    frame = scored_frame(
        keywords=[['beach'], ['album'], ['sea'], ['Album']],
        image_order=[0, 1, 2, 3],
        cluster_context=['dancing'] * 4,
    )
    gate = CandidateGate(Scorer(None, [], {}, {}, LOGGER), unscored=True,
                         allocation={'dancing': 4}, logger=LOGGER,
                         prefer_subject=False)
    _scored, candidates, used_image_order = gate.candidates(frame, 'dancing')

    assert candidates == [1, 3, 0, 2]
    assert used_image_order is True

