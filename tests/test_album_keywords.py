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


# ------------------------------------------------------- rating, and its place

def rated_frame(ratings, keywords=None):
    data = {
        'image_id': np.arange(len(ratings), dtype=np.int64),
        'metadata_rating': ratings,
        'image_order': list(range(len(ratings))),
    }
    data['keywords'] = keywords if keywords is not None else [[]] * len(ratings)
    return pd.DataFrame(data)


def test_a_higher_rating_lifts_further():
    from src.core.keywords import metadata_rating_lift

    lift = metadata_rating_lift(rated_frame([5, 4, 1, 0]))
    assert lift[0] > lift[1] > lift[2] > lift[3] == 0.0


def test_the_scale_is_taken_from_the_gallery_rather_than_assumed():
    """Nothing in the payload says the rating is out of five. A gallery rated
    out of 100 and one rated out of 5 must order the same way."""
    from src.core.keywords import metadata_rating_lift

    out_of_five = metadata_rating_lift(rated_frame([5, 4, 1, 0])).tolist()
    out_of_hundred = metadata_rating_lift(rated_frame([100, 80, 20, 0])).tolist()
    assert out_of_five == out_of_hundred


def test_the_album_keyword_beats_the_best_rating():
    """The ordering asked for: `album` > 5 > 4 > unrated, between photographs
    the blend scores alike."""
    from src.core.keywords import FOR_ALBUM_BONUS, METADATA_RATING_BONUS, priority_lift

    frame = rated_frame([0, 4, 5, 0], keywords=[[], [], [], ['album']])
    lift = priority_lift(frame)

    assert lift[3] > lift[2] > lift[1] > lift[0] == 0.0
    assert METADATA_RATING_BONUS < FOR_ALBUM_BONUS, "the invariant the ordering rests on"


def test_a_photo_carrying_both_is_lifted_by_both():
    from src.core.keywords import FOR_ALBUM_BONUS, METADATA_RATING_BONUS, priority_lift

    frame = rated_frame([5, 5, 0], keywords=[['album'], [], ['album']])
    lift = priority_lift(frame)

    assert lift[0] == pytest.approx(FOR_ALBUM_BONUS + METADATA_RATING_BONUS)
    assert lift[1] == pytest.approx(METADATA_RATING_BONUS)
    assert lift[2] == pytest.approx(FOR_ALBUM_BONUS)
    assert lift[0] > lift[2] > lift[1], "album+5 > album > 5"


def test_a_zero_or_negative_rating_is_unrated():
    from src.core.keywords import metadata_rating_lift

    assert metadata_rating_lift(rated_frame([0, -3, 5])).tolist()[:2] == [0.0, 0.0]


def test_a_gallery_nobody_rated_lifts_nothing():
    from src.core.keywords import metadata_rating_lift

    assert metadata_rating_lift(rated_frame([0, 0, 0])).tolist() == [0.0, 0.0, 0.0]


def test_a_gallery_with_no_rating_column_lifts_nothing():
    from src.core.keywords import priority_lift

    frame = pd.DataFrame({'image_id': [1, 2], 'keywords': [[], []]})
    assert priority_lift(frame).tolist() == [0.0, 0.0]


# ------------------------------------------------------------------ the report

class Recorder:
    def __init__(self):
        self.lines = []

    def __getattr__(self, _name):
        return self._record

    def _record(self, message, *a, **k):
        self.lines.append(str(message))


def test_the_counts_are_taken_over_the_whole_gallery_before_it_narrows():
    photos = rated_frame(
        [5, 0, 4, 0, 0],
        keywords=[['album'], ['not for album'], [], ['Album'], ['beach']])
    context = AlbumContext(photos=photos, logger=LOGGER)
    context = get('enrich.album_keywords')()(context)

    summary = context.keyword_summary
    assert summary.read is True
    assert summary.excluded == 1
    assert summary.prioritised == 2, "both `album` photos survived the exclusion"
    assert summary.rated == 2, "the 5 and the 4; the excluded photo is not counted"


def test_a_gallery_that_was_never_vectorised_is_marked_unread():
    """So the zeros in the report cannot be read as 'nobody marked anything'."""
    context = AlbumContext(photos=pd.DataFrame({'image_id': [1, 2]}), logger=LOGGER)
    context = get('enrich.album_keywords')()(context)
    assert context.keyword_summary.read is False


def test_the_report_says_the_three_numbers_that_were_asked_for():
    from src.pipeline.contracts import KeywordSummary

    logger = Recorder()
    kept = rated_frame([5, 0], keywords=[['album'], ['beach']])
    context = AlbumContext(
        photos=kept, logger=logger,
        keyword_summary=KeywordSummary(excluded=4, prioritised=3, rated=7, read=True))
    get('select.keyword_report')()(context)

    line = "\n".join(logger.lines)
    assert "excluded 4" in line
    assert "priority set 3 keyworded album" in line
    assert "used 1/3 keyworded" in line


def test_nothing_is_logged_for_a_gallery_whose_keywords_were_never_read():
    logger = Recorder()
    context = AlbumContext(photos=rated_frame([0]), logger=logger)
    get('select.keyword_report')()(context)
    assert logger.lines == []


def test_the_report_runs_last_so_that_used_is_known():
    from src.pipeline.registry import SELECT

    assert list(SELECT)[-1] == 'select.keyword_report'
    assert list(SELECT).index('select.publish') < list(SELECT).index('select.keyword_report')


# ------------------------------------------------------ the Highlights reel

class Scene:
    """As `ptinfra.utils.gallery.Scene` presents itself to the reader."""

    def __init__(self, is_highlights, photo_ids):
        self.isHighlights = is_highlights
        self.photos = [type('P', (), {'photoId': i})() for i in photo_ids]


class FakeGallery:
    def __init__(self, scenes):
        self.scenes = scenes


def a_gallery(scenes):
    import src.readers.gallery as reader
    return lambda _url: FakeGallery(scenes)


def test_only_the_reel_scenes_contribute_ids(monkeypatch):
    import src.readers.gallery as reader

    monkeypatch.setattr(reader, 'Gallery', a_gallery([
        Scene(False, [1, 2, 3]),
        Scene(True, [2, 9]),
        Scene(False, [4]),
        Scene(True, [9, 10]),
    ]))
    assert reader.highlight_photo_ids('url', LOGGER) == {2, 9, 10}


def test_a_gallery_with_no_reel_marks_nothing(monkeypatch):
    import src.readers.gallery as reader

    monkeypatch.setattr(reader, 'Gallery', a_gallery([Scene(False, [1, 2])]))
    assert reader.highlight_photo_ids('url', LOGGER) == set()


def test_a_scene_list_that_cannot_be_read_marks_nothing(monkeypatch):
    """Blob storage is unreachable. That is a gallery whose photographs are
    unmarked, not a gallery with no album."""
    import src.readers.gallery as reader

    def explode(_url):
        raise OSError('blob storage is unreachable')

    monkeypatch.setattr(reader, 'Gallery', explode)
    assert reader.highlight_photo_ids('url', LOGGER) == set()


def test_the_substage_writes_the_flag(monkeypatch):
    import src.readers.gallery as reader

    monkeypatch.setattr(reader, 'Gallery', a_gallery([Scene(True, [1, 3])]))
    context = AlbumContext(
        photos=pd.DataFrame({'image_id': np.array([1, 2, 3], dtype=np.int64)}),
        logger=LOGGER, project_url='url')
    context = get('ingest.highlights')()(context)

    assert context.photos[Col.IN_HIGHLIGHTS].tolist() == [True, False, True]


def test_the_substage_writes_the_flag_even_when_the_read_fails(monkeypatch):
    """So no later rule has to ask whether this gallery had a reel."""
    import src.readers.gallery as reader

    def explode(_url):
        raise OSError('nope')

    monkeypatch.setattr(reader, 'Gallery', explode)
    context = AlbumContext(
        photos=pd.DataFrame({'image_id': np.array([1, 2], dtype=np.int64)}),
        logger=LOGGER, project_url='url')
    context = get('ingest.highlights')()(context)

    assert context.photos[Col.IN_HIGHLIGHTS].tolist() == [False, False]
    assert not context.failed


def test_the_reel_is_read_before_anything_counts_or_ranks():
    from src.pipeline.registry import ENRICH, INGEST

    order = list(INGEST) + list(ENRICH)
    assert order.index('ingest.highlights') < order.index('enrich.album_keywords')


# -------------------------------------------- what the reel is worth

def standing_frame(ratings, reel):
    return pd.DataFrame({
        'image_id': np.arange(len(ratings), dtype=np.int64),
        'keywords': [[]] * len(ratings),
        'metadata_rating': ratings,
        'in_highlights': reel,
    })


def test_the_reel_stands_for_four_and_a_half_stars():
    from src.core.keywords import HIGHLIGHTS_STARS, standing_ratings

    standing = standing_ratings(standing_frame([0, 3, 4, 5], [True] * 4))
    assert standing.tolist() == [HIGHLIGHTS_STARS, HIGHLIGHTS_STARS,
                                 HIGHLIGHTS_STARS, 5.0]


def test_the_reel_is_a_floor_and_never_lowers_a_rating():
    from src.core.keywords import standing_ratings

    assert standing_ratings(standing_frame([5], [True])).tolist() == [5.0]


def test_the_brief_s_order_holds_end_to_end():
    """`album` > 5 stars > Highlights > 4 stars > unmarked."""
    from src.core.keywords import priority_lift

    frame = pd.DataFrame({
        'image_id': np.arange(5, dtype=np.int64),
        'keywords': [['album'], [], [], [], []],
        'metadata_rating': [0, 5, 0, 4, 0],
        'in_highlights': [False, False, True, False, False],
    })
    album, five, reel, four, nothing = priority_lift(frame)

    assert album > five > reel > four > nothing == 0.0


def test_a_gallery_with_no_reel_column_is_rated_as_before():
    """Adding the signal must not move a gallery whose scenes were never read."""
    from src.core.keywords import metadata_rating_lift

    with_column = standing_frame([5, 4, 0], [False, False, False])
    without = with_column.drop(columns='in_highlights')

    assert (metadata_rating_lift(with_column).tolist()
            == metadata_rating_lift(without).tolist())


def test_a_reel_in_a_gallery_nobody_rated_still_lifts():
    from src.core.keywords import METADATA_RATING_BONUS, metadata_rating_lift

    lift = metadata_rating_lift(standing_frame([0, 0], [True, False]))
    assert lift.tolist() == [METADATA_RATING_BONUS, 0.0]


def test_the_counts_include_the_reel():
    photos = pd.DataFrame({
        'image_id': np.arange(4, dtype=np.int64),
        'keywords': [['album'], ['not for album'], [], []],
        'metadata_rating': [0.0, 5.0, 4.0, 0.0],
        'in_highlights': [False, True, False, True],
    })
    context = get('enrich.album_keywords')()(
        AlbumContext(photos=photos, logger=LOGGER))

    summary = context.keyword_summary
    assert summary.excluded == 1
    assert summary.prioritised == 1
    assert summary.rated == 1, "the 4-star photo; the excluded 5-star is gone"
    assert summary.highlighted == 1, "the excluded photo's reel place goes with it"
