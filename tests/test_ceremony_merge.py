"""The ceremony sequence is bounded by the clock, not by a class name.

Most of a wedding album is edited rather than reported: on a dance floor or in a
couple session the design decides the order of a page, and two frames a minute
apart may sit either way round. The processional, the vows and the send-off are
the exception -- that stretch is a narrative, and printing it out of order reads
as a mistake.

The couple's own frames at the altar are the case that forces a time rule. The
content model files them 'bride and groom', exactly like the ones taken two
hours later at sunset; nothing but the clock tells the two apart. A class-level
rule therefore either misses the altar frames or drags every couple group in the
album in with them.

So `ceremony_window` reads the span off the classes that *locate* the sequence,
and `is_chronology_first` asks whether a group overlaps it. These tests pin both
halves, and close with the group that prompted the rule, carried here as its
real times.
"""

import pandas as pd

from src.groups_operations.ceremony import (GROUP_KEY, ceremony_window, groups_inside,
                                            merge_ceremony_sequence)
from src.core.photos import Photo
from src.spreads_layout.combinations import Combination, filter_time_disjoint
from src.spreads_layout.main import is_chronology_first


def photo(general_time, photo_id=1):
    return Photo(id=photo_id, ar=0.66, color=True, rank=1.0, photo_class='x',
                 cluster_label=0, general_time=general_time, original_context='x')


def group(*times):
    return pd.DataFrame({'general_time': list(times)})


def gallery(groups):
    """One flat frame of photos, which is what the window is read from."""
    rows = [{'cluster_context': context, 'general_time': moment}
            for (_, context, _), frame in groups.items()
            for moment in frame['general_time']]
    return pd.DataFrame(rows)


def photos(*times):
    return [photo(t, photo_id=100 + i) for i, t in enumerate(times)]


#: A wedding's shape in seconds: preparations, the sequence, the party.
ALBUM = {
    (2, 'bride', -1): group(1000, 1200),
    (6, 'groom walking the aisle', -1): group(5000, 5100),
    (6, 'ceremony', -1): group(5200, 5600),
    (6, 'bride and groom', -1): group(5400, 5800),      # at the altar
    (6, 'send off', 3): group(6000, 6100),
    (7, 'bride and groom', -1): group(6500, 7000),      # after it
    (14, 'dancing', -1): group(20000, 30000),
}


def test_the_window_spans_the_sequence_not_the_album():
    assert ceremony_window(gallery(ALBUM)) == (5000, 6100)


def test_counts_are_not_frames_and_do_not_bring_the_album_down():
    """`album_processing` keeps two dicts on the same keys and only one holds frames.

    `get_images_per_groups` maps a group to its photo *count*; the frames come
    from `updated_groups`. Handed the wrong one, the window is worth losing and
    the album is not -- there is no try/except around this call, so raising here
    fails the whole request.
    """
    counts = {(6, 'ceremony', -1): 4, (6, 'send off', 3): 2}

    assert ceremony_window(counts) is None


def test_a_group_with_no_usable_time_is_skipped_not_fatal():
    album = dict(ALBUM)
    album[(6, 'may kiss bride', -1)] = pd.DataFrame({'general_time': [None, None]})

    assert ceremony_window(gallery(album)) == (5000, 6100)


def test_a_gallery_with_no_sequence_has_no_window():
    """A non-wedding album, or one whose every processional frame was dropped."""
    assert ceremony_window({(14, 'dancing', -1): group(1, 2)}) is None
    assert ceremony_window(gallery({})) is None


def test_every_class_of_the_sequence_reads_forward():
    """Including the ones the merge may leave standing alone."""
    assert is_chronology_first((6, 'ceremony', -1))
    assert is_chronology_first((6, 'groom walking the aisle', -1))
    assert is_chronology_first((6, 'send off', 3))


def test_the_couple_is_never_chronological_by_class():
    """It gets the rule by being folded into the sequence, never by its label."""
    assert not is_chronology_first((6, 'bride and groom', -1))
    assert not is_chronology_first((7, 'bride and groom', -1))
    assert not is_chronology_first((14, 'dancing', -1))


#: The group that prompted this, as its own run recorded it: six photos over two
#: spreads, times in seconds from the gallery's start. Stage 2 scored the
#: contiguous split best and stage 3 shipped the worst of the four, which put a
#: 20:21 frame on the first spread and 20:15 and 20:18 on the second.
AT_THE_ALTAR = [15829, 16069, 16130, 16310, 16477, 16651]

#: The four splits stage 2 generated for it, as photo indices into the above.
CANDIDATES = [[[0, 1, 2], [3, 4, 5]],
              [[0, 1, 3], [2, 4, 5]],
              [[0, 2, 3], [1, 4, 5]],
              [[0, 1, 4], [2, 3, 5]]]      # what shipped


def test_the_filter_keeps_only_the_contiguous_split_of_the_real_group():
    pool = photos(*AT_THE_ALTAR)
    kept = filter_time_disjoint([Combination(spreads=c) for c in CANDIDATES], pool)

    assert [c.spreads for c in kept] == [[[0, 1, 2], [3, 4, 5]]]


def test_the_real_group_is_folded_in_once_the_window_comes_from_the_gallery():
    """The case that drove the window off the selection and onto the gallery.

    On the run this is taken from, the earliest processional frame that survived
    selection was at 20:15:19. The couple group opens at 20:10:50 -- inside the
    processional, but before any frame of it was kept -- so against an
    album-derived window the group had a foot outside a span its own photos
    helped define, containment excluded it, and it stayed the atomic block that
    printed the bride entering after the vows.

    The gallery kept processional frames from 20:09 regardless of what selection
    did with them, and against that window the group is wholly inside.
    """
    album = {(6, 'walking the aisle', -1): frame(16098, 16282, start_id=1),
             (6, 'send off', 3): frame(17407, 17412, start_id=10),
             (6, 'bride and groom', -1): frame(*AT_THE_ALTAR, start_id=20)}
    photos = album_frame(album)
    whole_gallery = pd.DataFrame({
        'cluster_context': ['walking the aisle', 'walking the aisle', 'send off'],
        'general_time': [15780, 16098, 17412]})

    assert ceremony_window(photos) == (16098, 17412)
    assert AT_THE_ALTAR[0] < 16098
    assert (6, 'bride and groom', -1) not in groups_inside(photos, ceremony_window(photos))

    assert (6, 'bride and groom', -1) in groups_inside(
        photos, ceremony_window(whole_gallery))


# -- the merge -------------------------------------------------------------
#
# A group is an atomic block downstream, which across the ceremony prints the
# bride entering after the vows. Merging the window's groups removes the
# boundary rather than trying to order across it. The risk it carries is the
# only thing worth testing hard: that it reaches a photo from elsewhere in the
# evening.

def frame(*times, start_id=1):
    return pd.DataFrame({'image_id': [start_id + i for i in range(len(times))],
                         'general_time': list(times)})


def album_frame(groups):
    """One photos_df out of {group_key: frame}, as the tree holds it."""
    rows = []
    for (cluster, context, sub), photos in groups.items():
        for row in photos.itertuples():
            rows.append({'image_id': row.image_id, 'general_time': row.general_time,
                         'time_cluster': cluster, 'cluster_context': context,
                         'group_sub_index': sub, 'group_size': len(photos)})
    return pd.DataFrame(rows)


SEQUENCE = {
    (6, 'groom walking the aisle', -1): frame(5000, 5100, start_id=1),
    (6, 'ceremony', -1): frame(5200, 5300, 5400, start_id=10),
    (6, 'bride and groom', -1): frame(5250, 5350, start_id=20),   # at the altar
    (6, 'send off', 3): frame(6000, 6100, start_id=30),
    (11, 'speech', -1): frame(40000, 41000, start_id=40),
    (12, 'bride and groom', -1): frame(50000, 51000, start_id=50),
}


def counts(photos):
    return {name: len(f) for name, f in photos.groupby(GROUP_KEY)}


def merged(groups):
    """What the tree does: merge over the album's own frame as its gallery."""
    photos = album_frame(groups)
    return merge_ceremony_sequence(photos, photos)


def test_the_sequence_becomes_one_group():
    after = counts(merged(SEQUENCE))

    assert after[(6, 'ceremony', -1)] == 9    # 2 aisle + 3 ceremony + 2 couple + 2 send-off
    assert (6, 'bride and groom', -1) not in after
    assert (6, 'groom walking the aisle', -1) not in after
    assert (6, 'send off', 3) not in after


def test_nothing_outside_the_window_is_touched():
    photos = merged(SEQUENCE)
    after = counts(photos)

    assert after[(11, 'speech', -1)] == 2
    assert after[(12, 'bride and groom', -1)] == 2
    moved = photos[photos['cluster_context'] == 'ceremony']
    assert moved['general_time'].min() == 5000
    assert moved['general_time'].max() == 6100


def test_group_size_is_recomputed_for_the_merged_group():
    """Stale sizes would carry the old groups' numbers into the split rules."""
    photos = merged(SEQUENCE)
    sequence = photos[photos['cluster_context'] == 'ceremony']

    assert set(sequence['group_size']) == {9}


def test_a_stray_late_label_cannot_stretch_the_window_over_the_evening():
    """The failure the gap guard exists for.

    One frame the content model called 'ceremony' at midnight would, on a plain
    hull, run the window from the processional to it and swallow every group in
    between -- the speeches, the cake, the couple's late session.
    """
    with_stray = dict(SEQUENCE)
    with_stray[(20, 'ceremony', 1)] = frame(60000, start_id=60)

    assert ceremony_window(gallery(with_stray)) == (5000, 6100)

    after = counts(merged(with_stray))
    assert after[(11, 'speech', -1)] == 2
    assert after[(12, 'bride and groom', -1)] == 2
    assert after[(20, 'ceremony', 1)] == 1


def test_a_group_with_a_foot_outside_the_window_is_not_merged():
    """Containment, not overlap -- the safety of the whole merge."""
    straddling = dict(SEQUENCE)
    straddling[(6, 'bride and groom', -1)] = frame(5250, 9000, start_id=20)
    photos = album_frame(straddling)
    window = ceremony_window(gallery(straddling))

    assert (6, 'bride and groom', -1) not in groups_inside(photos, window)
    assert counts(merged(straddling))[(6, 'bride and groom', -1)] == 2


def test_the_window_comes_from_the_gallery_not_the_selection():
    """Selection is not deterministic; the window must not move with it.

    Here the only processional frame that survived selection is late, so a
    window read off the album would start after the couple's first frame and
    leave that group straddling. Read off the gallery it does not.
    """
    album = {(6, 'walking the aisle', -1): frame(5300, start_id=1),
             (6, 'bride and groom', -1): frame(5100, 5400, start_id=10),
             (6, 'send off', 3): frame(6000, start_id=20)}
    whole_gallery = pd.DataFrame({
        'cluster_context': ['walking the aisle', 'walking the aisle', 'send off'],
        'general_time': [5000, 5300, 6000]})
    photos = album_frame(album)

    assert ceremony_window(photos) == (5300, 6000)          # the album alone
    assert ceremony_window(whole_gallery) == (5000, 6000)   # the gallery

    assert (6, 'bride and groom', -1) not in groups_inside(photos, ceremony_window(photos))
    assert (6, 'bride and groom', -1) in groups_inside(photos, ceremony_window(whole_gallery))


def test_an_album_with_no_ceremony_is_left_alone():
    plain = {(11, 'speech', -1): frame(40000, 41000, start_id=40),
             (14, 'dancing', -1): frame(50000, 51000, start_id=50)}
    photos = album_frame(plain)

    assert counts(merge_ceremony_sequence(photos, photos)) == counts(photos)


def test_the_flag_turns_the_whole_thing_off():
    """So an A/B against the old behaviour costs one config line."""
    from utils.configs import CONFIGS

    photos = album_frame(SEQUENCE)
    CONFIGS['merge_ceremony_sequence'] = False
    try:
        assert counts(merge_ceremony_sequence(photos, photos)) == counts(photos)
    finally:
        CONFIGS['merge_ceremony_sequence'] = True
