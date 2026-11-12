"""The last decision about order: which box on the page each photo goes in.

`_assign_part_photos_order` buckets a page's boxes into area tiers, walks the
tiers largest first, and hands each one the best-ranked photos still in the
pool. It sorts chronologically *inside* a tier and never across tiers, so a page
built from boxes of different sizes reads in rank order: the biggest box takes
the strongest photo whenever it was taken.

A ceremony has to read forward. `CHRONOLOGY_FIRST_CLASSES` already narrows what
stage 2 may emit and what stage 3 pays for, but neither stage reaches this far --
the inversion is introduced after the layout has been chosen and scored, and the
spread scorer cannot see it either, because `get_score` sorts each page's photos
by time before counting inversions. These tests pin the fourth stage: with the
flag on, a page re-reads chronologically among boxes that can hold each other's
photos; with it off, nothing changes.
"""

from src.core.photos import Photo
from src.spreads_layout.main import is_chronology_first
from src.spreads_layout.spreads.spread import SingleSpreadLayout


def photo(photo_id, general_time, rank, ar=0.66):
    return Photo(id=photo_id, ar=ar, color=True, rank=rank, photo_class='ceremony',
                 cluster_label=0, general_time=general_time, original_context='ceremony')


def page(areas, orientations=None, side=0):
    """One page's boxes, in reading order, with the areas given."""
    orientations = orientations or ['portrait'] * len(areas)
    return [{'id': idx, 'side': side, 'position': idx, 'area': area, 'orientation': orientation}
            for idx, (area, orientation) in enumerate(zip(areas, orientations))]


def assign(boxes, photos, chronology_first):
    """The page as laid out, as `(id, time)` pairs in reading order."""
    pool = sorted(photos, key=lambda ph: (ph.rank, ph.general_time))
    left, right = SingleSpreadLayout._assign_part_photos_order(boxes, pool, chronology_first)
    return [(ph.id, ph.general_time) for ph in (left or right) if ph is not None]


#: The oldest photo is the weakest, so the area tiers put it last.
AGAINST_THE_CLOCK = [photo(101, 1000, rank=3.0),
                     photo(102, 1010, rank=1.0),
                     photo(103, 1020, rank=2.0)]


def test_unequal_boxes_read_in_rank_order_without_the_flag():
    laid_out = assign(page([0.40, 0.25, 0.10]), AGAINST_THE_CLOCK, chronology_first=False)
    assert laid_out == [(102, 1010), (103, 1020), (101, 1000)]


def test_unequal_boxes_read_forward_with_the_flag():
    laid_out = assign(page([0.40, 0.25, 0.10]), AGAINST_THE_CLOCK, chronology_first=True)
    assert laid_out == [(101, 1000), (102, 1010), (103, 1020)]


def test_equal_boxes_already_read_forward():
    """One tier, so the existing per-tier sort was enough; the flag changes nothing."""
    boxes = page([0.25, 0.25, 0.25])
    assert (assign(boxes, AGAINST_THE_CLOCK, chronology_first=False)
            == assign(boxes, AGAINST_THE_CLOCK, chronology_first=True)
            == [(101, 1000), (102, 1010), (103, 1020)])


def test_a_photo_never_moves_into_a_box_of_another_orientation():
    """Only the two portraits trade places; the landscape keeps its landscape box."""
    photos = [photo(201, 1020, rank=3.0),
              photo(202, 1010, rank=1.0, ar=1.5),
              photo(203, 1000, rank=2.0)]
    boxes = page([0.40, 0.25, 0.10], ['portrait', 'landscape', 'portrait'])

    laid_out = assign(boxes, photos, chronology_first=True)

    assert [pid for pid, _ in laid_out] == [203, 202, 201]
    assert laid_out[1] == (202, 1010)


def test_the_right_page_is_reordered_on_its_own():
    boxes = page([0.40, 0.25, 0.10], side=1)
    assert assign(boxes, AGAINST_THE_CLOCK, chronology_first=True) == [(101, 1000),
                                                                       (102, 1010),
                                                                       (103, 1020)]


def test_the_page_keeps_the_photos_it_was_given():
    boxes = page([0.40, 0.25, 0.10])
    before = assign(boxes, AGAINST_THE_CLOCK, chronology_first=False)
    after = assign(boxes, AGAINST_THE_CLOCK, chronology_first=True)
    assert sorted(before) == sorted(after)


def test_only_the_named_classes_ask_for_it():
    assert is_chronology_first((6, 'ceremony', -1))
    assert not is_chronology_first((6, 'bride and groom', -1))
    assert not is_chronology_first((14, 'dancing', -1))
