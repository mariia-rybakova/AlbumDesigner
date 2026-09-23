"""The time-based split's allow-list: which classes a temporal hole may cut.

`split.get_split_points` cuts a group wherever more than 2 photos from other
groups sit between two consecutive photos of it, but only for the classes in
`TIME_SPLIT_ALLOWED_CLASSES`. The criterion for that list is temporal
*tightness*, not subject importance -- see the comment on the tuples in
`utils.configs`.

The fixture below is one wedding gallery's split stage. Its EXIF was rejected, so
`general_time` is the artificial timeline rebuilt from scene order -- which is
the axis the split reads either way.

There `couple` was one of only two groups out of sixteen to leave the split
stage uncut, still spanning the whole day at 27670 wide where every class that
was cut came out at 3300 or less. The merge stage then grew it from 7 photos to
10, because a group nothing ever split is the one with room to absorb
singletons.
"""

import pytest

from src.groups_operations.split import get_split_points, split_diverse_group
from utils.configs import TIME_SPLIT_ALLOWED_CLASSES

# The 112 photos the timeline was built from: every photo in a group of
# size >= CONFIGS['max_img_split'], sorted. Singletons never reach it, which is
# why ">2 photos between" is a coarse signal.
RUN_TIMELINE = [
    400, 1370, 1400, 1790, 3310, 3400, 4600, 4670, 4890, 4970, 5010, 5020,
    5090, 5250, 5360, 5370, 5650, 5660, 5760, 5880, 6090, 6180, 6240, 6380,
    6410, 6600, 7500, 7690, 7820, 8350, 8850, 9710, 10060, 10450, 10480, 10500,
    12100, 12120, 12840, 12980, 13630, 13720, 13810, 14180, 14230, 14260,
    14270, 14520, 14590, 14880, 15110, 15380, 15410, 15460, 15890, 15900,
    15920, 16600, 16670, 16710, 16720, 16840, 17290, 18170, 18350, 20440,
    20450, 20760, 20790, 21290, 21410, 22190, 22280, 22710, 22830, 22860,
    22870, 22890, 22920, 22950, 22970, 23280, 23290, 23320, 23520, 24000,
    25570, 25650, 25670, 25740, 26160, 26350, 26410, 27150, 27270, 27280,
    27420, 27560, 27620, 27690, 27820, 27990, 28130, 28790, 29460, 30190,
    30460, 30780, 31750, 31790, 31940, 31970,
]

# The seven `couple` photos, spanning 1790..29460 -- first look through the
# reception, the bursts the class is photographed in.
COUPLE_TIMES = [1790, 3310, 4890, 12980, 17290, 18350, 29460]

# The four holes in it, each with 3, 30, 22 and 39 foreign photos inside.
COUPLE_SPLIT_POINTS = [3310, 4890, 12980, 18350]

# The single `dancing` photo at 4970 sits 15470 from the rest of its class, but
# `dancing` spans the whole reception with speeches and cake in the middle, so
# its gap distribution is flat and it stays off the list deliberately.
DANCING_TIMES = [
    4970, 20440, 20450, 21290, 25650, 25670, 25740, 26160, 26350, 26410,
    27150, 27270, 27280, 27420, 27560, 27620, 27690, 27820, 27990, 28130,
]


def widest_span(times, split_points):
    """The widest subgroup `split_diverse_group` would produce."""
    subgroups, current = [], []
    for moment in times:
        current.append(moment)
        if split_points and moment in split_points:
            subgroups.append(current)
            current = []
    if current:
        subgroups.append(current)
    return max(max(s) - min(s) for s in subgroups), len(subgroups)


def test_couple_is_eligible_for_the_time_split():
    """`couple` is the one couple class that was missing from the list.

    Its siblings were all already there -- `bride and groom`, `kiss`,
    `first dance`, `cake cutting`, `send off`, `may kiss bride`.
    """
    assert 'couple' in TIME_SPLIT_ALLOWED_CLASSES


def test_a_couple_group_is_cut_at_its_holes():
    assert get_split_points(RUN_TIMELINE, COUPLE_TIMES, 'couple') == COUPLE_SPLIT_POINTS


def test_cutting_couple_brings_it_back_in_line_with_its_siblings():
    """27670 wide as one group, 1520 as five -- the shape `bride and groom`
    already had on the same run (11 photos, 4 points, 5 subgroups, 570 wide)."""
    assert COUPLE_TIMES[-1] - COUPLE_TIMES[0] == 27670

    span, subgroups = widest_span(
        COUPLE_TIMES, get_split_points(RUN_TIMELINE, COUPLE_TIMES, 'couple'))
    assert (span, subgroups) == (1520, 5)


def test_a_scattered_class_is_still_left_alone():
    """The list did not widen into the classes it deliberately excludes."""
    assert 'dancing' not in TIME_SPLIT_ALLOWED_CLASSES
    assert get_split_points(RUN_TIMELINE, DANCING_TIMES, 'dancing') is None


@pytest.mark.parametrize('sibling', ['bride and groom', 'kiss', 'first dance',
                                     'cake cutting', 'send off', 'may kiss bride'])
def test_every_couple_class_is_eligible(sibling):
    assert sibling in TIME_SPLIT_ALLOWED_CLASSES
