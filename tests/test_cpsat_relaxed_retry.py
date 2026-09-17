"""The softer second solve, when the first leaves the album visibly thin.

`quota_ceiling` makes the allowance a ceiling with no floor outside the `yes`
classes, so coming in a little under it is the intended behaviour. What is not
intended is `_add_distinct_shots` collapsing a class whose whole supply is one
pose: it caps a class at one photo per (people, subquery) and fires exactly
where the class has no slack. On project 53739038 that took `bride` from 5 to
1, `groom` from 8 to 2 and `first dance` from 4 to 2 -- 12 of the 15 photos
missing from a 74-photo allowance.

    python -m pytest tests/test_cpsat_relaxed_retry.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from src.pipeline.select.cpsat import STRICT, CpSatPicker, Relaxation  # noqa: E402
from test_cpsat import gallery, in_class, pick  # noqa: E402

#: Run 1's `bride`: five frames of one person in one pose, allowance five.
ONE_POSE = [('portrait', [10], 'a formal portrait')] * 5


def retry(**settings):
    """A `relaxed_retry` block, on unless a test says otherwise."""
    base = {
        'enabled': True,
        'shortfall_trigger': 3,
        'drop_distinct_shots': True,
        'drop_exclusions': True,
        'drop_contradictions': True,
        'allowance_is_target': True,
    }
    return {'relaxed_retry': {**base, **settings}}


# --------------------------------------------------------------------------
# The regression
# --------------------------------------------------------------------------

def test_a_class_of_one_pose_is_filled_on_the_second_solve():
    """Strict keeps 1 of 5; four short is past the slack, so the retry runs."""
    photos = gallery(ONE_POSE)

    chosen = pick(photos, {'portrait': 5}, **retry())

    assert len(in_class(photos, chosen, 'portrait')) == 5


def test_without_the_retry_it_stays_at_one():
    """The behaviour being overridden, pinned so the fix is visibly the fix."""
    photos = gallery(ONE_POSE)

    chosen = pick(photos, {'portrait': 5}, **retry(enabled=False))

    assert len(in_class(photos, chosen, 'portrait')) == 1


def test_dropping_distinct_shots_alone_is_enough_here():
    """The other three walls are not what bound this class."""
    photos = gallery(ONE_POSE)

    chosen = pick(photos, {'portrait': 5},
                  **retry(drop_exclusions=False, drop_contradictions=False,
                          allowance_is_target=False))

    assert len(in_class(photos, chosen, 'portrait')) == 5


# --------------------------------------------------------------------------
# When it must stay out of the way
# --------------------------------------------------------------------------

def test_a_miss_inside_the_slack_does_not_trigger_a_retry():
    """Two distinct keys out of three is the loop's own answer, and one short.

    `test_the_same_people_doing_the_same_thing_counts_once` is the same case
    from the other side; this one pins that the retry leaves it alone.
    """
    photos = gallery([
        ('entertainment', [18], 'live band or musician performing'),
        ('entertainment', [18], 'live band or musician performing'),
        ('entertainment', [], 'live band or musician performing'),
    ])

    chosen = pick(photos, {'entertainment': 3}, **retry())

    assert len(in_class(photos, chosen, 'entertainment')) == 2


def test_the_retry_never_overfills_a_class_with_slack():
    """`dancing` holds twelve frames of one couple against an allowance of four.

    The relaxed quota is still an equality, so a class fills to its own
    allowance and no further -- dropping `distinct_shots` must not turn this
    into a twelve-photo class.
    """
    photos = gallery([
        ('dancing', [1, 2], 'bride and groom dancing') for _ in range(12)
    ])

    chosen = pick(photos, {'dancing': 4}, **retry())

    assert len(in_class(photos, chosen, 'dancing')) == 4


def test_a_full_album_is_left_on_the_first_solve():
    photos = gallery([
        ('portrait', [10], 'a formal portrait'),
        ('portrait', [11], 'a formal portrait'),
        ('portrait', [12], 'a formal portrait'),
    ])

    chosen = pick(photos, {'portrait': 3}, **retry())

    assert len(in_class(photos, chosen, 'portrait')) == 3


# --------------------------------------------------------------------------
# The trigger
# --------------------------------------------------------------------------

@pytest.mark.parametrize('trigger, expected', [(3, 5), (4, 1)])
def test_the_trigger_is_a_strict_threshold(trigger, expected):
    """Four short retries at a trigger of three and not at four."""
    photos = gallery(ONE_POSE)

    chosen = pick(photos, {'portrait': 5}, **retry(shortfall_trigger=trigger))

    assert len(in_class(photos, chosen, 'portrait')) == expected


# --------------------------------------------------------------------------
# Relaxation itself
# --------------------------------------------------------------------------

def test_strict_is_falsy_and_any_relaxation_is_truthy():
    assert not STRICT
    assert Relaxation(distinct_shots=True)
    assert Relaxation(allowance_is_target=True)


def test_describe_names_only_what_was_dropped():
    assert STRICT.describe() == 'nothing'
    assert Relaxation(distinct_shots=True).describe() == 'distinct_shots'
    assert 'allowance_as_target' in Relaxation(
        distinct_shots=True, allowance_is_target=True).describe()


def test_a_disabled_retry_asks_for_no_relaxation():
    picker = CpSatPicker.__new__(CpSatPicker)
    picker.cfg = {'relaxed_retry': {'enabled': False}}
    assert picker._relaxation_for(99) is STRICT


def test_the_ceiling_stands_down_only_for_the_relaxed_solve():
    picker = CpSatPicker.__new__(CpSatPicker)
    picker.cfg = {'quota_ceiling': {'enabled': True}}

    picker.relaxation = STRICT
    assert picker._ceiling_on() is True

    picker.relaxation = Relaxation(allowance_is_target=True)
    assert picker._ceiling_on() is False
