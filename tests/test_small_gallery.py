"""Tests for ``select.small_gallery``: a small wedding goes in nearly whole, gradually."""

from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.contracts import GalleryFacts  # noqa: E402
from src.pipeline.select import small_gallery as sg  # noqa: E402


@pytest.mark.parametrize("n, expected", [
    (40, 1.0), (100, 1.0), (101, 0.99), (150, 0.5), (199, 0.01), (200, 0.0), (5000, 0.0),
])
def test_the_weight_falls_linearly_from_a_hundred_to_two_hundred(n, expected):
    assert sg.weight(n) == pytest.approx(expected)


def test_the_weight_has_no_step():
    """One photo more never moves it by more than one step of the ramp."""
    steps = [abs(sg.weight(n + 1) - sg.weight(n)) for n in range(1, 400)]

    assert max(steps) <= 0.0101


def _context(n, wedding=True, manual=False):
    return types.SimpleNamespace(
        photos=[None] * n, facts=GalleryFacts(is_wedding=wedding),
        selection=types.SimpleNamespace(manual=manual))


def test_only_an_ai_wedding_selection_is_weighted():
    assert sg.weight_for(_context(60)) == 1.0
    assert sg.weight_for(_context(60, manual=True)) == 0.0
    assert sg.weight_for(_context(60, wedding=False)) == 0.0


def test_the_budget_moves_the_weighted_share_toward_everything():
    images, spreads = sg.blended_budget(
        {'ceremony': 4, 'bride': 2}, {'ceremony': 1.0, 'bride': 1.0},
        {'ceremony': 12, 'bride': 6}, {'ceremony': (4, 1), 'bride': (2, 1)}, w=0.5)

    assert images == {'ceremony': 8, 'bride': 4}
    assert spreads == {'ceremony': 2.0, 'bride': 2.0}


def test_the_budget_grows_no_further_than_the_album_can_hold():
    images, _ = sg.blended_budget(
        {'ceremony': 10}, {'ceremony': 2.0}, {'ceremony': 100}, {'ceremony': (5, 1)},
        w=1.0, ceiling=40)

    assert images == {'ceremony': 40}


def test_the_ceiling_never_cuts_below_the_profiles_own_budget():
    images, _ = sg.blended_budget(
        {'ceremony': 50}, {'ceremony': 10.0}, {'ceremony': 100}, {'ceremony': (5, 1)},
        w=1.0, ceiling=40)

    assert images == {'ceremony': 50}


def test_the_picker_scales_its_penalties_by_one_minus_the_weight(monkeypatch):
    from src.pipeline.select import cpsat
    monkeypatch.setattr(sg, 'weight_for', lambda context: 0.75)
    picker = cpsat.CpSatPicker.__new__(cpsat.CpSatPicker)
    picker.penalty_scale = 1.0 - sg.weight_for(None)

    assert picker._scaled(400) == 100
    assert picker._scaled(0) == 0
