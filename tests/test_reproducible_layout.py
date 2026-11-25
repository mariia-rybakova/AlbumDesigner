"""One request laid out twice is the same album.

The layout search samples whenever a stage generates more candidates than it is
allowed to score, and for a long time it sampled from the global `random`. Two
runs of one request then evaluated different candidate sets and could land on
different spreads from identical photos -- measured on one wedding, two runs at
the same commit chose the same 107 photos and cut them into the same 18 spreads
in a different page order.

That is the kind of defect no amount of reading finds and no ordinary test
catches, because each run is individually correct. It only shows up when a run
is compared with itself. So these tests do that: drive the sampling twice and
require the same answer, and separately require that an explicit reseed still
changes it, because per-album variety depends on that.
"""

import random

from src.spreads_layout import math_tools
from src.spreads_layout.math_tools import limit_sample_size


def sample_twice(population, keep, between=None):
    """`limit_sample_size` run twice from the same seed, with `between` in the middle."""
    math_tools.reseed(0)
    first = limit_sample_size(list(population), keep)
    if between is not None:
        between()
    math_tools.reseed(0)
    second = limit_sample_size(list(population), keep)
    return first, second


def test_the_same_seed_samples_the_same_candidates():
    first, second = sample_twice(range(500), 50)

    assert first == second
    assert len(first) == 50


def test_the_global_random_cannot_disturb_it():
    """The search draws from its own generator, so nothing else reseeding matters.

    `pipeline.albums` seeds the global RNG per album, and before this the layout
    search inherited that by accident. Whether it varies is now a decision, not
    a side effect of what else happened to draw a number first.
    """
    def meddle():
        random.seed(999)
        random.random()

    first, second = sample_twice(range(500), 50, between=meddle)

    assert first == second


def test_reseeding_is_what_changes_the_answer():
    """Per-album variety rides on this; a generator nothing can move is no use."""
    math_tools.reseed(0)
    first = limit_sample_size(list(range(500)), 50)
    math_tools.reseed(1)
    second = limit_sample_size(list(range(500)), 50)

    assert first != second


def test_a_list_inside_the_cap_is_returned_whole_and_in_order():
    """No sampling, so no seed matters -- and the order is the caller's."""
    math_tools.reseed(0)
    assert limit_sample_size([3, 1, 2], 5) == [3, 1, 2]


def test_every_sampling_site_draws_from_the_one_generator():
    """A site left on the global `random` would be invisible in the tests above.

    Checked as source, deliberately: the other sites sit behind layout data that
    a unit test cannot cheaply build, and the failure being guarded against is
    someone adding a fifth `random.sample` rather than any of the four behaving
    badly.
    """
    import inspect

    from src.spreads_layout.spreads import list_of_spreads

    for module in (math_tools, list_of_spreads):
        source = inspect.getsource(module)
        for call in ('random.sample(', 'random.choice(', 'random.shuffle('):
            offenders = [line.strip() for line in source.splitlines()
                         if call in line and 'SEARCH_RNG' not in line]
            assert not offenders, f'{module.__name__} draws from the global RNG: {offenders}'
