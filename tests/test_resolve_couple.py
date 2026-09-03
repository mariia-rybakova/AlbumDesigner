"""Tests for ``resolve_bride_groom``.

The couple ids drive a great deal: the `bride` / `groom` / `bride and groom`
categories all filter on them, the processional detector treats them as
mandatory, and the cover rule needs both. A partner resolved to NaN is not a
loud failure -- the category filter ``persons_ids == [nan]`` simply matches
nothing -- so that partner quietly vanishes from the album.

    python -m pytest tests/test_resolve_couple.py -v
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.read_protos_files import _other_partner, resolve_bride_groom  # noqa: E402

BRIDE, PARTNER, GUEST = 1, 5, 18


def quiet():
    logger = logging.getLogger("resolve-couple-test")
    logger.addHandler(logging.NullHandler())
    return logger


def gallery(rows, main_persons):
    """``rows`` of ``(cluster_context, persons_ids)``."""
    return pd.DataFrame([
        {'image_id': 1000 + i, 'cluster_context': context,
         'persons_ids': list(people), 'main_persons': list(main_persons)}
        for i, (context, people) in enumerate(rows)
    ])


def resolved(df):
    out = resolve_bride_groom(df, quiet())
    return out['bride_id'].iloc[0], out['groom_id'].iloc[0]


# -- the ordinary case -----------------------------------------------------


def test_an_opposite_sex_couple_comes_from_the_two_contexts():
    rows = [('bride', [BRIDE])] * 6 + [('groom', [PARTNER])] * 5
    bride, groom = resolved(gallery(rows, [BRIDE, PARTNER]))

    assert (bride, groom) == (BRIDE, PARTNER)


def test_main_persons_breaks_a_tie():
    """Two identities equally often in the `bride` context; the one the model
    calls a main person wins."""
    rows = ([('bride', [GUEST])] * 4 + [('bride', [BRIDE])] * 4
            + [('groom', [PARTNER])] * 3)
    bride, groom = resolved(gallery(rows, [BRIDE, PARTNER]))

    assert bride == BRIDE
    assert groom == PARTNER


# -- a same-sex couple, which names only one partner -----------------------


def test_the_second_partner_is_found_when_main_persons_is_empty():
    """The gap this closes. A same-sex couple puts both partners into one solo
    context and leaves the other empty, so the second partner is only named by
    `main_persons` -- and that list can be empty, as it was on gallery
    52894932, leaving the partner NaN.
    """
    rows = [('bride', [BRIDE])] * 32 + [('bride', [PARTNER])] * 32
    bride, groom = resolved(gallery(rows, []))

    assert not np.isnan(bride), "one partner should always be named"
    assert not np.isnan(groom), "and so should the other"
    assert {bride, groom} == {BRIDE, PARTNER}


def test_main_persons_is_still_preferred_when_it_has_an_answer():
    """The new path is a fallback, not a replacement."""
    rows = [('bride', [BRIDE])] * 6 + [('bride', [PARTNER])] * 4
    bride, groom = resolved(gallery(rows, [BRIDE, GUEST]))

    assert bride == BRIDE
    assert groom == GUEST, "main_persons named the second one, so it decides"


def test_two_grooms_resolves_symmetrically():
    rows = [('groom', [PARTNER])] * 20 + [('groom', [BRIDE])] * 18
    bride, groom = resolved(gallery(rows, []))

    assert {bride, groom} == {BRIDE, PARTNER}


def test_a_lone_partner_stays_unresolved_rather_than_invented():
    """One identity and nothing else: there is no second partner to find, and
    guessing a guest would be worse than admitting it."""
    rows = [('bride', [BRIDE])] * 8
    bride, groom = resolved(gallery(rows, []))

    assert bride == BRIDE
    assert np.isnan(groom)


# -- the helper ------------------------------------------------------------


def test_other_partner_takes_the_next_most_common():
    from collections import Counter

    counts = Counter({BRIDE: 32, PARTNER: 30, GUEST: 2})

    assert _other_partner(counts, BRIDE) == PARTNER
    assert _other_partner(counts, PARTNER) == BRIDE


def test_other_partner_has_no_answer_from_one_identity():
    from collections import Counter

    assert np.isnan(_other_partner(Counter({BRIDE: 9}), BRIDE))
    assert np.isnan(_other_partner(Counter(), BRIDE))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
