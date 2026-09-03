"""Tests for ``enrich.same_sex_couple``.

`map_cluster_label` folds the model's `two brides` / `two grooms` into
`bride and groom`, which is right for the couple shots and wrong for the solo
ones: both partners' portraits land in a single class and the other stays
empty, so the category filter in `CoupleTimelineStrategy` can only ever see one
of them.

    python -m pytest tests/test_same_sex.py -v
"""

from __future__ import annotations

import logging
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import ENRICH, AlbumContext, Col  # noqa: E402
from src.pipeline.contracts import GalleryFacts  # noqa: E402
from src.pipeline.registry import get  # noqa: E402
from utils.reading_tools import label_list, map_cluster_label  # noqa: E402

BRIDE, PARTNER, GUEST = 1, 5, 18

TWO_BRIDES = label_list.index('two brides')
TWO_GROOMS = label_list.index('two grooms')
COUPLE = label_list.index('bride and groom')
BRIDE_CLASS = label_list.index('bride')
GROOM_CLASS = label_list.index('groom')


def quiet():
    logger = logging.getLogger("same-sex-test")
    logger.addHandler(logging.NullHandler())
    return logger


def gallery(rows):
    """``rows`` of ``(image_id, cluster_class, persons_ids)``."""
    return pd.DataFrame([
        {Col.IMAGE_ID: image_id,
         Col.CLUSTER_CLASS: cluster_class,
         Col.CLUSTER_CONTEXT: map_cluster_label(cluster_class),
         Col.PERSONS_IDS: list(people),
         Col.BRIDE_ID: BRIDE,
         Col.GROOM_ID: PARTNER}
        for image_id, cluster_class, people in rows
    ])


def two_brides(solo_class=BRIDE_CLASS):
    """What the model produces: both partners' solos in one class, plus the
    couple shots under `two brides`."""
    rows = []
    rows += [(1000 + i, solo_class, [BRIDE]) for i in range(6)]
    rows += [(2000 + i, solo_class, [PARTNER]) for i in range(6)]
    rows += [(3000 + i, TWO_BRIDES, [BRIDE, PARTNER]) for i in range(8)]
    return gallery(rows)


def run(photos, bride=BRIDE, groom=PARTNER):
    context = AlbumContext(
        logger=quiet(), photos=photos,
        facts=GalleryFacts(is_wedding=True, bride_id=bride, groom_id=groom))
    return get("enrich.same_sex_couple")()(context)


def classes(context):
    return context.photos[Col.CLUSTER_CONTEXT].value_counts().to_dict()


# -- composition -----------------------------------------------------------


def test_runs_after_the_couple_is_resolved():
    """It needs bride_id and groom_id, and it must come before the categories
    are budgeted or picked."""
    order = list(ENRICH)
    assert order.index("enrich.identities") < order.index("enrich.same_sex_couple")


def test_the_model_really_does_flatten_the_couple_classes():
    """The premise. If this ever stops being true the substage is unnecessary."""
    assert map_cluster_label(TWO_BRIDES) == 'bride and groom'
    assert map_cluster_label(TWO_GROOMS) == 'bride and groom'


# -- the split -------------------------------------------------------------


def test_the_second_partner_gets_her_own_class():
    """On the real gallery the `bride` class held 68 photos split 32/32 between
    the two brides, and `groom` was empty -- so 32 solo portraits were invisible
    to the category that would have picked them."""
    context = run(two_brides())

    assert classes(context)['bride'] == 6
    assert classes(context)['groom'] == 6


def test_the_fact_is_recorded():
    assert run(two_brides()).facts.same_sex_couple is True


def test_the_couple_shots_are_left_where_they_are():
    """`bride and groom` is the right class for those; only the solos move."""
    context = run(two_brides())

    assert classes(context)['bride and groom'] == 8


def test_only_exact_solo_frames_move():
    """What moves must be exactly what the receiving category accepts --
    `CoupleTimelineStrategy` filters `groom` to `persons_ids == [groom_id]`."""
    photos = two_brides()
    photos = pd.concat([photos, gallery([
        (4000, BRIDE_CLASS, [PARTNER, GUEST]),   # partner with a guest
    ])], ignore_index=True)

    context = run(photos)
    moved = context.photos[context.photos[Col.CLUSTER_CONTEXT] == 'groom']

    assert all(list(ids) == [PARTNER] for ids in moved[Col.PERSONS_IDS])
    assert 4000 not in set(moved[Col.IMAGE_ID])


def test_two_grooms_is_handled_the_same_way():
    rows = []
    rows += [(1000 + i, GROOM_CLASS, [PARTNER]) for i in range(5)]
    rows += [(2000 + i, GROOM_CLASS, [BRIDE]) for i in range(5)]
    rows += [(3000 + i, TWO_GROOMS, [BRIDE, PARTNER]) for i in range(6)]
    photos = gallery(rows)

    context = run(photos)

    assert context.facts.same_sex_couple is True
    assert classes(context)['groom'] == 5
    assert classes(context)['bride'] == 5


# -- when it must not act --------------------------------------------------


def test_an_opposite_sex_gallery_is_untouched():
    rows = []
    rows += [(1000 + i, BRIDE_CLASS, [BRIDE]) for i in range(6)]
    rows += [(2000 + i, GROOM_CLASS, [PARTNER]) for i in range(4)]
    rows += [(3000 + i, COUPLE, [BRIDE, PARTNER]) for i in range(8)]
    photos = gallery(rows)
    before = classes(AlbumContext(photos=photos.copy()))

    context = run(photos)

    assert context.facts.same_sex_couple is False
    assert classes(context) == before


def test_nothing_moves_when_both_solo_classes_are_populated():
    """An empty counterpart class is the whole signature. If the model did
    manage to fill both, it needs no help."""
    photos = two_brides()
    photos = pd.concat([photos, gallery([(5000, GROOM_CLASS, [PARTNER])])],
                       ignore_index=True)

    context = run(photos)

    # All twelve solos stay put -- six of each partner, as the model left them.
    assert classes(context)['bride'] == 12, "the bride class should be left alone"
    assert classes(context)['groom'] == 1


def test_a_missing_partner_identity_is_not_guessed():
    context = run(two_brides(), groom=None)

    assert context.facts.same_sex_couple is True, "still recorded"
    assert classes(context).get('groom', 0) == 0, "but nothing moved"


def test_skipped_for_a_non_wedding():
    context = AlbumContext(logger=quiet(), photos=two_brides(),
                           facts=GalleryFacts(is_wedding=False))
    context = get("enrich.same_sex_couple")()(context)

    assert context.diagnostics[-1].note == "skipped"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
