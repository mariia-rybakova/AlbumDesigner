"""Tests for the ceremony-anchored detectors (enrich.ceremony_anchor).

Embeddings are synthesised from the real concept bin, so "looks like a
send-off" means the same thing here as in production.

    python -m pytest tests/test_ceremony_anchor.py -v
    python tests/test_ceremony_anchor.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import AlbumContext, Col  # noqa: E402
from src.pipeline.contracts import GalleryFacts  # noqa: E402
from src.pipeline.enrich.ceremony_anchor import (  # noqa: E402
    BRIDE_AISLE, GROOM_AISLE, MAY_KISS_BRIDE, SEND_OFF, CeremonyAnchorSubStage)
from src.selection.auto_selection import load_pre_queries_embeddings  # noqa: E402
from utils.configs import CONFIGS  # noqa: E402

MODEL_VERSION = 2


def _bank(name):
    bank = np.asarray(load_pre_queries_embeddings(name, MODEL_VERSION), dtype=np.float32)
    return bank / np.linalg.norm(bank, axis=1, keepdims=True)


#: The real concept banks, so "looks like X" means the same here as in production.
CONCEPT = _bank(CONFIGS['send_off_concept'])
BANKS = {'send_off': CONCEPT,
         'bride': _bank(CONFIGS['aisle_concepts']['bride']),
         'groom': _bank(CONFIGS['aisle_concepts']['groom'])}
DIM = CONCEPT.shape[1]

#: image_class indices into utils.configs.label_list
CLASS = {'ceremony': 6, 'bride and groom': 2, 'other': 30, 'dancing': 8,
         'walking the aisle': 28, 'portrait': 21, 'kiss': 19}

BRIDE_ID, GROOM_ID, GUEST_ID = 101, 202, 303


def _embedding(rng, like: float, bank=None) -> np.ndarray:
    """A unit vector whose cosine to `bank` is roughly `like`.

    Defaults to the send-off bank; the processional frames need to resemble
    their own concept instead, or they score near zero against it.
    """
    bank = CONCEPT if bank is None else bank
    base = bank[rng.integers(len(bank))]
    noise = rng.normal(size=DIM).astype(np.float32)
    noise -= noise.dot(base) * base
    noise /= np.linalg.norm(noise)
    vector = like * base + np.sqrt(max(0.0, 1 - like ** 2)) * noise
    return vector / np.linalg.norm(vector)


def make_gallery(burst_size=12, burst_score=0.55, burst_label='bride and groom',
                 burst_at='after', kiss_frames=4, kiss_offset=0,
                 bride_walk=5, groom_walk=4, walk_subquery=True, walk_score=0.55,
                 seed=3) -> pd.DataFrame:
    """A synthetic wedding timeline with an optional planted send-off.

    Layout: prep -> ceremony (with a climax) -> [burst] -> portraits -> dancing.
    """
    rng = np.random.default_rng(seed)
    rows = []

    def add(n, label, subquery, score, people=(), bank=None):
        for _ in range(n):
            rows.append({
                'image_id': 900000 + len(rows),
                'image_class': CLASS[label],
                'cluster_class': CLASS[label],
                'cluster_context': label,
                'image_subquery_content': subquery,
                'embedding': _embedding(rng, score, bank),
                'model_version': MODEL_VERSION,
                'general_time': len(rows) * 10.0,
                'persons_ids': list(people),
                'bride_id': BRIDE_ID,
                'groom_id': GROOM_ID,
            })

    add(30, 'portrait', 'bride only portrait', 0.10, [BRIDE_ID])      # prep
    # the processional, before the ceremony
    add(groom_walk, 'walking the aisle',
        'groom waiting for bride at the aisle' if walk_subquery else 'unknown_walking_the_aisle',
        walk_score, [GROOM_ID], BANKS['groom'])
    add(bride_walk, 'walking the aisle',
        'bride walking down aisle with father' if walk_subquery else 'unknown_walking_the_aisle',
        walk_score, [BRIDE_ID, GUEST_ID], BANKS['bride'])
    add(25, 'ceremony', 'guests watching ceremony', 0.12, [BRIDE_ID, GROOM_ID])
    add(10, 'ceremony', 'bride and groom exchanging vows', 0.12, [BRIDE_ID, GROOM_ID])
    if burst_at == 'before':
        add(burst_size, burst_label, 'bride and groom during the ceremony', burst_score,
            [BRIDE_ID, GROOM_ID])
    if kiss_offset < 0:                                            # kiss placed early
        add(-kiss_offset, 'ceremony', 'guests watching ceremony', 0.12, [BRIDE_ID, GROOM_ID])
    add(kiss_frames, 'kiss', 'wedding kiss at ceremony', 0.12, [BRIDE_ID, GROOM_ID])
    add(8, 'ceremony', 'ring exchange during ceremony', 0.12, [BRIDE_ID, GROOM_ID])
    if burst_at == 'after':
        add(burst_size, burst_label, 'bride and groom during the ceremony', burst_score,
            [BRIDE_ID, GROOM_ID])
    add(40, 'portrait', 'bride and groom posing for a portrait', 0.15, [BRIDE_ID, GROOM_ID])
    add(60, 'dancing', 'guests dancing at wedding reception', 0.12, [GUEST_ID])
    return pd.DataFrame(rows)


def run(df: pd.DataFrame, is_wedding=True) -> AlbumContext:
    context = AlbumContext(photos=df,
                           facts=GalleryFacts(is_wedding=is_wedding, is_artificial_time=False,
                                              bride_id=BRIDE_ID, groom_id=GROOM_ID))
    return CeremonyAnchorSubStage()(context)


def tagged(context) -> pd.DataFrame:
    return context.photos[context.photos[Col.CLUSTER_CONTEXT] == SEND_OFF]


# --------------------------------------------------------------------------


def test_detects_a_planted_send_off():
    context = run(make_gallery())
    assert not context.failed, context.error
    picked = tagged(context)
    assert len(picked) >= CONFIGS['send_off_min_photos'], f"only tagged {len(picked)}"
    assert picked[Col.SEND_OFF_SCORE].mean() >= CONFIGS['send_off_burst_floor']


def test_always_provides_the_score_column():
    """Even when nothing is detected, the declared column must exist."""
    context = run(make_gallery(burst_size=0))
    assert not context.failed, context.error
    assert Col.SEND_OFF_SCORE in context.photos.columns
    assert context.missing(CeremonyAnchorSubStage.provides) == []
    assert len(tagged(context)) == 0


def test_visual_evidence_is_mandatory():
    """A burst in the right place that does not look like a send-off is not one
    — otherwise the plain recessional would be tagged in every wedding."""
    context = run(make_gallery(burst_size=15, burst_score=0.15))
    assert len(tagged(context)) == 0


def test_a_short_burst_is_not_an_event():
    small = CONFIGS['send_off_min_photos'] - 1
    context = run(make_gallery(burst_size=small))
    assert len(tagged(context)) == 0, "fewer than the minimum must not tag"


def test_minimum_sized_burst_does_fire():
    context = run(make_gallery(burst_size=CONFIGS['send_off_min_photos']))
    assert len(tagged(context)) >= CONFIGS['send_off_min_photos']


def test_ineligible_label_is_not_tagged():
    """Send-offs get mislabelled into a known set of classes; a strong burst
    labelled something else (a portrait session with bubbles) is not the event."""
    context = run(make_gallery(burst_label='portrait'))
    assert len(tagged(context)) == 0


def test_a_burst_before_the_ceremony_is_ignored():
    df = make_gallery(burst_size=0)
    rng = np.random.default_rng(9)
    for i in df.index[:12]:
        df.at[i, 'embedding'] = _embedding(rng, 0.55)
        df.at[i, 'image_class'] = CLASS['other']
    context = run(df)
    assert len(tagged(context)) == 0, "prep-time confetti-alikes must not tag"


def test_skipped_for_non_wedding_galleries():
    context = run(make_gallery(), is_wedding=False)
    assert len(tagged(context)) == 0
    assert context.diagnostics[-1].note == "skipped"


def test_no_ceremony_means_no_anchor():
    df = make_gallery()
    df = df[df['cluster_context'] != 'ceremony'].reset_index(drop=True)
    df['image_class'] = df['image_class'].replace(CLASS['ceremony'], CLASS['portrait'])
    context = run(df)
    assert not context.failed
    assert len(tagged(context)) == 0


def test_only_one_send_off_per_gallery():
    """Two qualifying bursts: the better-scoring one wins, not both."""
    df = make_gallery(burst_size=12, burst_score=0.60)
    rng = np.random.default_rng(11)
    # a second, weaker burst right after the first -- located from the fixture
    # rather than hardcoded, so it survives changes to the timeline above it
    planted = df.index[df['image_subquery_content'] == 'bride and groom during the ceremony']
    idx = df.index[planted.max() + 1: planted.max() + 13]
    for i in idx:
        # .at, not .loc: pandas will not broadcast a list of ndarrays
        df.at[i, 'embedding'] = _embedding(rng, 0.42)
        df.at[i, 'image_class'] = CLASS['other']
    context = run(df)
    picked = tagged(context)
    # contiguous -> a single run
    positions = context.photos.index.get_indexer(picked.index)
    assert len(picked) > 0
    assert max(positions) - min(positions) + 1 == len(picked), "tagged photos must be one burst"


def test_ceremony_anchor_is_wired_into_the_enrich_pipeline():
    from src.pipeline import build_enrich
    from src.pipeline.registry import ENRICH
    assert "enrich.ceremony_anchor" in ENRICH
    pipeline = build_enrich()
    assert pipeline.unsatisfied() == []
    # it must run after the classifier that produces the labels it reads
    names = [s.name for s in pipeline]
    assert names.index("enrich.ceremony_anchor") > names.index("enrich.content_class")


# --------------------------------------------------------------------------
# the kiss: the anchor read as a centre
# --------------------------------------------------------------------------


def kissed(context) -> pd.DataFrame:
    return context.photos[context.photos[Col.CLUSTER_CONTEXT] == MAY_KISS_BRIDE]


def test_detects_the_kiss():
    context = run(make_gallery())
    assert not context.failed, context.error
    assert len(kissed(context)) > 0, "kiss frames next to the climax should be tagged"


def test_kiss_is_capped_to_one_moment():
    """A kiss is a moment, not a run of thirty frames."""
    context = run(make_gallery(kiss_frames=30))
    assert len(kissed(context)) <= CONFIGS['kiss_max_photos']


def test_no_kiss_frames_means_no_kiss_tag():
    context = run(make_gallery(kiss_frames=0))
    assert len(kissed(context)) == 0


def test_kiss_far_from_the_anchor_is_ignored():
    """A kiss-subquery frame at the reception is not the ceremony kiss."""
    df = make_gallery(kiss_frames=0)
    # relabel a late reception run as kiss-like, well outside kiss_radius
    for i in df.index[-25:-20]:
        df.at[i, 'image_subquery_content'] = 'bride and groom kissing romantically'
        df.at[i, 'image_class'] = CLASS['kiss']
    context = run(df)
    assert len(kissed(context)) == 0


def test_kiss_and_send_off_do_not_share_photos():
    """Both read the same anchor; a photo must not be claimed by both."""
    context = run(make_gallery())
    overlap = set(kissed(context)[Col.IMAGE_ID]) & set(tagged(context)[Col.IMAGE_ID])
    assert not overlap, f"{len(overlap)} photos tagged as both kiss and send off"


def test_both_detectors_share_one_anchor():
    """The point of merging them: one timeline, one anchor."""
    from src.pipeline.enrich import timeline as tl
    df = make_gallery()
    frame = tl.ordered(df)
    ceremony = tl.ceremony_timeline(frame, CONFIGS['send_off_min_photos'])
    assert ceremony is not None and ceremony.has_climax
    context = run(df)
    # kiss sits near the anchor, send-off after it
    k = frame.loc[kissed(context).index, tl.POSITION] if len(kissed(context)) else None
    s = frame.loc[tagged(context).index, tl.POSITION] if len(tagged(context)) else None
    if k is not None and s is not None:
        assert k.median() < s.median(), "the kiss should precede the send-off"


def test_no_ceremony_means_neither_moment():
    df = make_gallery()
    df = df[~df['image_class'].isin([CLASS['ceremony'], CLASS['kiss']])].reset_index(drop=True)
    context = run(df)
    assert not context.failed
    assert len(kissed(context)) == 0 and len(tagged(context)) == 0


# --------------------------------------------------------------------------
# the processional: the anchor read as an upper bound
# --------------------------------------------------------------------------


def walked(context, tag) -> pd.DataFrame:
    return context.photos[context.photos[Col.CLUSTER_CONTEXT] == tag]


def test_detects_both_processionals_as_separate_classes():
    context = run(make_gallery())
    assert not context.failed, context.error
    assert len(walked(context, BRIDE_AISLE)) > 0, "bride's walk in should be tagged"
    assert len(walked(context, GROOM_AISLE)) > 0, "groom's walk in should be tagged"


def test_identity_is_mandatory():
    """No solo-bride frame before the ceremony means no bride processional,
    however aisle-like the photos look."""
    df = make_gallery(bride_walk=6)
    # keep the frames but put the groom in every one of them
    walk = df.index[30 + 4: 30 + 4 + 6]
    for i in walk:
        df.at[i, 'persons_ids'] = [BRIDE_ID, GROOM_ID]
    context = run(df)
    assert len(walked(context, BRIDE_AISLE)) == 0


def test_the_couple_together_is_neither_processional():
    context = run(make_gallery(bride_walk=0, groom_walk=0))
    assert len(walked(context, BRIDE_AISLE)) == 0
    assert len(walked(context, GROOM_AISLE)) == 0


def test_subquery_is_an_indication_not_a_requirement():
    """The groom has no 'walking to the altar' subquery at all, so a matching
    subquery cannot be required."""
    context = run(make_gallery(walk_subquery=False))
    assert len(walked(context, BRIDE_AISLE)) > 0, "should tag without a matching subquery"
    assert len(walked(context, GROOM_AISLE)) > 0


def test_a_solo_bride_frame_after_the_ceremony_is_not_the_processional():
    df = make_gallery(bride_walk=0, groom_walk=0)
    for i in df.index[-30:-24]:                       # late, during dancing
        df.at[i, 'persons_ids'] = [BRIDE_ID]
        df.at[i, 'image_class'] = CLASS['walking the aisle']
    context = run(df)
    assert len(walked(context, BRIDE_AISLE)) == 0


def test_processional_is_capped():
    context = run(make_gallery(bride_walk=30))
    assert len(walked(context, BRIDE_AISLE)) <= CONFIGS['aisle_max_photos']


def test_a_single_frame_is_not_a_processional():
    context = run(make_gallery(bride_walk=1, groom_walk=1))
    assert len(walked(context, BRIDE_AISLE)) == 0
    assert len(walked(context, GROOM_AISLE)) == 0


def test_the_four_moments_never_share_a_photo():
    """One anchor, four readings — but each photo belongs to at most one."""
    context = run(make_gallery())
    sets = {
        'kiss': set(kissed(context)[Col.IMAGE_ID]),
        'send off': set(tagged(context)[Col.IMAGE_ID]),
        'bride aisle': set(walked(context, BRIDE_AISLE)[Col.IMAGE_ID]),
        'groom aisle': set(walked(context, GROOM_AISLE)[Col.IMAGE_ID]),
    }
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (sets[a] & sets[b]), f"{a} and {b} share {len(sets[a] & sets[b])} photos"


def test_processionals_precede_the_ceremony_climax():
    from src.pipeline.enrich import timeline as tl
    df = make_gallery()
    frame = tl.ordered(df)
    ceremony = tl.ceremony_timeline(frame, CONFIGS['send_off_min_photos'])
    context = run(df)
    for tag in (BRIDE_AISLE, GROOM_AISLE):
        picked = walked(context, tag)
        if len(picked):
            assert frame.loc[picked.index, tl.POSITION].max() <= ceremony.anchor,                 f"{tag} should sit before the climax"


def test_a_processional_with_no_visual_evidence_is_not_tagged():
    """Indications only rank -- but a run with no resemblance at all must not
    tag, or a gallery without a processional labels its prep portraits."""
    context = run(make_gallery(walk_score=0.05, walk_subquery=False))
    assert len(walked(context, BRIDE_AISLE)) == 0
    assert len(walked(context, GROOM_AISLE)) == 0


def test_aisle_classes_are_known_content_classes_everywhere():
    from utils.configs import (limit_imgs, min_images_per_category, priority_categories,
                               relations, selection_threshold,
                               spreads_required_per_category)
    from utils.lookup_table_tools import wedding_lookup_table
    for cls in (BRIDE_AISLE, GROOM_AISLE):
        for focus in ('brideAndGroom', 'parents', 'everyoneElse'):
            assert cls in relations[focus], f"relations[{focus}] missing {cls}"
        for name, table in (('limit_imgs', limit_imgs),
                            ('spreads_required', spreads_required_per_category),
                            ('min_images', min_images_per_category),
                            ('selection_threshold', selection_threshold),
                            ('wedding_lookup_table', wedding_lookup_table)):
            assert cls in table, f"{name} missing {cls}"
        assert cls in priority_categories
        assert wedding_lookup_table[cls] == (2, 1), "mean 2, std 1 as specified"


def test_aisle_classes_get_one_photo_on_a_well_stocked_gallery():
    """'yes' events, like the send-off: present without taking spreads."""
    images, spreads = _allocate({
        'ceremony': 40, SEND_OFF: 21, BRIDE_AISLE: 8, GROOM_AISLE: 6,
        'bride and groom': 80, 'dancing': 120, 'portrait': 40, 'bride': 30,
        'groom': 30, 'first dance': 20, 'speech': 15, 'bride party': 20,
        'groom party': 20, 'settings': 20, 'detail': 20, 'food': 20,
    })
    for cls in (BRIDE_AISLE, GROOM_AISLE):
        assert images.get(cls) == 1, f"{cls} got {images.get(cls)} photos"
        assert spreads.get(cls) == 0


def test_send_off_is_a_known_content_class_everywhere():
    """A new cluster_context must exist in every table the selection stage
    indexes by class, or the budget allocator raises KeyError mid-run."""
    from utils.configs import (limit_imgs, min_images_per_category, priority_categories,
                               relations, selection_threshold,
                               spreads_required_per_category)
    from utils.lookup_table_tools import wedding_lookup_table

    for focus in ('brideAndGroom', 'parents', 'everyoneElse'):
        assert SEND_OFF in relations[focus], f"relations[{focus}]"
    for name, table in (('limit_imgs', limit_imgs),
                        ('spreads_required', spreads_required_per_category),
                        ('min_images', min_images_per_category),
                        ('selection_threshold', selection_threshold),
                        ('wedding_lookup_table', wedding_lookup_table)):
        assert SEND_OFF in table, name
    assert SEND_OFF in priority_categories


def test_budget_allocator_handles_the_new_class():
    """The end-to-end integration point: calculate_optimal_selection indexes the
    lookup table for every focus-profile event, so a missing entry is a
    KeyError mid-allocation."""
    images, spreads = _allocate({'ceremony': 40, SEND_OFF: 21, 'bride and groom': 60,
                                 'dancing': 100})
    assert images is not None, "allocator failed"
    assert SEND_OFF in images and SEND_OFF in spreads


def test_send_off_does_not_draw_from_the_spread_pool():
    """It is a 'yes' event, not a percentage: on a gallery with plenty of
    everything it earns exactly one photo and no spreads, so it never competes
    with the categories the album is actually built from."""
    images, spreads = _allocate({
        'ceremony': 40, SEND_OFF: 21, 'bride and groom': 80, 'dancing': 120,
        'portrait': 40, 'bride': 30, 'groom': 30, 'first dance': 20, 'speech': 15,
        'bride party': 20, 'groom party': 20, 'settings': 20, 'detail': 20, 'food': 20,
    })
    assert images.get(SEND_OFF) == 1, (
        f"a 'yes' event should earn one photo on a well-stocked gallery, "
        f"got {images.get(SEND_OFF)}")
    assert spreads.get(SEND_OFF) == 0


def test_send_off_grows_only_as_filler_and_only_in_pairs():
    """On a starved gallery the rebalance loop may top it up — but the lookup
    table's mean of 2 keeps each step to a pair of photos."""
    images, _spreads = _allocate({'ceremony': 40, SEND_OFF: 21, 'bride and groom': 60,
                                  'dancing': 100})
    grown = images.get(SEND_OFF)
    assert grown > 1, "a starved gallery should be allowed to fill from send off"
    assert (grown - 1) % 2 == 0, (
        f"top-ups should come in pairs (lookup table mean 2), got {grown}")


def _allocate(actual_counts):
    """Run the budget allocator over a category->count mapping."""
    import logging
    from src.selection.ai_wedding_selection import calculate_optimal_selection, load_event_mapping
    from utils.configs import relations
    from utils.lookup_table_tools import wedding_lookup_table

    log = logging.getLogger("t")
    log.addHandler(logging.NullHandler())
    # reloaded per call: the allocator mutates the mapping it is handed
    mapping = load_event_mapping(CONFIGS['focus_csv_path'], log)
    assert SEND_OFF in mapping['brideAndGroom'], "focus_csv.csv row missing"

    images, spreads, _lo, _hi = calculate_optimal_selection(
        actual_counts, relations['brideAndGroom'], wedding_lookup_table,
        mapping['brideAndGroom'], 3,
        pd.DataFrame({'persons_ids': [[1, 2]] * 60}), log)
    return images, spreads


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} send-off tests passed")
