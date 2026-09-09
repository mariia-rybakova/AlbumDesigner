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
from src.pipeline.enrich import timeline as tl  # noqa: E402
from src.pipeline.enrich.ceremony_anchor import (  # noqa: E402
    BRIDE_AISLE, GROOM_AISLE, MAY_KISS_BRIDE, SEND_OFF, CeremonyAnchorSubStage)
from src.selection.auto_selection import load_pre_queries_embeddings  # noqa: E402
from utils.configs import CONFIGS  # noqa: E402

MODEL_VERSION = 2


def _bank(name, model_version=MODEL_VERSION):
    bank = np.asarray(load_pre_queries_embeddings(name, model_version), dtype=np.float32)
    return bank / np.linalg.norm(bank, axis=1, keepdims=True)


def _banks(model_version):
    """The real concept banks for one space, so "looks like X" means the same
    here as in production."""
    return {'send_off': _bank(CONFIGS['send_off_concept'], model_version),
            'bride': _bank(CONFIGS['aisle_concepts']['bride'], model_version),
            'groom': _bank(CONFIGS['aisle_concepts']['groom'], model_version)}


BANKS = _banks(MODEL_VERSION)
CONCEPT = BANKS['send_off']
DIM = CONCEPT.shape[1]

#: image_class indices into utils.configs.label_list
CLASS = {'ceremony': 6, 'bride and groom': 2, 'other': 30, 'dancing': 8,
         'walking the aisle': 28, 'portrait': 21, 'kiss': 19, 'settings': 23}

BRIDE_ID, GROOM_ID, GUEST_ID = 101, 202, 303


def _embedding(rng, like: float, bank=None) -> np.ndarray:
    """A unit vector whose cosine to `bank` is roughly `like`.

    Defaults to the send-off bank; the processional frames need to resemble
    their own concept instead, or they score near zero against it.
    """
    bank = CONCEPT if bank is None else bank
    base = bank[rng.integers(len(bank))]
    noise = rng.normal(size=bank.shape[1]).astype(np.float32)
    noise -= noise.dot(base) * base
    noise /= np.linalg.norm(noise)
    vector = like * base + np.sqrt(max(0.0, 1 - like ** 2)) * noise
    return vector / np.linalg.norm(vector)


def make_gallery(burst_size=12, burst_score=0.55, burst_label='bride and groom',
                 burst_at='after', kiss_frames=4, kiss_offset=0,
                 bride_walk=5, groom_walk=4, walk_subquery=True, walk_score=0.55,
                 recessional=0, model_version=MODEL_VERSION, seed=3) -> pd.DataFrame:
    """A synthetic wedding timeline with an optional planted send-off.

    Layout: prep -> ceremony (with a climax) -> [burst] -> portraits -> dancing.
    """
    rng = np.random.default_rng(seed)
    banks = _banks(model_version)
    dim = banks['send_off'].shape[1]
    rows = []

    def add(n, label, subquery, score, people=(), bank=None):
        for _ in range(n):
            rows.append({
                'image_id': 900000 + len(rows),
                'image_class': CLASS[label],
                'cluster_class': CLASS[label],
                'cluster_context': label,
                'image_subquery_content': subquery,
                'embedding': _embedding(rng, score, bank if bank is not None else banks['send_off']),
                'model_version': model_version,
                'general_time': len(rows) * 10.0,
                'persons_ids': list(people),
                'bride_id': BRIDE_ID,
                'groom_id': GROOM_ID,
            })

    add(30, 'portrait', 'bride only portrait', 0.10, [BRIDE_ID])      # prep
    add(40, 'settings', 'table with decorations', 0.05)               # venue, no people
    # the processional, before the ceremony
    add(groom_walk, 'walking the aisle',
        'groom waiting for bride at the aisle' if walk_subquery else 'unknown_walking_the_aisle',
        walk_score, [GROOM_ID], banks['groom'])
    add(bride_walk, 'walking the aisle',
        'bride walking down aisle with father' if walk_subquery else 'unknown_walking_the_aisle',
        walk_score, [BRIDE_ID, GUEST_ID], banks['bride'])
    add(25, 'ceremony', 'guests watching ceremony', 0.12, [BRIDE_ID, GROOM_ID])
    add(10, 'ceremony', 'bride and groom exchanging vows', 0.12, [BRIDE_ID, GROOM_ID])
    if burst_at == 'before':
        add(burst_size, burst_label, 'bride and groom during the ceremony', burst_score,
            [BRIDE_ID, GROOM_ID])
    if kiss_offset < 0:                                            # kiss placed early
        add(-kiss_offset, 'ceremony', 'guests watching ceremony', 0.12, [BRIDE_ID, GROOM_ID])
    add(kiss_frames, 'kiss', 'wedding kiss at ceremony', 0.12, [BRIDE_ID, GROOM_ID])
    add(8, 'ceremony', 'ring exchange during ceremony', 0.12, [BRIDE_ID, GROOM_ID])
    # The couple walking back out. The classifier calls this 'walking the aisle'
    # too, which is the mislabel `_demote_late_processional` exists for.
    add(recessional, 'walking the aisle', 'bride and groom walking back down the aisle',
        0.20, [BRIDE_ID, GROOM_ID])
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
    assert picked[Col.SEND_OFF_SCORE].mean() >= tl.floor_for(
        CONFIGS['send_off_burst_floor'], MODEL_VERSION)


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
    """Identity is the one hard requirement: a frame with both of them in it is
    neither of them walking in, so it can never be tagged."""
    df = make_gallery(bride_walk=6)
    walk = df.index[df['image_subquery_content'] == 'bride walking down aisle with father']
    for i in walk:
        df.at[i, 'persons_ids'] = [BRIDE_ID, GROOM_ID]
    context = run(df)
    tagged_ids = set(walked(context, BRIDE_AISLE)['image_id'])
    both = set(df.loc[walk, 'image_id'])
    assert not (tagged_ids & both), "a couple-together frame must never be tagged"


def test_with_no_processional_it_falls_back_to_the_nearest_solo_run():
    """A known consequence of ranking without a floor, recorded deliberately.

    There is no evidence gate, because none survives both embedding spaces: on
    the one v1 gallery with ground truth, the groom's real walk in scores BELOW
    his gallery's median on the groom concept, so any relative floor would
    reject it. With the processional removed from the fixture the detector still
    picks the nearest pre-ceremony solo run -- prep, in that case.

    The cost is bounded: it is a 'yes' event, so it reaches the album as one
    photo. If that turns out to be too loose, the lever is a floor on
    `_rank_run`, not a change to the ranking.
    """
    # v2 is gated, so the weak fallback run is rejected there ...
    v2 = run(make_gallery(bride_walk=0, groom_walk=0))
    assert len(walked(v2, BRIDE_AISLE)) == 0, "v2's floor should reject a prep run"

    # ... but v1 is ungated and will take it, which is the accepted trade
    v1 = run(make_gallery(bride_walk=0, groom_walk=0, model_version=1))
    picked = walked(v1, BRIDE_AISLE)
    assert len(picked) > 0, "documents the v1 fallback; see the docstring"
    assert all(BRIDE_ID in ids and GROOM_ID not in ids for ids in picked['persons_ids'])


def test_subquery_is_an_indication_not_a_requirement():
    """The groom has no 'walking to the altar' subquery at all, so a matching
    subquery cannot be required."""
    context = run(make_gallery(walk_subquery=False))
    assert len(walked(context, BRIDE_AISLE)) > 0, "should tag without a matching subquery"
    assert len(walked(context, GROOM_AISLE)) > 0


def test_frames_after_the_ceremony_are_never_the_processional():
    """The window is bounded above by the ceremony start, so a solo-bride run
    during the reception cannot be picked however aisle-like it looks."""
    df = make_gallery(bride_walk=0, groom_walk=0)
    late = df.index[-30:-24]
    for i in late:
        df.at[i, 'persons_ids'] = [BRIDE_ID]
        df.at[i, 'image_class'] = CLASS['walking the aisle']
    context = run(df)
    assert not (set(walked(context, BRIDE_AISLE)['image_id']) & set(df.loc[late, 'image_id']))


def test_processional_is_capped():
    context = run(make_gallery(bride_walk=30))
    assert len(walked(context, BRIDE_AISLE)) <= CONFIGS['aisle_max_photos']


def test_a_lone_frame_cannot_start_a_processional():
    """A run needs `aisle_min_photos` to be considered at all -- though the
    winning run may later absorb a nearby single frame."""
    df = make_gallery(bride_walk=1, groom_walk=1)
    lone = set(df.loc[df['image_subquery_content'].isin(
        ['bride walking down aisle with father', 'groom waiting for bride at the aisle']),
        'image_id'])
    context = run(df)
    picked = set(walked(context, BRIDE_AISLE)['image_id']) | set(walked(context, GROOM_AISLE)['image_id'])
    assert not (picked & lone), "a one-frame run must not be selected on its own"


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


def test_proximity_decides_when_there_is_no_subquery_evidence():
    """The processional sits next to the ceremony start; prep sits further back.
    With no subquery and a flat concept score, adjacency is what picks."""
    from src.pipeline.enrich import timeline as tl
    df = make_gallery(walk_score=0.50, walk_subquery=False)
    frame = tl.ordered(df)
    ceremony = tl.ceremony_timeline(frame, CONFIGS['send_off_min_photos'])
    context = run(df)
    picked = walked(context, BRIDE_AISLE)
    assert len(picked) > 0
    distance = (frame.loc[picked.index, tl.POSITION] - ceremony.core_start).abs().min()
    assert distance <= CONFIGS['aisle_lead_in'] / 2, (
        f"picked a run {distance} from the ceremony start; prep should lose to adjacency")


def test_subquery_evidence_outweighs_adjacency():
    """A run carrying "bride walking down aisle with father" should win over a
    nearer run that carries nothing -- this is the case that made the concept
    term have to be raw rather than normalised."""
    df = make_gallery(bride_walk=4, walk_subquery=True)
    context = run(df)
    picked = walked(context, BRIDE_AISLE)
    hits = picked['image_subquery_content'].isin(
        ['bride walking down aisle with father']).sum()
    assert hits > 0, "the run with explicit aisle subqueries should be chosen"


def test_thresholds_are_resolved_per_embedding_space():
    """v1 and v2 CLIP put a gallery's cosines on different scales, so a single
    absolute floor cannot serve both: a v2-calibrated floor sits above a v1
    gallery's maximum score. Only the send-off still uses absolute floors -- the
    processional ranks instead, which is scale-free."""
    for setting in (CONFIGS['send_off_photo_floor'], CONFIGS['send_off_burst_floor']):
        assert isinstance(setting, dict) and {1, 2} <= set(setting), setting
        # v1 is deliberately inert until calibrated: a guessed floor produced
        # the wrong answer on all four moments of the one v1 gallery available.
        assert tl.floor_for(setting, 1) >= 1.0, "v1 must stay inert until calibrated" 
    # a plain float still works, for pinning one value deliberately
    assert tl.floor_for(0.4, 1) == 0.4
    # an unknown version falls back to the highest configured one
    assert tl.floor_for({1: 0.1, 2: 0.5}, 99) == 0.5


def test_concept_scores_stay_attached_to_the_right_photos():
    """Regression: the concept scores are computed over the photo table in its
    own row order, but the timeline re-sorts by general_time. Pairing a
    positional array with the sorted frame's index attaches every score to the
    wrong photo whenever the two orders differ -- silently, and only on
    galleries whose stored order is not already chronological."""
    from src.pipeline.enrich import timeline as tl

    df = make_gallery()
    # shuffle the stored order while leaving general_time as the truth
    df = df.sample(frac=1.0, random_state=5).reset_index(drop=True)
    context = run(df)
    assert not context.failed, context.error

    out = context.photos
    recomputed = tl.concept_scores(out, CONFIGS['aisle_concepts']['bride'])
    per_photo = dict(zip(out[Col.IMAGE_ID], recomputed))
    # AISLE_SCORE is the max of the two concepts, so it must be >= the bride one
    for photo_id, stored in zip(out[Col.IMAGE_ID], out[Col.AISLE_SCORE]):
        assert stored >= per_photo[photo_id] - 1e-5, (
            f"photo {photo_id}: stored aisle score {stored:.4f} is below its own "
            f"bride-concept score {per_photo[photo_id]:.4f}")

    # and the picked run must be a genuine solo-bride run before the ceremony
    picked = walked(context, BRIDE_AISLE)
    if len(picked):
        assert all(BRIDE_ID in ids and GROOM_ID not in ids for ids in picked['persons_ids'])


def test_the_aisle_floor_is_applied_per_space():
    """v2 keeps its gate; v1 is ungated because no floor survives that space."""
    from src.pipeline.enrich import timeline as tl
    setting = CONFIGS['aisle_score_floor']
    assert tl.floor_for(setting, 2) > 0, "v2 must stay gated"
    assert tl.floor_for(setting, 1) == 0, "v1 must be ungated"


def test_kiss_is_found_without_a_kiss_subquery():
    """The query bank's kiss subqueries do not fire on every gallery -- one
    validation gallery's only kiss frame is labelled "officiant leading wedding
    ceremony" and carries no identity. A concept bank is the second route in."""
    df = make_gallery(kiss_frames=5)
    # strip the label evidence, leave the frames looking like a kiss
    kiss_rows = df.index[df['image_subquery_content'] == 'wedding kiss at ceremony']
    rng = np.random.default_rng(21)
    kiss_bank = _bank(CONFIGS['kiss_concept'])
    for i in kiss_rows:
        df.at[i, 'image_subquery_content'] = 'officiant leading wedding ceremony'
        df.at[i, 'embedding'] = _embedding(rng, 0.60, kiss_bank)
        df.at[i, 'persons_ids'] = []
    context = run(df)
    picked = set(kissed(context)['image_id'])
    assert picked & set(df.loc[kiss_rows, 'image_id']), (
        "a kiss with no subquery and no identity should still be found by concept")


def test_kiss_concept_floor_is_per_space():
    from src.pipeline.enrich import timeline as tl
    setting = CONFIGS['kiss_concept_floor']
    assert 0 < tl.floor_for(setting, 2) < 1.0, "v2 gated"
    assert tl.floor_for(setting, 1) >= 1.0, "v1 falls back to subqueries only"


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
    """'yes' events, like the send-off: present without taking spreads.

    Only on an album with nothing missing. When there *are* pages to fill, two
    or more of these together are worth one -- see
    test_two_highlights_fill_exactly_one_missing_page.
    """
    images, spreads = _allocate(
        _stocked(**{SEND_OFF: 21, BRIDE_AISLE: 8, GROOM_AISLE: 6}))

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


# -- what the ceremony "yes" classes cost the album ------------------------
#
# A starved gallery used to be allowed to fill from a send-off without limit.
# A burst is fifteen frames against a lookup-table mean of two, so its surplus
# read as seven spare pages and the fill loop would keep drawing on it.

#: A gallery that genuinely cannot fill an album: three categories present, all
#: holding far fewer photos than their share of the profile asks for. It has to
#: be short *this* way now -- since `budget_normalise_present_only`, a gallery
#: merely missing most categories is not short at all, because the few it has
#: are normalised up to cover the whole album.
STARVED = {'ceremony': 8, 'bride and groom': 10, 'dancing': 12}


def _stocked(**extra):
    """Counts generous enough that nothing in the album is missing.

    Every category the focus profile budgets a percentage to has to be present
    and well supplied, not just the ones a gallery usually has: an *absent*
    percentage category misses its whole allowance, so a hand-written list of
    plausible categories still leaves the album nine pages short.
    """
    counts = {
        event: 200
        for event, config in _profile(_quiet()).items()
        if isinstance(config.get('value'), (int, float)) and config['value'] > 0
    }
    counts.update(extra)
    return counts


def test_one_highlight_alone_is_never_worth_a_page():
    """Below the threshold it stays a single photo however short the album is."""
    result = _allocate_full({**STARVED, SEND_OFF: 21})

    assert result.shortfall >= 1, "fixture should be short of pages"
    assert result.ceremony_yes == [SEND_OFF]
    assert not result.ceremony_page_granted
    assert result.images[SEND_OFF] == 1
    assert result.spreads[SEND_OFF] == 0


def test_two_highlights_fill_exactly_one_missing_page():
    result = _allocate_full({**STARVED, SEND_OFF: 21, BRIDE_AISLE: 8})

    assert result.shortfall >= 1, "fixture should be short of pages"
    assert result.ceremony_page_granted
    assert sorted(result.ceremony_yes) == sorted([SEND_OFF, BRIDE_AISLE])
    # One page between them, and a frame each at the very least.
    assert sum(result.spreads[c] for c in result.ceremony_yes) == 1
    assert all(result.images[c] >= 1 for c in result.ceremony_yes)


def test_the_granted_page_is_the_only_one_they_get():
    """The whole point: a 21-frame send-off has seven spare pages of surplus,
    and the fill loop must not be able to reach any of them."""
    result = _allocate_full({**STARVED, SEND_OFF: 21, BRIDE_AISLE: 8, GROOM_AISLE: 6})

    photos = sum(result.images[c] for c in result.ceremony_yes)
    one_page = max(round(result.lookup_table[c][0]) for c in result.ceremony_yes)

    assert sum(result.spreads[c] for c in result.ceremony_yes) == 1
    assert photos <= max(one_page, len(result.ceremony_yes)), (
        f"{photos} photos is more than the one page they were granted")


HIGHLIGHTS = (SEND_OFF, BRIDE_AISLE, GROOM_AISLE)


def test_a_full_album_charges_them_to_the_ceremony():
    """Nothing to fill, so they must not lengthen the album: the photos come
    out of the ceremony own allowance instead."""
    with_them = _allocate_full(_stocked(**{SEND_OFF: 21, BRIDE_AISLE: 8, GROOM_AISLE: 6}))
    without = _allocate_full(_stocked())

    assert with_them.shortfall == 0, "fixture should need no filling"
    assert not with_them.ceremony_page_granted
    assert with_them.charged_to_ceremony == 3, "one photo each"
    assert with_them.images['ceremony'] == without.images['ceremony'] - 3
    assert all(with_them.spreads[c] == 0 for c in with_them.ceremony_yes)


def test_charging_the_ceremony_keeps_the_album_the_same_length():
    with_them = _allocate_full(_stocked(**{SEND_OFF: 21, BRIDE_AISLE: 8, GROOM_AISLE: 6}))
    without = _allocate_full(_stocked())

    assert sum(with_them.images.values()) == sum(without.images.values())


def test_the_kiss_is_budgeted_as_a_yes():
    """may kiss bride carries `yes` in focus_csv.csv, so it joins the ceremony
    group: one photo if the moment happened, and no page share of its own.

    It was a percentage until it was measured going missing. A percentage class
    goes to `select.pick`, where temporal narrowing emptied it -- three frames
    of one instant have no neighbours twenty minutes either side, so all three
    were dropped as isolated and the moment left the album. A `yes` class is
    settled in `select.preselect` and never reaches that filter.
    """
    result = _allocate_full(
        _stocked(**{SEND_OFF: 21, BRIDE_AISLE: 8, MAY_KISS_BRIDE: 6}))

    assert MAY_KISS_BRIDE in result.ceremony_yes
    # No page share of its own is exactly what makes it a token rather than a
    # category: the group between them is worth one page, not one each.
    assert result.spreads[MAY_KISS_BRIDE] == 0
    assert all(result.spreads[c] == 0 for c in result.ceremony_yes)


# -- the processional cannot happen after the ceremony ----------------------
#
# Walking in happens before the ceremony begins. What the classifier labels
# 'walking the aisle' afterwards is the recessional -- the couple walking back
# out -- or the guests leaving, and neither belongs among the frames the album
# builds a processional spread from.


def _context_of(label):
    from src.pipeline.enrich.ceremony_anchor import OTHER, WALKING_THE_AISLE
    return {'other': OTHER, 'aisle': WALKING_THE_AISLE}[label]


def test_a_walking_the_aisle_photo_after_the_ceremony_becomes_other():
    from src.pipeline.enrich.ceremony_anchor import OTHER, WALKING_THE_AISLE

    df = make_gallery(recessional=6)
    late = set(df.loc[df['image_subquery_content'].str.contains('walking back'),
                      'image_id'])
    assert late, "fixture should plant a recessional"

    context = run(df)
    end = context.photos.set_index('image_id')['cluster_context']

    assert all(end[i] != WALKING_THE_AISLE for i in late), (
        "a recessional frame should not still read as the processional")
    assert any(end[i] == OTHER for i in late)


def test_the_processional_before_the_ceremony_is_left_alone():
    """The frames that really are people walking in must survive -- either as
    the class, or claimed by one of the processional tags."""
    from src.pipeline.enrich.ceremony_anchor import OTHER

    df = make_gallery(recessional=6)
    early = set(df.loc[df['image_subquery_content'].str.contains('aisle with father|waiting for bride'),
                       'image_id'])
    assert early, "fixture should plant a processional"

    context = run(df)
    end = context.photos.set_index('image_id')['cluster_context']

    assert all(end[i] != OTHER for i in early), (
        "the real processional was demoted along with the recessional")


def test_only_the_content_class_is_rewritten():
    """`image_class` is what the content model said, and enrich does not edit
    the model's own output -- `enrich.parents` rewrites the same column only."""
    df = make_gallery(recessional=6)
    before = df.set_index('image_id')['image_class'].to_dict()

    context = run(df)
    after = context.photos.set_index('image_id')['image_class'].to_dict()

    assert after == before


def test_nothing_is_demoted_without_a_ceremony_to_anchor_on():
    from src.pipeline.enrich.ceremony_anchor import WALKING_THE_AISLE

    df = make_gallery(recessional=6)
    df = df[~df['cluster_context'].isin(['ceremony', 'kiss'])].reset_index(drop=True)
    was = int((df['cluster_context'] == WALKING_THE_AISLE).sum())

    context = run(df)

    assert int((context.photos['cluster_context'] == WALKING_THE_AISLE).sum()) == was


def test_a_gallery_with_no_recessional_is_untouched():
    from src.pipeline.enrich.ceremony_anchor import OTHER

    plain = make_gallery()
    before = int((plain['cluster_context'] == OTHER).sum())

    context = run(plain)

    # Nothing new in 'other': every 'walking the aisle' frame here is early.
    assert int((context.photos['cluster_context'] == OTHER).sum()) == before


def test_every_yes_category_is_cheap_in_the_lookup_table():
    """A `yes` category is worth one photo, so a page of one should be small.

    It sits out the first round of filling, but the second round grants it a
    page at whatever size the lookup table says. On one real album `food` was
    granted a page at a mean of 4 and took 5 of the 6 photos it had; on the
    degenerate case where nothing else can fill, the same twelve pages cost 49
    photos at the old sizes against 25 at (2, 1).

    So this pins the pair for every category the profile budgets as a string --
    add a `yes` row to focus_csv.csv without a lookup entry to match and this
    is what catches it.
    """
    import csv
    import re

    from utils.lookup_table_tools import wedding_lookup_table

    rows = list(csv.DictReader(open(CONFIGS['focus_csv_path'], encoding='utf-8')))
    columns = [c for c in rows[0] if c.strip().lower() != 'sub event']

    string_valued = [
        re.sub(r"^['\"]+|['\"]+$", '', str(row['sub event'])).strip()
        for row in rows
        if any(str(row[c]).strip().lower() in ('yes', 'no') for c in columns)
    ]
    assert string_valued, "the profile should carry some yes/no categories"

    wrong = {c: wedding_lookup_table.get(c) for c in string_valued
             if wedding_lookup_table.get(c) != (2, 1)}
    assert not wrong, f"yes categories not at (2, 1): {wrong}"


# -- two rounds of filling -------------------------------------------------
#
# `yes` means one photo if the thing happened. That is what it is worth while
# the album can still be built from the categories the profile weighted, so a
# `yes` category sits out the first walk of the fill loop and is only reached
# if the shortfall survives it. Before this, `settings` and `food` each took a
# full page on the *first* pass of a real album, `food` reaching 5 photos of
# the 6 it had.


def _fill_table():
    """Two categories with pages to spare: one weighted, one `yes`.

    The `yes` one has far more spare capacity, so under the old rule it would
    win a page immediately -- which is the behaviour being pinned out.
    """
    table = {
        'weighted': {'value': 5.0, 'photos': 2.0, 'spreads': 1,
                     'miss_spreads': 0, 'over_photos': 8, 'over_spreads': 4.0},
        'yes_thing': {'value': 'yes', 'photos': 1.0, 'spreads': 0,
                      'miss_spreads': 0, 'over_photos': 40, 'over_spreads': 20.0},
    }
    return table, {'weighted': (2, 1), 'yes_thing': (2, 1)}


def test_a_yes_category_sits_out_the_first_round():
    """One walk of the weighted categories fills the album, so the `yes` one is
    never reached and stays at its single photo."""
    from src.pipeline.select.allocation import redistribute

    table, lut = _fill_table()
    redistribute(table, lut, shortfall=2)

    assert table['weighted']['spreads'] == 2, "the weighted category filled it"
    assert table['yes_thing']['spreads'] == 0
    assert table['yes_thing']['photos'] == 1.0


def test_a_yes_category_is_reached_in_the_second_round():
    """When the shortfall outlasts a full walk of the weighted categories, a
    `yes` category is finally worth a page."""
    from src.pipeline.select.allocation import redistribute

    table, lut = _fill_table()
    redistribute(table, lut, shortfall=5)

    assert table['yes_thing']['spreads'] >= 1, "the second round should reach it"
    assert table['weighted']['spreads'] > 1, "and the weighted one went first"


def test_the_yes_category_never_outpaces_the_weighted_one():
    """The invariant: a `yes` category cannot have taken more pages than a
    weighted category that still had capacity to give."""
    from src.pipeline.select.allocation import redistribute

    table, lut = _fill_table()
    redistribute(table, lut, shortfall=6)

    assert table['yes_thing']['spreads'] <= table['weighted']['spreads']


def test_a_yes_category_still_fills_when_nothing_else_can():
    """End to end: the weighted categories have nothing spare, so the album
    would go unfilled if `yes` were held back for good."""
    result = _allocate_full({'ceremony': 4, 'settings': 200})

    assert result.shortfall >= 1
    assert result.images['settings'] > 1, (
        "with nothing else able to fill, settings should be granted a page")


# -- what the album is normalised against ----------------------------------


def test_an_absent_category_is_not_a_shortfall():
    """A wedding with no cake cutting is not an album three pages short.

    The profile weights sum to 107% and a quarter to a third of that routinely
    goes to categories a given wedding has none of. Counted in, the present
    categories asked for only ~70% of the album and the rest came back as
    shortfall for the fill loop to patch with whatever sat high in the file --
    `other` and `None` among them, at 0%.
    """
    plenty = _allocate_full({'ceremony': 200, 'bride and groom': 200, 'dancing': 200})

    assert plenty.shortfall == 0, (
        "three well-stocked categories should fill the album between them")
    assert plenty.images.get('other', 0) == 0
    assert plenty.images.get('None', 0) == 0


def test_the_present_categories_are_normalised_up_to_the_whole_album():
    """Their shares are taken over what the gallery has, so together they ask
    for the whole album rather than their slice of the full profile.

    Measured on the first pass, before the fill loop runs -- afterwards the two
    look alike, because with the old normalisation the loop spends the whole
    shortfall growing these same categories back up. The difference is *where*
    the album's length comes from: the profile's weights, or a scramble down
    `focus_csv.csv`.
    """
    from src.pipeline.select import allocation as al
    from utils.lookup_table_tools import wedding_lookup_table

    counts = {'ceremony': 200, 'bride and groom': 200, 'dancing': 200}
    lut = al.density_scaled(wedding_lookup_table, 3)

    shares = {}
    for present_only in (False, True):
        original = CONFIGS['budget_normalise_present_only']
        CONFIGS['budget_normalise_present_only'] = present_only
        try:
            table = _profile(_quiet())
            al.budget_each(table, counts, lut, target=19)
            shares[present_only] = sum(table[c]['spreads'] for c in counts)
        finally:
            CONFIGS['budget_normalise_present_only'] = original

    assert shares[True] > shares[False], (
        f"the three present categories should claim more of the album: "
        f"{shares[False]:.1f} -> {shares[True]:.1f} spreads")
    assert shares[True] >= 19 * 0.9, (
        f"and very nearly all of it, got {shares[True]:.1f} of 19 spreads")


def test_a_genuine_shortfall_still_counts():
    """Only *absent* categories stop contributing. A category that is there and
    cannot supply its share is still a real gap."""
    result = _allocate_full(dict(STARVED))

    assert result.shortfall >= 1


# -- the port ---------------------------------------------------------------


def test_matches_the_reference_when_no_highlight_is_present():
    """The select.budget allocator is a port of calculate_optimal_selection.
    Where the new rules cannot fire, the two must still agree exactly -- which
    is what makes the rest of these tests measurements of the rules and not of
    a drifting reimplementation.

    Both deliberate departures are switched off here. The ceremony rule used
    to be unable to fire on these fixtures, so it was left alone; once `may
    kiss bride` became a `yes` class the group gained a second member and the
    rule started firing, which is a departure from the monolith and not the
    drift this test exists to catch. So it is now disabled explicitly, by
    putting the minimum out of reach.
    """
    original = CONFIGS.get('budget_normalise_present_only', True)
    original_min = CONFIGS.get('ceremony_yes_min_classes', 2)
    CONFIGS['budget_normalise_present_only'] = False
    CONFIGS['ceremony_yes_min_classes'] = 99
    try:
        # `may kiss bride` used to be stocked here as an extra ordinary
        # category. It is a `yes` class now, so stocking it puts a third
        # member in the ceremony group and fires the settlement this
        # test exists to have switched off. `kiss` is still a
        # percentage, and keeps the coverage that fixture was for.
        for counts in (STARVED, _stocked(), _stocked(**{'kiss': 6})):
            images, spreads = _allocate(dict(counts))
            ref_images, ref_spreads, _lo, _hi = _allocate_reference(dict(counts))

            assert images == ref_images, f"photos diverged on {sorted(counts)}"
            assert spreads == ref_spreads, f"spreads diverged on {sorted(counts)}"
    finally:
        CONFIGS['budget_normalise_present_only'] = original
        CONFIGS['ceremony_yes_min_classes'] = original_min


def test_totals_match_the_reference_too():
    result = _allocate_full(_stocked())
    _i, _s, lo, hi = _allocate_reference(_stocked())

    assert (result.min_total_spreads, result.max_total_spreads) == (lo, hi)


def _quiet():
    import logging
    log = logging.getLogger("t")
    log.addHandler(logging.NullHandler())
    return log


def _profile(log):
    """A fresh focus profile. Both allocators mutate the mapping handed to them."""
    from src.selection.ai_wedding_selection import load_event_mapping
    mapping = load_event_mapping(CONFIGS['focus_csv_path'], log)
    assert SEND_OFF in mapping['brideAndGroom'], "focus_csv.csv row missing"
    return mapping['brideAndGroom']


def _allocate(actual_counts):
    """Run the select.budget allocator over a category->count mapping."""
    from src.pipeline.select.allocation import allocate
    from utils.lookup_table_tools import wedding_lookup_table

    log = _quiet()
    result = allocate(actual_counts, _profile(log), wedding_lookup_table, 3,
                      pd.DataFrame({'persons_ids': [[1, 2]] * 60}), log)
    return result.images, result.spreads


def _allocate_full(actual_counts):
    """As _allocate, but the whole Allocation so the reasoning is visible."""
    from src.pipeline.select.allocation import allocate
    from utils.lookup_table_tools import wedding_lookup_table

    log = _quiet()
    return allocate(actual_counts, _profile(log), wedding_lookup_table, 3,
                    pd.DataFrame({'persons_ids': [[1, 2]] * 60}), log)


def _allocate_reference(actual_counts):
    """The pre-rewrite calculate_optimal_selection, kept as the oracle."""
    from src.selection.ai_wedding_selection import calculate_optimal_selection
    from utils.configs import relations
    from utils.lookup_table_tools import wedding_lookup_table

    log = _quiet()
    images, spreads, lo, hi = calculate_optimal_selection(
        actual_counts, relations['brideAndGroom'], wedding_lookup_table,
        _profile(log), 3, pd.DataFrame({'persons_ids': [[1, 2]] * 60}), log)
    return images, spreads, lo, hi


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} send-off tests passed")


# -- present at the ceremony is not walking into it -------------------------


def _make_groom_walk(persons, subquery):
    """The gallery, with the groom's processional frames rewritten."""
    df = make_gallery()
    walk = (df[Col.CLUSTER_CONTEXT] == 'walking the aisle') & \
           df['persons_ids'].apply(lambda ids: GROOM_ID in ids and BRIDE_ID not in ids)
    assert walk.any(), "fixture should have groom-solo processional frames"
    df.loc[walk, 'persons_ids'] = pd.Series(
        [list(persons)] * int(walk.sum()), index=df.index[walk])
    df.loc[walk, 'image_subquery_content'] = subquery
    return df


def test_a_crowd_the_groom_happens_to_be_in_is_not_his_walk_in():
    """49995684: the frame that reached the album as `groom walking the aisle`
    holds six people and is captioned "guests watching ceremony". He is sitting
    in it. "Solo" only ever excluded the *bride*, so a hall full of guests
    passed the one hard test the tag has."""
    df = _make_groom_walk([GROOM_ID, 11, 12, 13, 14, 15], 'guests watching ceremony')

    context = run(df)

    assert len(walked(context, GROOM_AISLE)) == 0


def test_a_narrow_frame_still_needs_no_caption():
    """The control. Below `aisle_max_people` the identity test is evidence on
    its own, so nothing changes for the frames that were always fine."""
    df = _make_groom_walk([GROOM_ID, 11], 'guests watching ceremony')

    context = run(df)

    assert len(walked(context, GROOM_AISLE)) > 0


def test_a_confirmed_crowd_survives():
    """The bride's genuine wide shots run to five and six people and carry
    "bride walking down aisle with father". Dropping those would trade one bug
    for a worse one."""
    df = _make_groom_walk([GROOM_ID, 11, 12, 13, 14, 15],
                          'groom waiting for bride at the aisle')

    context = run(df)

    assert len(walked(context, GROOM_AISLE)) > 0


def test_the_brides_wide_processional_is_untouched():
    df = make_gallery()
    wide = (df[Col.CLUSTER_CONTEXT] == 'walking the aisle') & \
           df['persons_ids'].apply(lambda ids: BRIDE_ID in ids)
    df.loc[wide, 'persons_ids'] = pd.Series(
        [[BRIDE_ID, GUEST_ID, 51, 52, 53, 54]] * int(wide.sum()), index=df.index[wide])

    context = run(df)

    assert len(walked(context, BRIDE_AISLE)) > 0, (
        "her subquery confirms the frame, however many guests are in it")


def test_the_crowd_gate_can_be_switched_off():
    df = _make_groom_walk([GROOM_ID, 11, 12, 13, 14, 15], 'guests watching ceremony')
    original = CONFIGS.get('aisle_max_people')
    CONFIGS['aisle_max_people'] = 0
    try:
        context = run(df)
    finally:
        CONFIGS['aisle_max_people'] = original

    assert len(walked(context, GROOM_AISLE)) > 0
