"""Which of the picker's decision points settled each category?

Phase 0 of `docs/cpsat_scoring_plan.md`. The loop's pick count per class is
bound by one of a dozen different things, and the difference is invisible from
the outside: a class comes back short because the candidate gate declined it,
because temporal narrowing emptied it, because `_take_all_distinct`
deduplicated it, or because a strategy saturated. Every later phase of that
plan is scored against the table this prints, so it is recorded as a committed
baseline rather than re-derived each time.

The attribution itself lives in `WeddingPicker._note` and `CpSatPicker._pool`,
not here -- an instrument that re-implemented the decision tree would drift
from it. This only runs both pickers and tabulates `per_category`.

    # from cached enriched frames (fast, repeatable)
    python tools/pick_attribution.py --cache <dir> 53459898 53147741

    # or from the request, reading the gallery for real (needs the services)
    python tools/pick_attribution.py --request 53459898_ai

    # write/refresh the committed baseline
    python tools/pick_attribution.py --cache <dir> --out tools/baselines/pick_attribution.json 53459898

A cached frame must carry `general_time`: a SELECT-only driver never runs
`enrich.temporal`, and the CP-SAT model declines without that axis.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

BASELINE = os.path.join('tools', 'baselines', 'pick_attribution.json')

#: Every mechanism the two pickers can name, in the order the loop reaches
#: them. Anything outside this list is a bug in one of them, not in the table.
MECHANISMS = (
    'all_committed',      # nothing left after select.preselect
    'no_allowance',       # select.budget gave the class nothing
    'tiny_shortcut',      # <=2 photos and one slot: taken unscored
    'gate_declined',      # CandidateGate found nothing worth scoring
    'user_preference',    # accessories / wedding dress: the user's pick wins
    'temporal_narrowing', # every photo was temporally isolated
    'greyscale_only',     # no colour in the class, so at most two greyscale
    'take_all_distinct',  # supply <= demand: all of it, minus duplicates
    'strategy_skip',      # the strategy declined outright
    'carry_over',         # the monolith's accident: reused another class's list
    'strategy',           # a strategy chose
    'solver',             # cp-sat only
)


#: Mechanisms whose restraint is a *quality* judgement rather than a diversity
#: one. A photo with no neighbour within twenty minutes is an outlier, and the
#: loop is right to drop it, so a class emptied this way is not headroom.
QUALITY_MECHANISMS = ('temporal_narrowing', 'greyscale_only')


def classify_shortfall(row: Dict) -> str:
    """Why a class came back short -- which decides whether the loop is a
    target or merely a starting point.

    `orphan`     the loop applied a quality filter; matching it is correct.
    `scarcity`   there were never enough photos; matching it is correct.
    `similarity` photos were there and a diversity pass declined them. That is
                 budgeted pages left unfilled, and it is where a model that
                 trades coverage off globally can *beat* the loop rather than
                 imitate it.
    """
    if row.get('bound_by') in QUALITY_MECHANISMS:
        return 'orphan'
    if (row['pool'] - row['committed']) <= row['need']:
        return 'scarcity'
    return 'similarity'


def quiet() -> logging.Logger:
    logger = logging.getLogger('pick-attribution')
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    return logger


def hints_from_request(name: str):
    """Real `aiMetadata`, so `select.preselect` commits what it commits in
    production -- with no hints almost nothing is committed and the table
    describes a request nobody makes."""
    from src.pipeline.contracts import AiHints
    from tools import local_request

    meta = (local_request.load_request(name) or {}).get('aiMetadata') or {}
    return AiHints(photo_ids=list(meta.get('photoIds') or []),
                   person_ids=list(meta.get('personIds') or []),
                   focus=list(meta.get('focus') or []),
                   subjects=[''],
                   density=meta.get('density', 3),
                   present=True)


def select(photos: pd.DataFrame, hints, cpsat: bool) -> Optional[Dict]:
    """Run SELECT once and hand back `per_category`."""
    from src.pipeline import AlbumContext, build_select
    from src.pipeline.contracts import GalleryFacts
    from utils.configs import CONFIGS

    logger = quiet()
    context = AlbumContext(
        logger=logger, request={'rating': []}, photos=photos.copy(),
        available_photo_ids=[], hints=hints,
        facts=GalleryFacts(is_wedding=True, is_artificial_time=False,
                           model_version=2),
    )
    original = CONFIGS['pick_cpsat']
    CONFIGS['pick_cpsat'] = {**original, 'enabled': cpsat}
    try:
        context = build_select(logger=logger).run(context)
    finally:
        CONFIGS['pick_cpsat'] = original

    if context.failed:
        print(f"    SELECT failed: {context.error}")
        return None

    # The substage records its own duration, so this is the pick alone rather
    # than the budget and preselect either picker shares. Watched because the
    # coverage dimensions add variables, and a model that solves beautifully in
    # four minutes is no use in a queue worker.
    picked = next((record for record in context.diagnostics
                   if record.name == 'select.pick'), None)
    return {
        'committed': len(context.selection_plan.committed),
        'total': len(context.selection.photo_ids),
        'photo_ids': list(context.selection.photo_ids),
        'per_category': context.selection.per_category,
        'seconds': round(picked.seconds, 2) if picked else None,
    }


def load_photos(gallery: str, cache: Optional[str], request: Optional[str]):
    """A cached enriched frame if one is offered, else read the gallery."""
    if cache:
        path = os.path.join(cache, f'enriched_{gallery}.pkl')
        if os.path.exists(path):
            from ptinfra.proto.pb import BGSegmentation_pb2  # noqa: F401
            payload = pd.read_pickle(path)
            return payload['df'] if isinstance(payload, dict) else payload
        print(f"    no cached frame at {path}")

    if not request:
        return None
    return read_gallery(request)


def read_gallery(name: str) -> Optional[pd.DataFrame]:
    """Read and enrich a gallery for real: INGEST + ENRICH, as the service does.

    The slow path, kept so the baseline is reproducible without anyone's local
    cache. Needs Mongo, Qdrant and blob storage reachable.
    """
    from datetime import datetime as _datetime

    from ptinfra import intialize
    from ptinfra.config import get_variable
    from ptinfra.pt_queue import Message

    intialize('PickAttribution', os.environ.get(
        'HostingSettingsPath',
        '/ptinternal/pictures/hosting/ai_settings_audiobeat.json.txt'))

    from pymongo import MongoClient
    from qdrant_client import QdrantClient

    from src.request_processing import read_messages
    from tools import local_request
    from utils.configs import CONFIGS

    class _Source:
        def __init__(self, id):
            self.id = id

    logger = quiet()
    request = local_request.load_request(name)
    collection = MongoClient(get_variable(CONFIGS['DB_CONNECTION_STRING_VAR'])) \
        [CONFIGS['DB_NAME']][CONFIGS['STATUS_COLLECTION_NAME']]
    qdrant = QdrantClient(host=CONFIGS['QDRANT_HOST'], port=6333,
                          grpc_port=6334, prefer_grpc=True)

    messages, error = read_messages(
        [Message(_Source(1), request, None, _datetime.now())],
        collection, qdrant, logger)
    if error is not None:
        print(f"    read failed: {error}")
        return None
    return messages[0].content.get('gallery_photos_info')


def report(gallery: str, photos: pd.DataFrame, hints) -> Optional[Dict]:
    loop = select(photos, hints, cpsat=False)
    solved = select(photos, hints, cpsat=True)
    if loop is None:
        return None

    print(f"\n=== {gallery}: {len(photos)} photos, "
          f"{loop['committed']} committed, "
          f"loop {loop['total']} / cp-sat {solved['total'] if solved else 'n/a'} ===")
    print(f"  select.pick took {loop['seconds']}s for the loop, "
          f"{solved['seconds'] if solved else 'n/a'}s for cp-sat")
    # `chose` is what the picker itself decided: `selected` counts the
    # committed photos too, and their allowance was already spent in
    # `select.preselect`, so comparing `selected` against `need` overstates
    # both the delivery and, where the class overshoots, the surplus.
    print(f"  {'class':<26} {'pool':>5} {'cmtd':>4} {'need':>4} "
          f"{'chose':>5} {'sat':>4} {'bound_by':<19} strategy")

    rows = {}
    sat_categories = (solved or {}).get('per_category', {})
    for category, entry in sorted(loop['per_category'].items()):
        sat = sat_categories.get(category, {})
        committed = entry.get('committed', 0)
        row = {
            'pool': entry.get('actual', 0),
            'committed': committed,
            'need': entry.get('need'),
            'loop_selected': entry.get('selected', 0),
            'loop_chose': entry.get('selected', 0) - committed,
            'cpsat_selected': sat.get('selected', 0),
            'cpsat_chose': sat.get('selected', 0) - sat.get('committed', 0),
            'bound_by': entry.get('bound_by'),
            'cpsat_bound_by': sat.get('bound_by'),
            'strategy': entry.get('strategy'),
        }
        row['shortfall_kind'] = None
        if row['need'] and row['loop_chose'] < row['need']:
            row['shortfall_kind'] = classify_shortfall(row)

        rows[str(category)] = row
        gap = ''
        if row['need']:
            delta = row['loop_chose'] - row['need']
            if delta < 0:
                gap = f"  short by {-delta} ({row['shortfall_kind']})"
            elif delta > 0:
                gap = f"  over by {delta}"
        print(f"  {str(category)[:26]:<26} {row['pool']:>5} {committed:>4} "
              f"{'' if row['need'] is None else row['need']:>4} "
              f"{row['loop_chose']:>5} {row['cpsat_chose']:>4} "
              f"{str(row['bound_by'] or '-'):<19} "
              f"{(row['strategy'] or '')}{gap}")

    unknown = {r['bound_by'] for r in rows.values()} - set(MECHANISMS) - {None}
    if unknown:
        print(f"  !! unrecognised mechanisms: {sorted(unknown)}")

    return {'photos': len(photos), 'committed': loop['committed'],
            'loop_total': loop['total'],
            'cpsat_total': (solved or {}).get('total'),
            'categories': rows}


def summarise(galleries: Dict[str, Dict]) -> None:
    """How often each mechanism binds, and how much it costs against need."""
    counts: Dict[str, int] = {}
    shortfall: Dict[str, int] = {}
    surplus: Dict[str, int] = {}
    divergence: Dict[str, int] = {}
    for result in galleries.values():
        for row in result['categories'].values():
            mechanism = row['bound_by'] or 'none'
            counts[mechanism] = counts.get(mechanism, 0) + 1
            divergence[mechanism] = divergence.get(mechanism, 0) + abs(
                row['cpsat_chose'] - row['loop_chose'])
            if row['need']:
                gap = row['need'] - row['loop_chose']
                if gap > 0:
                    shortfall[mechanism] = shortfall.get(mechanism, 0) + gap
                elif gap < 0:
                    surplus[mechanism] = surplus.get(mechanism, 0) - gap

    print("\n=== which mechanism settles a class, across every gallery ===")
    print(f"  {'mechanism':<20} {'classes':>8} {'short':>6} {'over':>6} "
          f"{'|cp-sat - loop|':>16}")
    for mechanism, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {mechanism:<20} {count:>8} {shortfall.get(mechanism, 0):>6} "
              f"{surplus.get(mechanism, 0):>6} {divergence.get(mechanism, 0):>16}")
    print(f"\n  against the budgeted allowance, the loop is "
          f"{sum(shortfall.values())} photos short and "
          f"{sum(surplus.values())} over")
    print(f"  cp-sat differs from it by {sum(divergence.values())} photos in total")

    kinds: Dict[str, int] = {}
    where: Dict[str, list] = {}
    for gallery, result in galleries.items():
        for category, row in result['categories'].items():
            kind = row.get('shortfall_kind')
            if not kind:
                continue
            gap = row['need'] - row['loop_chose']
            kinds[kind] = kinds.get(kind, 0) + gap
            where.setdefault(kind, []).append(f"{gallery}/{category} -{gap}")

    if kinds:
        print("\n=== is the shortfall the loop being right, or being timid? ===")
        print(f"  orphan     {kinds.get('orphan', 0):>3} photos  "
              f"-- a quality filter dropped them; matching this is correct")
        print(f"  scarcity   {kinds.get('scarcity', 0):>3} photos  "
              f"-- the pool was never big enough; matching this is correct")
        print(f"  similarity {kinds.get('similarity', 0):>3} photos  "
              f"-- photos were there and a diversity pass declined them. "
              f"Budgeted")
        print(f"{'':24}pages left unfilled, so this is headroom, not a target.")
        for kind in ('similarity', 'scarcity', 'orphan'):
            for entry in where.get(kind, []):
                print(f"    {kind:<11} {entry}")

    scoreboard(galleries)


def scoreboard(galleries: Dict[str, Dict], quiet_output: bool = False):
    """Did cp-sat fill the pages the loop left on the table, and leave alone
    the ones it was right to skip?

    Matching the loop's *count* is not the goal and never was. Where the loop
    stopped for scarcity or on a quality filter it was right, and the model
    should stop too. Where it stopped because a diversity pass declined photos
    that were there, those are budgeted pages the album did not get, and
    filling them is the win. So the two halves are scored separately.
    """
    filled = {'similarity': [0, 0], 'scarcity': [0, 0], 'orphan': [0, 0]}
    for result in galleries.values():
        for row in result['categories'].values():
            kind = row.get('shortfall_kind')
            if not kind:
                continue
            gap = row['need'] - row['loop_chose']
            recovered = min(gap, max(0, row['cpsat_chose'] - row['loop_chose']))
            filled[kind][0] += recovered
            filled[kind][1] += gap

    # Taking *fewer* than the loop is a third kind of error, and neither of
    # the two lines below catches it: `restraint` only counts taking more.
    # It hid the distinct-shot rule dropping a photo the loop kept.
    dropped = 0
    for result in galleries.values():
        for row in result['categories'].values():
            dropped += max(0, row['loop_chose'] - row['cpsat_chose'])

    good, bad = filled['similarity'], filled['scarcity'][1] + filled['orphan'][1]
    over = filled['scarcity'][0] + filled['orphan'][0]
    score = {'headroom': (good[0], good[1]), 'restraint': (bad - over, bad),
             'dropped': dropped}
    if not quiet_output:
        print("\n=== did cp-sat fill what the loop left, and stop where it should? ===")
        print(f"  headroom taken   {good[0]}/{good[1]} of the similarity "
              f"shortfall -- higher is better")
        print(f"  restraint kept   {bad - over}/{bad} of the scarcity and "
              f"orphan shortfall left alone -- higher is better")
        print(f"  photos dropped   {dropped} that the loop kept "
              f"-- lower is better")
    return score
    return score


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Attribute each category's pick count to the decision that settled it.")
    parser.add_argument('galleries', nargs='*',
                        help="Gallery ids. With --cache, matched to enriched_<id>.pkl.")
    parser.add_argument('--cache', help="Directory of cached enriched frames.")
    parser.add_argument('--request', action='append', default=[],
                        help="Request name under files/test_requests/ (repeatable). "
                             "Supplies the hints, and the gallery when no cache is given.")
    parser.add_argument('--out', nargs='?', const=BASELINE,
                        help=f"Write the table as JSON. Bare flag writes {BASELINE}.")
    args = parser.parse_args()

    requests = {name.split('_')[0]: name for name in args.request}
    galleries = args.galleries or sorted(requests)
    if not galleries:
        parser.error("name at least one gallery, or pass --request")

    results: Dict[str, Dict] = {}
    for gallery in galleries:
        request = requests.get(gallery)
        photos = load_photos(gallery, args.cache, request)
        if photos is None or photos.empty:
            print(f"\n=== {gallery}: no photos (need --cache or --request) ===")
            continue
        hints = hints_from_request(request) if request else hints_from_request(gallery)
        result = report(gallery, photos, hints)
        if result:
            results[gallery] = result

    if not results:
        return 1

    summarise(results)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        payload = {'recorded': datetime.now().strftime('%Y-%m-%d'),
                   'note': 'Phase 0 baseline for docs/cpsat_scoring_plan.md. '
                           'Later phases are scored against this.',
                   'galleries': results}
        with open(args.out, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=1, sort_keys=True)
        print(f"\nbaseline written to {args.out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
