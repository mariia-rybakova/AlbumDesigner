"""What users changed between the album the designer proposed and the one they ordered.

Reads the ordered triples found by `scan_provenance.py`, caches them locally,
diffs `response.json:enriched` against `ordered.json:orderedComposition`, and
sorts every run into one bucket:

    no change                    the order is the proposal
    only text                    text entered/edited, nothing else
    only crops                   at least one kept photo re-cropped
    changed photos               the photo set differs
    changed layout               the spread designs differ
    changed photos and layouts   both
    changed more                 something outside those four axes also moved

The buckets are a ladder: a run lands in the highest one it reaches, so "only
crops" means crops and possibly text, not crops alone. Text sits at the bottom
because `placementsTxt` is empty in every proposal -- typing the cover name is
what the user must do, not a rejection of the design.

A crop is only counted for a photo present in BOTH sides. A photo the user
swapped in arrives with its own crop, and calling that a re-crop would count one
edit twice.

    python tools/user_behavior/diff_report.py              # cache, diff, report
    python tools/user_behavior/diff_report.py --refresh    # re-download the cache
    python tools/user_behavior/diff_report.py --download-only

Ordering rewrites a lot that no user touched -- placeholder ids become real ones,
a status is set, the photo objects are hydrated. Which fields are bookkeeping was
decided by the 65 orders the backend itself marked `changed: False`: whatever
moves in those is not a user edit. See the comment above the field lists.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scan_provenance import (  # noqa: E402 - path set above
    KNOWN_FILES, ORDERED, PREFIX, REPO_ROOT, REQUEST, RESPONSE, parse_run_id,
    read_json, scan,
)

DEFAULT_CACHE = os.path.join("output", "user_behavior", "runs")
DEFAULT_OUT = os.path.join("output", "user_behavior", "diff_report")

# The diff compares an explicit list of fields rather than diffing the two
# packages wholesale, because ordering rewrites a great deal that has nothing to
# do with the user. Which fields are bookkeeping was settled empirically against
# the 65 orders the backend itself marked `changed: False`: anything that moves
# in those moved without a user touching it. That ruled out `compositionPackageId`,
# `status`, `revisionCounter`, `userId`, `guserId`, `userJobId`, `externalReference`,
# `origin`, `ptPlatform`, `alerts`, `countCompositions`, `packageDesignId`,
# `specialOptions`, package `tags`, `copies` and `productId`, the per-placement
# `photo` object (null in every proposal, hydrated in every order), composition
# `tags` (they mirror `designId`) and composition `styleId` (only ever 0 <-> -1,
# an unset placeholder).

#: Placement fields that are a real user edit but none of the four axes.
OTHER_IMG_FIELDS = ("rotate", "photoFilter")
#: Package fields likewise. Deliberately short: `tags`, `productId`, `copies`
#: and `storeId` all differ in orders the backend itself marked unchanged
#: (`tags` in 65/65 of them), so they are order bookkeeping, not design edits.
#: `productId` is still reported per run, just not as a change.
OTHER_PACKAGE_FIELDS = ("packageStyleId", "packageTypeId", "packageFinishingId")

CATEGORIES = [
    "no change", "only text", "only crops", "changed photos",
    "changed layout", "changed photos and layouts", "changed more",
]


# --------------------------------------------------------------------------
# local cache
# --------------------------------------------------------------------------

def cache_paths(cache_dir, rec):
    folder = os.path.join(cache_dir, str(rec["projectId"]), rec["runId"])
    return folder, {name: os.path.join(folder, name) for name in KNOWN_FILES}


def fetch_triple(cache_dir, rec, refresh=False):
    """Make sure one run's three files are on disk; return (rec, ok, error)."""
    folder, paths = cache_paths(cache_dir, rec)
    wanted = [n for n in KNOWN_FILES if n in rec["files"]]
    if not refresh and all(os.path.exists(paths[n]) for n in wanted):
        return rec, True, None
    os.makedirs(folder, exist_ok=True)
    try:
        for name in wanted:
            doc = read_json(f"{PREFIX}{rec['projectId']}/{rec['runId']}/{name}")
            with open(paths[name], "w", encoding="utf-8") as fh:
                json.dump(doc, fh)
    except Exception as exc:                            # noqa: BLE001 - reported, not fatal
        return rec, False, str(exc)
    return rec, True, None


def ensure_cache(cache_dir, triples, refresh=False, workers=16):
    ok, failed = [], []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_triple, cache_dir, rec, refresh) for rec in triples]
        for fut in futures:
            rec, good, err = fut.result()
            done += 1
            if done % 50 == 0:
                print(f"  ... {done}/{len(triples)} cached", file=sys.stderr)
            (ok if good else failed).append(rec if good else (rec, err))
    if failed:
        print(f"!! {len(failed)} run(s) could not be cached, e.g. "
              f"{failed[0][0]['runId']}: {failed[0][1]}", file=sys.stderr)
    return ok, failed


def load_triple(cache_dir, rec):
    _, paths = cache_paths(cache_dir, rec)
    out = {}
    for name in KNOWN_FILES:
        if os.path.exists(paths[name]):
            with open(paths[name], encoding="utf-8") as fh:
                out[name] = json.load(fh)
    return out


# --------------------------------------------------------------------------
# the diff
# --------------------------------------------------------------------------

def _ranks(pkg):
    """compositionId -> its position in the album.

    Orders renumber the spreads: the proposal's compositionIds start at 0 and
    the order's at 1 in 27 of the 401 triples, which made every spread look
    re-templated and every photo look moved. Position is what both sides agree
    on, so every per-spread comparison below is keyed by it.
    """
    cids = sorted(c.get("compositionId") for c in pkg.get("compositions") or [])
    return {cid: i for i, cid in enumerate(cids)}


def _crops(placements):
    """photoId -> sorted crop rectangles, so a photo placed twice still compares."""
    by_photo = defaultdict(list)
    for p in placements or []:
        by_photo[p.get("photoId")].append((round(p.get("cropX") or 0.0, 6),
                                           round(p.get("cropY") or 0.0, 6),
                                           round(p.get("cropWidth") or 0.0, 6),
                                           round(p.get("cropHeight") or 0.0, 6)))
    return {k: sorted(v) for k, v in by_photo.items()}


def _slots(pkg):
    """(spread position, boxId) -> the placement sitting there."""
    rank = _ranks(pkg)
    return {(rank.get(p.get("compositionId")), p.get("boxId")): p
            for p in pkg.get("placementsImg") or []}


def _designs(pkg):
    """The design template of each spread, in album order."""
    return [c.get("designId") for c in
            sorted(pkg.get("compositions") or [], key=lambda c: c.get("compositionId"))]


def _signatures(pkg):
    """One signature per spread: its template and which photo sits in which frame.

    This is what "the spread changed" means: a different design, a photo swapped
    for another, or a photo moved to a different frame. Crops and text are
    deliberately outside the signature -- re-cropping a photo or typing a caption
    leaves the spread's layout as the designer composed it.
    """
    rank = _ranks(pkg)
    designs = {rank[c["compositionId"]]: c.get("designId")
               for c in pkg.get("compositions") or []}
    content = defaultdict(list)
    for p in pkg.get("placementsImg") or []:
        content[rank.get(p.get("compositionId"))].append((p.get("boxId"), p.get("photoId")))
    return [(designs.get(r), tuple(sorted(content.get(r, [])))) for r in sorted(designs)]


def _align(a, b):
    """Edit counts between two spread sequences.

    Aligned with difflib rather than position by position: inserting one spread
    at the front shifts every later position, and a naive comparison would call
    a 30-spread album entirely re-laid-out when the user added a single page.
    The alignment charges that as one insert and leaves the rest equal.
    """
    replaced = added = removed = unchanged = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            unchanged += i2 - i1
        elif tag == "replace":
            replaced += max(i2 - i1, j2 - j1)
        elif tag == "insert":
            added += j2 - j1
        elif tag == "delete":
            removed += i2 - i1
    return replaced, added, removed, unchanged


def _spread_edit(proposal, order):
    """How MANY spreads the user changed, and how many only got a new template."""
    replaced, added, removed, unchanged = _align(_signatures(proposal), _signatures(order))
    changed = replaced + added + removed
    # The denominator is the alignment's own span (changed + survived), not the
    # longer album. An alignment can charge a delete and an insert at the same
    # position, so counting against `max(len(a), len(b))` produced 45 changed
    # spreads in a 37-spread album -- and a "100% re-laid-out" verdict for albums
    # that kept half their spreads.
    total = changed + unchanged
    # Secondary: of the album's spreads, how many carry a different design. A
    # spread can change without this (same template, different photo in a frame).
    retemplated, _, _, _ = _align(_designs(proposal), _designs(order))
    return {
        "spreadsChanged": changed,
        "spreadsUnchanged": unchanged,
        "spreadsReplaced": replaced,
        "spreadsAdded": added,
        "spreadsRemoved": removed,
        "spreadsRetemplated": retemplated,
        "spreadsChangedShare": round(changed / total, 4) if total else 0.0,
        # No spread survived the user untouched.
        "allSpreadsChanged": bool(total) and unchanged == 0,
    }


def _boxes(pkg):
    """(spread position, boxId) -> box geometry, to spot a resized frame."""
    rank = _ranks(pkg)
    out = {}
    for c in pkg.get("compositions") or []:
        r = rank.get(c.get("compositionId"))
        for b in c.get("boxes") or []:
            out[(r, b.get("id"))] = tuple(round(b.get(f) or 0.0, 6)
                                          for f in ("x", "y", "width", "height"))
    return out


def _texts(pkg):
    """(spread position, boxId) -> the text and the styling that renders it."""
    rank = _ranks(pkg)
    out = {}
    for p in pkg.get("placementsTxt") or []:
        style = p.get("text") or {}
        out[(rank.get(p.get("compositionId")), p.get("boxId"))] = (
            (p.get("textToRender") or "").strip(),
            tuple(sorted((k, json.dumps(v, sort_keys=True)) for k, v in style.items())),
        )
    return out


def diff_run(proposal, order):
    """Compare one proposal with its order. Returns the axis booleans plus counts."""
    p_img, o_img = proposal.get("placementsImg") or [], order.get("placementsImg") or []
    p_crops, o_crops = _crops(p_img), _crops(o_img)
    p_photos, o_photos = set(p_crops), set(o_crops)

    kept = p_photos & o_photos
    added, removed = o_photos - p_photos, p_photos - o_photos
    # Only photos on both sides can have been re-cropped; a swapped-in photo
    # brings its own crop and is already counted as a photo change.
    recropped = sorted(pid for pid in kept if p_crops[pid] != o_crops[pid])

    p_des, o_des = _designs(proposal), _designs(order)
    spread_edit = _spread_edit(proposal, order)
    p_slots, o_slots = _slots(proposal), _slots(order)
    # A rearrangement of the SAME photos is a layout change too, but only judged
    # over spreads whose photo survived -- otherwise every photo swap reads as one.
    moved = sorted(k for k in set(p_slots) & set(o_slots)
                   if p_slots[k].get("photoId") != o_slots[k].get("photoId")
                   and p_slots[k].get("photoId") in kept
                   and o_slots[k].get("photoId") in kept)

    p_txt, o_txt = _texts(proposal), _texts(order)
    text_changed = any(p_txt.get(k) != o_txt.get(k) for k in set(p_txt) | set(o_txt))

    p_box, o_box = _boxes(proposal), _boxes(order)
    box_changed = any(p_box[k] != o_box[k] for k in set(p_box) & set(o_box))

    other = []
    for field in OTHER_PACKAGE_FIELDS:
        if proposal.get(field) != order.get(field):
            other.append(f"package.{field}")
    for key in set(p_slots) & set(o_slots):
        a, b = p_slots[key], o_slots[key]
        if a.get("photoId") != b.get("photoId"):
            continue                       # a swapped slot: already a photo change
        for field in OTHER_IMG_FIELDS:
            if a.get(field) != b.get(field):
                other.append(f"placementImg.{field}")
    p_rank, o_rank = _ranks(proposal), _ranks(order)
    p_copies = {p_rank[c["compositionId"]]: c.get("copies")
                for c in proposal.get("compositions") or []}
    o_copies = {o_rank[c["compositionId"]]: c.get("copies")
                for c in order.get("compositions") or []}
    if any(p_copies[r] != o_copies[r] for r in set(p_copies) & set(o_copies)):
        other.append("composition.copies")

    return {
        "photosChanged": bool(added or removed),
        "cropsChanged": bool(recropped),
        # The CATEGORY axis stays template-level: a spread that differs only
        # because a photo was swapped into it is a photo change, and letting it
        # also count here would empty the `changed photos` bucket into
        # `changed photos and layouts`. `spreadsChanged` below is the wider,
        # per-spread measure, and is what the magnitude table counts.
        "layoutChanged": p_des != o_des or bool(moved),
        "textChanged": text_changed,
        "otherChanged": bool(other),
        # Reported, not categorised: a resized frame is arguably a layout edit,
        # but it also fired on a control run, so it is not trusted to move a run
        # up the ladder on its own. Left here for whoever wants to dig.
        "boxGeometryChanged": box_changed,
        # Order-time product/quantity choices. Real decisions, but not edits to
        # the design, and they occur in control runs too -- so they stay out of
        # the ladder and are reported on their own.
        "productChanged": proposal.get("productId") != order.get("productId"),
        "copiesOrdered": order.get("copies"),
        "photosProposed": len(p_photos),
        "photosOrdered": len(o_photos),
        # Distinct photos above, filled slots here. They come apart when a photo
        # is repeated on another spread or a repeat is dropped: the selection is
        # untouched, so that is a layout edit, but it would otherwise be invisible.
        "placementsProposed": len(p_img),
        "placementsOrdered": len(o_img),
        "photosAdded": len(added),
        "photosRemoved": len(removed),
        "photosRecropped": len(recropped),
        "photosMoved": len(moved),
        "spreadsProposed": len(p_des),
        "spreadsOrdered": len(o_des),
        # How much of the album was re-laid-out, not just whether any of it was.
        **spread_edit,
        # Spreads whose template survived but whose photos were shuffled between
        # its frames -- a layout edit the designId sequence cannot see.
        "spreadsRearranged": len({k[0] for k in moved}),
        "textsProposed": sum(1 for v in p_txt.values() if v[0]),
        "textsOrdered": sum(1 for v in o_txt.values() if v[0]),
        "otherFields": sorted(set(other)),
    }


def categorize(d):
    """The ladder: a run lands in the highest bucket it reaches."""
    if d["otherChanged"]:
        return "changed more"
    if d["photosChanged"] and d["layoutChanged"]:
        return "changed photos and layouts"
    if d["photosChanged"]:
        return "changed photos"
    if d["layoutChanged"]:
        return "changed layout"
    if d["cropsChanged"]:
        return "only crops"
    if d["textChanged"]:
        return "only text"
    return "no change"


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def build_rows(cache_dir, triples):
    rows, skipped = [], []
    for rec in triples:
        docs = load_triple(cache_dir, rec)
        if RESPONSE not in docs or ORDERED not in docs:
            skipped.append((rec, "missing cached file"))
            continue
        proposal = docs[RESPONSE].get("enriched")
        order = docs[ORDERED].get("orderedComposition")
        if not proposal or not order:
            skipped.append((rec, "no enriched/orderedComposition"))
            continue
        d = diff_run(proposal, order)
        req = docs.get(REQUEST) or {}
        rows.append({
            "projectId": rec["projectId"],
            "runId": rec["runId"],
            "requestedAt": rec.get("requestedAtFromId"),
            "orderedDate": docs[ORDERED].get("orderedDate"),
            "isManual": req.get("isManual", rec.get("isManualByName")),
            "productId": req.get("productId"),
            "changedFlag": docs[ORDERED].get("changed"),
            "category": categorize(d),
            **d,
        })
    return rows, skipped


def print_report(rows, skipped):
    total = len(rows)
    by_cat = Counter(r["category"] for r in rows)

    print(f"\n{'=' * 74}\nPROPOSAL vs ORDER -- {total} ordered triples\n{'=' * 74}")
    print(f"{'category':<30} {'runs':>6} {'share':>7}")
    print("-" * 46)
    for cat in CATEGORIES:
        n = by_cat.get(cat, 0)
        bar = "#" * round(30 * n / total) if total else ""
        print(f"{cat:<30} {n:>6} {100.0 * n / total if total else 0:>6.1f}%  {bar}")

    print(f"\naxes (not exclusive):")
    for axis, label in (("photosChanged", "photo set changed"),
                        ("layoutChanged", "layout changed"),
                        ("cropsChanged", "kept photo re-cropped"),
                        ("textChanged", "text entered/edited"),
                        ("otherChanged", "something else changed")):
        n = sum(1 for r in rows if r[axis])
        print(f"  {label:<26} {n:>6} {100.0 * n / total if total else 0:>6.1f}%")

    # A spread counts as changed when its template, its photos, or which frame a
    # photo sits in differs. Re-cropping a photo or typing a caption leaves the
    # spread as composed, so those alone never move a spread into this count.
    touched = [r for r in rows if r["spreadsChanged"]]
    if touched:
        print(f"\nhow MUCH of the album changed ({len(touched)} runs with at least one "
              f"changed spread;\nspread = different template, different photo, or a photo "
              f"in a different frame):")
        buckets = [(1, 1, "1 spread"), (2, 3, "2-3 spreads"),
                   (4, 10, "4-10 spreads"), (11, 25, "11-25 spreads"),
                   (26, 10 ** 6, "26+ spreads")]
        for lo, hi, label in buckets:
            sel = [r for r in touched if lo <= r["spreadsChanged"] <= hi]
            share = f"{100.0 * len(sel) / len(touched):.1f}%"
            print(f"  {label:<16} {len(sel):>6} {share:>7}   "
                  + "#" * round(30 * len(sel) / len(touched)))
        whole = [r for r in touched if r["allSpreadsChanged"]]
        shares = sorted(r["spreadsChangedShare"] for r in touched)
        median = shares[len(shares) // 2]
        print(f"  {'-' * 44}")
        print(f"  every spread changed: {len(whole)} run(s) "
              f"({100.0 * len(whole) / len(touched):.1f}% of these, "
              f"{100.0 * len(whole) / total:.1f}% of all orders)")
        print(f"  median share of the album changed: {100.0 * median:.0f}%")
        print(f"  spreads: {sum(r['spreadsChanged'] for r in touched)} changed in all "
              f"({sum(r['spreadsReplaced'] for r in touched)} reworked in place, "
              f"{sum(r['spreadsAdded'] for r in touched)} added, "
              f"{sum(r['spreadsRemoved'] for r in touched)} removed)")
        print(f"  of those, {sum(r['spreadsRetemplated'] for r in touched)} took a "
              f"different design template; the rest changed by photo alone")

    informational = [
        ("boxGeometryChanged", "a frame was resized/moved"),
        ("productChanged", "a different product was ordered"),
    ]
    print(f"\nreported, but not category-driving:")
    for key, label in informational:
        n = sum(1 for r in rows if r[key])
        alone = sum(1 for r in rows if r[key] and r["category"] in ("no change", "only text"))
        print(f"  {label:<32} {n:>6} {100.0 * n / total if total else 0:>6.1f}%"
              f"   ({alone} in runs otherwise unchanged)")

    other_fields = Counter(f for r in rows for f in r["otherFields"])
    if other_fields:
        print(f"\nwhat drove 'changed more':")
        for field, n in other_fields.most_common():
            print(f"  {field:<28} {n:>6}")

    print(f"\nmanual vs automatic runs:")
    print(f"  {'category':<30} {'manual':>8} {'auto':>8}")
    for cat in CATEGORIES:
        man = sum(1 for r in rows if r["category"] == cat and r["isManual"])
        auto = sum(1 for r in rows if r["category"] == cat and not r["isManual"])
        if man or auto:
            print(f"  {cat:<30} {man:>8} {auto:>8}")

    # `changed` is the backend's own flag; where it disagrees with the diff, the
    # diff is the one that looked at the content.
    print(f"\nordered.json `changed` vs this diff:")
    agree = Counter((bool(r["changedFlag"]), r["category"] != "no change") for r in rows)
    for (flag, diffed), n in sorted(agree.items()):
        note = "" if flag == diffed else "   <- disagree"
        print(f"  changed={str(flag):<5} diff found changes={str(diffed):<5} {n:>6}{note}")

    kept = [r for r in rows if r["photosProposed"]]
    if kept:
        retained = sum(r["photosProposed"] - r["photosRemoved"] for r in kept)
        proposed = sum(r["photosProposed"] for r in kept)
        print(f"\nphotos: {proposed} proposed, {retained} kept "
              f"({100.0 * retained / proposed:.1f}%), "
              f"{sum(r['photosAdded'] for r in kept)} added by users")

    print(f"\nexamples:")
    for cat in CATEGORIES:
        ex = [r for r in rows if r["category"] == cat][:2]
        for r in ex:
            print(f"  {cat:<30} {r['projectId']}/{r['runId'][:46]}")

    if skipped:
        print(f"\n!! {len(skipped)} run(s) skipped: "
              f"{Counter(reason for _, reason in skipped).most_common()}")


def write_outputs(rows, out_base):
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    csv_path, json_path = out_base + ".csv", out_base + "_summary.json"
    if rows:
        fields = [k for k in rows[0] if k != "otherFields"] + ["otherFields"]
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({**r, "otherFields": ";".join(r["otherFields"])})
    touched = [r for r in rows if r["spreadsChanged"]]
    shares = sorted(r["spreadsChangedShare"] for r in touched)
    summary = {
        "runs": len(rows),
        "layoutMagnitude": {
            "runsWithAChangedSpread": len(touched),
            "everySpreadChanged": sum(1 for r in touched if r["allSpreadsChanged"]),
            "medianShareChanged": shares[len(shares) // 2] if shares else 0.0,
            "spreadsChanged": sum(r["spreadsChanged"] for r in touched),
            "spreadsReplaced": sum(r["spreadsReplaced"] for r in touched),
            "spreadsAdded": sum(r["spreadsAdded"] for r in touched),
            "spreadsRemoved": sum(r["spreadsRemoved"] for r in touched),
            "spreadsRetemplated": sum(r["spreadsRetemplated"] for r in touched),
        },
        "categories": {cat: sum(1 for r in rows if r["category"] == cat) for cat in CATEGORIES},
        "axes": {axis: sum(1 for r in rows if r[axis]) for axis in
                 ("photosChanged", "layoutChanged", "cropsChanged", "textChanged", "otherChanged")},
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return csv_path, json_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cache", default=DEFAULT_CACHE, help=f"local triple cache (default {DEFAULT_CACHE})")
    p.add_argument("--out", default=DEFAULT_OUT, help=f"report path without extension (default {DEFAULT_OUT})")
    p.add_argument("--project", help="one projectId only")
    p.add_argument("--refresh", action="store_true", help="re-download cached triples")
    p.add_argument("--download-only", action="store_true", help="fill the cache and stop")
    p.add_argument("--no-download", action="store_true", help="use the cache as-is, scan nothing")
    p.add_argument("--workers", type=int, default=16, help="parallel downloads (default 16)")
    args = p.parse_args(argv)

    cache_dir = os.path.abspath(args.cache)
    out_base = os.path.abspath(args.out)
    os.chdir(REPO_ROOT)

    if args.no_download:
        triples = []
        for project in sorted(os.listdir(cache_dir)):
            for run_id in sorted(os.listdir(os.path.join(cache_dir, project))):
                triples.append({"projectId": int(project) if project.isdigit() else project,
                                "runId": run_id, "files": {}, **parse_run_id(run_id)})
        print(f"{len(triples)} run(s) in the cache", file=sys.stderr)
    else:
        print("scanning the provenance store for ordered triples ...", file=sys.stderr)
        runs, _ = scan(project=args.project)
        triples = [r for r in runs.values() if r["isTriple"]]
        print(f"{len(triples)} ordered triple(s); caching to {cache_dir}", file=sys.stderr)
        triples, _ = ensure_cache(cache_dir, triples, refresh=args.refresh, workers=args.workers)

    if args.download_only:
        print(f"cache ready: {cache_dir}")
        return 0

    rows, skipped = build_rows(cache_dir, triples)
    rows.sort(key=lambda r: (r["category"], str(r["projectId"])))
    print_report(rows, skipped)
    csv_path, json_path = write_outputs(rows, out_base)
    print(f"\nper-run detail: {csv_path}\nsummary:        {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
