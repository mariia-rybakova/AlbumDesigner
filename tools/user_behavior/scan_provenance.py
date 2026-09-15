"""What the provenance store on ptstorage_120 holds, and which albums were ordered.

The pic-time backend writes one folder per Album Designer call to
`ptstorage_120://pictures/aad-provenance/<projectId>/<runId>/`, containing:

    request.json    what was asked of the designer      (always)
    response.json   what the designer answered          (always)
    ordered.json    the composition as actually ordered (only if ordered)

A folder with three files is therefore a *completed* story: proposal plus
outcome. Those are the only runs from which user behaviour -- what the user kept
and what they changed -- can be read, so finding them is what this script is
for. Two files means the design was produced and not ordered, which is signal
too, just of the negative kind.

    # scan everything, write the index, print the summary
    python tools/user_behavior/scan_provenance.py

    # cheap first look
    python tools/user_behavior/scan_provenance.py --limit 5000

    # one project only
    python tools/user_behavior/scan_provenance.py --project 36025536

    # derive the JSON structure from N ordered triples (downloads them)
    python tools/user_behavior/scan_provenance.py --structure 5

    # keep the ordered triples on disk for analysis
    python tools/user_behavior/scan_provenance.py --ordered-only --download

Run IDs are `aad[m]_<projectId>_<flag>.<YYMMDD-HHMMSS>.<guid>.<n>.<n>`. The
`aadm_` prefix marks a manual (user-initiated) design and `aad_` an automatic
one -- the scan checks that against each request's own `isManual` rather than
trusting the name. The letter after the project id takes four observed values
(`p`, `d`, `e`, `t` -- `p` dominates), and nothing in the data explains what it
selects, so it is recorded as `flag` and left uninterpreted.

Must run from the repo root: ptinfra resolves storage credentials through
`.secrets.yml` in the current directory, so the script chdirs there itself.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STORAGE = "ptstorage_120"
CONTAINER = "pictures"
PREFIX = "aad-provenance/"

#: The three file names the backend writes. `ordered.json` is the one that makes
#: a folder a complete story.
REQUEST = "request.json"
RESPONSE = "response.json"
ORDERED = "ordered.json"
KNOWN_FILES = (REQUEST, RESPONSE, ORDERED)

DEFAULT_OUT = os.path.join("output", "user_behavior", "provenance_index.jsonl")
DEFAULT_DOWNLOAD = os.path.join("output", "user_behavior", "runs")


# --------------------------------------------------------------------------
# blob access
# --------------------------------------------------------------------------

def _container_client():
    """The raw Azure container client for the provenance storage.

    PTFile addresses one blob at a time and `AzureBlobStorage.list_blobs()`
    takes no prefix -- listing the whole `pictures` container to reach one
    folder is not an option at its size. So the connection is built through
    PTFile (which keeps credential resolution in ptinfra, where it belongs) and
    only the listing call reaches past it.
    """
    from ptinfra.azure.pt_file import PTFile

    return PTFile(f"{STORAGE}://{CONTAINER}/{PREFIX}_").blob.container_client


def read_json(blob_path):
    from ptinfra.azure.pt_file import PTFile

    return json.loads(PTFile(f"{STORAGE}://{CONTAINER}/{blob_path}").read_blob())


# --------------------------------------------------------------------------
# run ids
# --------------------------------------------------------------------------

def parse_run_id(run_id):
    """`aadm_36025536_p.260905-150920.<guid>.110.4343` -> its parts.

    Returns dict with `isManualByName`, `flag`, `requestedAtFromId` (UTC, from
    the YYMMDD-HHMMSS stamp). Anything unparseable comes back with None fields
    rather than raising -- an odd name is a finding to report, not a crash.
    """
    out = {"isManualByName": None, "flag": None, "requestedAtFromId": None}
    head, _, rest = run_id.partition(".")
    parts = head.split("_")
    if len(parts) >= 3:
        out["isManualByName"] = parts[0].lower() == "aadm"
        out["flag"] = parts[2]
    stamp = rest.split(".")[0] if rest else ""
    try:
        out["requestedAtFromId"] = datetime.strptime(stamp, "%y%m%d-%H%M%S").replace(
            tzinfo=timezone.utc).isoformat()
    except ValueError:
        pass
    return out


# --------------------------------------------------------------------------
# scan
# --------------------------------------------------------------------------

def scan(project=None, limit=None, since=None, progress_every=20000):
    """Walk the prefix and group blobs into runs.

    Returns (runs, unparsed) where runs maps (projectId, runId) -> record.
    """
    prefix = f"{PREFIX}{project}/" if project else PREFIX
    runs = {}
    unparsed = []
    seen = 0

    for blob in _container_client().list_blobs(name_starts_with=prefix):
        seen += 1
        if progress_every and seen % progress_every == 0:
            print(f"  ... {seen} blobs, {len(runs)} runs", file=sys.stderr)
        if limit and seen > limit:
            print(f"  (stopped at --limit {limit})", file=sys.stderr)
            break

        rel = blob.name[len(PREFIX):]
        parts = rel.split("/")
        if len(parts) != 3:
            # Not <projectId>/<runId>/<file> -- worth surfacing, not skipping silently.
            unparsed.append(blob.name)
            continue
        project_id, run_id, file_name = parts

        key = (project_id, run_id)
        rec = runs.get(key)
        if rec is None:
            rec = runs[key] = {
                "projectId": int(project_id) if project_id.isdigit() else project_id,
                "runId": run_id,
                "files": {},
                **parse_run_id(run_id),
            }
        rec["files"][file_name] = {
            "size": blob.size,
            "modified": blob.last_modified.isoformat() if blob.last_modified else None,
        }

    for rec in runs.values():
        files = rec["files"]
        rec["fileCount"] = len(files)
        rec["hasOrdered"] = ORDERED in files
        rec["designed"] = REQUEST in files and RESPONSE in files
        # The analyzable set. Not every folder holding an ordered.json is one:
        # orders placed against a design made before this store existed keep the
        # outcome without the proposal, and those cannot be compared.
        rec["isTriple"] = rec["designed"] and rec["hasOrdered"]
        rec["unexpectedFiles"] = sorted(set(files) - set(KNOWN_FILES))

    if since:
        cutoff = since.isoformat()
        runs = {k: r for k, r in runs.items()
                if (r["requestedAtFromId"] or "") >= cutoff}

    return runs, unparsed


def summarize(runs, unparsed):
    total = len(runs)
    triples = [r for r in runs.values() if r["isTriple"]]
    orphan_orders = [r for r in runs.values() if r["hasOrdered"] and not r["designed"]]
    by_count = Counter(r["fileCount"] for r in runs.values())
    by_flag = Counter(r["flag"] for r in runs.values())
    manual = Counter(r["isManualByName"] for r in runs.values())
    per_project = Counter(r["projectId"] for r in runs.values())
    stamps = sorted(r["requestedAtFromId"] for r in runs.values() if r["requestedAtFromId"])

    print(f"\nruns:                 {total}")
    print(f"projects:             {len(per_project)}")
    if stamps:
        print(f"run id timestamps:    {stamps[0]}  ..  {stamps[-1]}")
    print(f"\nfiles per run:        " + ", ".join(
        f"{n} file{'s' if n != 1 else ''}: {c}" for n, c in sorted(by_count.items())))
    print(f"ORDERED TRIPLES:      {len(triples)}"
          + (f"  ({100.0 * len(triples) / total:.1f}% of runs)" if total else "")
          + "   <- request + response + ordered, the analyzable set")
    print(f"manual (aadm_):       {manual.get(True, 0)}   auto (aad_): {manual.get(False, 0)}")
    print(f"flag letter:          " + ", ".join(f"{k}: {v}" for k, v in by_flag.most_common()))

    print(f"projects with a triple: {len({r['projectId'] for r in triples})}")

    if orphan_orders:
        # An order whose design is not in the store: the proposal is unknown, so
        # there is nothing to diff the outcome against.
        print(f"\n   {len(orphan_orders)} run(s) have ordered.json but no design "
              f"(excluded from triples), e.g.:")
        for r in orphan_orders[:3]:
            print(f"   {r['projectId']}/{r['runId']} -> {sorted(r['files'])}")

    stranded = [r for r in runs.values() if not r["designed"] and not r["hasOrdered"]]
    if stranded:
        print(f"\n!! {len(stranded)} run(s) with neither a full design nor an order, e.g.:")
        for r in stranded[:3]:
            print(f"   {r['projectId']}/{r['runId']} -> {sorted(r['files'])}")

    odd = [r for r in runs.values() if r["unexpectedFiles"]]
    if odd:
        names = Counter(n for r in odd for n in r["unexpectedFiles"])
        print(f"\n!! file names beyond the three known ones: {dict(names)}")

    if unparsed:
        print(f"\n!! {len(unparsed)} blob(s) not shaped <projectId>/<runId>/<file>, e.g.:")
        for name in unparsed[:5]:
            print(f"   {name}")

    if triples:
        print("\nordered triples (first 10):")
        for r in sorted(triples, key=lambda r: r["requestedAtFromId"] or "")[:10]:
            print(f"   {r['projectId']}/{r['runId']}")

    return triples


# --------------------------------------------------------------------------
# structure inference
# --------------------------------------------------------------------------

def observe(node, schema, path="", max_depth=6, depth=0):
    """Fold one JSON document into `schema`: path -> types, sizes, an example."""
    entry = schema.setdefault(path or "<root>", {
        "types": Counter(), "arrayLens": [], "keyCount": [], "example": None})
    entry["types"][type(node).__name__] += 1

    if isinstance(node, dict):
        entry["keyCount"].append(len(node))
        if depth < max_depth:
            # `designCatalog.designs` and friends are maps keyed by id, not
            # records with named fields. Folding every id onto one `{id}` path
            # keeps the schema the size of the shape instead of the size of the
            # catalog -- unfolded, one request printed thousands of lines.
            keyed_by_id = bool(node) and all(k.lstrip("-").isdigit() for k in node)
            for k, v in node.items():
                child = "{id}" if keyed_by_id else k
                observe(v, schema, f"{path}.{child}" if path else child, max_depth, depth + 1)
    elif isinstance(node, list):
        entry["arrayLens"].append(len(node))
        if node and depth < max_depth:
            # Elements are homogeneous in this store; fold them all into one
            # path so a field that is null in the first element but set in the
            # tenth still shows up.
            for item in node[:50]:
                observe(item, schema, f"{path}[]", max_depth, depth + 1)
    elif entry["example"] is None and node is not None:
        s = repr(node)
        entry["example"] = s[:80] + "..." if len(s) > 80 else s
    return schema


def print_schema(title, schema):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    for path in sorted(schema):
        e = schema[path]
        types = "|".join(t for t, _ in e["types"].most_common())
        bits = [f"{path}: {types}"]
        if e["arrayLens"]:
            lens = e["arrayLens"]
            bits.append(f"len {min(lens)}..{max(lens)}")
        if e["example"] is not None:
            bits.append(f"e.g. {e['example']}")
        print("  " + "  ".join(bits))


def inspect_structure(ordered_runs, sample, download_dir=None):
    """Download `sample` ordered triples, infer each file's structure, and check
    the claims the run id makes against the payload."""
    schemas = {name: {} for name in KNOWN_FILES}
    mismatches = []
    changed = Counter()
    picked = sorted(ordered_runs, key=lambda r: r["requestedAtFromId"] or "", reverse=True)[:sample]

    for rec in picked:
        folder = f"{PREFIX}{rec['projectId']}/{rec['runId']}"
        print(f"reading {folder}", file=sys.stderr)
        docs = {}
        for name in KNOWN_FILES:
            if name not in rec["files"]:
                continue
            try:
                docs[name] = read_json(f"{folder}/{name}")
            except Exception as exc:                       # noqa: BLE001 - report, keep going
                print(f"  !! {name}: {exc}", file=sys.stderr)
                continue
            observe(docs[name], schemas[name])
            if download_dir:
                dest = os.path.join(download_dir, str(rec["projectId"]), rec["runId"])
                os.makedirs(dest, exist_ok=True)
                with open(os.path.join(dest, name), "w", encoding="utf-8") as fh:
                    json.dump(docs[name], fh, indent=2)

        req, ordered_doc = docs.get(REQUEST), docs.get(ORDERED)
        if req is not None and rec["isManualByName"] is not None:
            if bool(req.get("isManual")) != rec["isManualByName"]:
                mismatches.append((rec["runId"], req.get("isManual")))
        if ordered_doc is not None:
            changed[ordered_doc.get("changed")] += 1

    for name in KNOWN_FILES:
        if schemas[name]:
            print_schema(f"{name}  (from {len(picked)} ordered run(s))", schemas[name])

    print(f"\nordered.json `changed` across the sample: {dict(changed)}"
          "   # True = the user altered the AI design before ordering")
    if mismatches:
        print(f"!! aadm_/aad_ prefix disagreed with request.isManual: {mismatches}")
    else:
        print("aadm_/aad_ prefix matched request.isManual in every sampled run.")
    if download_dir:
        print(f"triples written to {download_dir}")


# --------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--project", help="scan one projectId folder only")
    p.add_argument("--limit", type=int, help="stop after N blobs (quick look)")
    p.add_argument("--since", type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc),
                   help="keep runs whose run-id timestamp is on/after YYYY-MM-DD")
    p.add_argument("--ordered-only", action="store_true",
                   help="write only the ordered triples to the index")
    p.add_argument("--structure", type=int, metavar="N", default=0,
                   help="download N ordered triples and print the inferred JSON structure")
    p.add_argument("--download", nargs="?", const=DEFAULT_DOWNLOAD, default=None,
                   metavar="DIR", help=f"save downloaded triples under DIR (default {DEFAULT_DOWNLOAD})")
    p.add_argument("--out", default=DEFAULT_OUT, help=f"index path (default {DEFAULT_OUT})")
    p.add_argument("--no-index", action="store_true", help="print the summary, write nothing")
    args = p.parse_args(argv)

    # Resolve paths against the caller's cwd before moving to the repo root,
    # which ptinfra needs for `.secrets.yml`.
    out_path = os.path.abspath(args.out)
    download_dir = os.path.abspath(args.download) if args.download else None
    os.chdir(REPO_ROOT)

    print(f"scanning {STORAGE}://{CONTAINER}/{PREFIX}"
          + (f"{args.project}/" if args.project else ""), file=sys.stderr)
    runs, unparsed = scan(project=args.project, limit=args.limit, since=args.since)
    triples = summarize(runs, unparsed)

    if not args.no_index:
        records = triples if args.ordered_only else list(runs.values())
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            for rec in sorted(records, key=lambda r: (str(r["projectId"]), r["runId"])):
                fh.write(json.dumps(rec) + "\n")
        print(f"\nindex: {len(records)} run(s) -> {out_path}")

    if args.structure or download_dir:
        if not triples:
            print("\nno ordered triples in this scan -- nothing to inspect.")
        else:
            inspect_structure(triples, args.structure or len(triples), download_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
