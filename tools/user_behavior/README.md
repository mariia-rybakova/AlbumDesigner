# User behaviour data

Scripts that read what users actually *did* with what the Album Designer
produced — as opposed to the pipeline's own view of a run, which lives in
`utils/stages_recorder` and the Datadog logs.

| script | what it does |
|---|---|
| `scan_provenance.py` | inventories the provenance store, finds the ordered albums, and infers the JSON structure of the files |
| `diff_report.py` | caches the ordered triples locally and reports what users changed between the proposed album and the ordered one |

## The provenance store

The pic-time backend writes one folder per Album Designer call to

```
ptstorage_120://pictures/aad-provenance/<projectId>/<runId>/
    request.json     what was asked of the designer        (always)
    response.json    what the designer answered            (always)
    ordered.json     the composition as actually ordered   (only if ordered)
```

A three-file folder is a complete story — proposal *and* outcome — and is the
only shape from which behaviour can be read. Reading it needs the VPN and a
`.secrets.yml` in the repo root (ptinfra resolves the storage key through the
config service); the script chdirs to the repo root itself.

`runId` is `aad[m]_<projectId>_<flag>.<YYMMDD-HHMMSS>.<guid>.<n>.<n>`:
`aadm_` is a manual (user-initiated) design and `aad_` an automatic one, which
matched `request.isManual` in every run sampled. The timestamp is UTC and equals
`requestedAt` to the second. `flag` takes four values (`p` 92%, then `d`, `e`,
`t`); nothing in the data says what it selects, so it is recorded and not
interpreted.

## What each file holds

**`request.json`** — 14 keys: `requestId`, `isManual`, `storeId`, `accountId`,
`projectId`, `productId`, `photoIds[]` (the candidate pool, up to a few
thousand), `designIds[]`, `minPages`/`maxPages`, `requestedCompositionsCount`,
`requestedAt`, plus two that are set on automatic runs and null on manual ones:

- `aiMetadata` — `{photoIds[] (the AI's own picks), focus[] (e.g. "brideAndGroom"),
  personIds[], subjects[], density}`
- `designCatalog` — `{productId, title, designs{<designId>: {boxes[]}},
  parts{cover, anyPage, …}, defaultPackageStyleId, minPages, maxPages}`.
  This is what makes an automatic request ~500 KB against a manual one's ~1 KB.

**`response.json`** — `requestId`, `isManual`, `respondedAt`, and two
compositions-package objects with identical 29-key shape:

- `rawAiReply` — the designer's answer as returned (null on manual runs)
- `enriched` — the same after the backend filled in box geometry and styles;
  **this is the one to compare against the order**

Inside either: `compositions[]` (one per spread, with `designId`, `boxes[]`,
`tags[]` like `des-1714800`/`part-cover`), `placementsImg[]` (one per placed
photo: `photoId`, `compositionId`, `boxId`, `cropX/Y/Width/Height`, `rotate`,
`photoFilter`), `placementsTxt[]`, and `specialOptions.aadRequestId`.

**`ordered.json`** — `requestId`, `compositionPackageId`, `userJobId`,
`orderedDate`, **`changed`** (bool), and `orderedComposition` in the same 29-key
shape as `enriched`. `changed` is the headline behaviour signal: it was `True`
in 16 of 20 sampled orders, i.e. most users edit before ordering. The real
per-album diff comes from comparing `response.enriched` with
`ordered.orderedComposition`. Note the order carries real ids where the proposal
has `-1` placeholders (`compositionPackageId`), so diffs must key on
`photoId`/`boxId`/`compositionId`, never on the package id.

Checked on 16 downloaded triples, every edit type is present and `changed` is
trustworthy — the two `changed: False` orders were byte-identical to the
proposal in all four axes below, and no `changed: True` order was:

| edit | how to read it | seen in the 16 |
|---|---|---|
| photos dropped / added | set difference on `placementsImg[].photoId` | 12 runs; from ±1 photo to 180→47 |
| photo re-cropped | same `(photoId, compositionId, boxId)`, different `cropX/Y/Width/Height` | 8 runs, 2–16 photos each |
| spreads re-laid-out | `compositions[].designId` sequence differs | 12 runs; 4 also changed the spread count |
| text entered | `placementsTxt[].textToRender` | all 16 — `placementsTxt` is **empty in every proposal** and has 1–2 entries in every order |

That last row is why `changed` is `True` far more often than the design was
really reworked: typing the cover name alone flips it. Any "did the user accept
the AI's design?" measure should diff the photo and layout axes, not read
`changed`.

## Store contents as of 2026-09-15

```
runs:               4720 across 2789 projects
retention:          2026-08-18 .. 2026-09-15 (≈4 weeks, so re-scan before relying on a run)
files per run:      1 file: 72, 2 files: 4247, 3 files: 401
ORDERED TRIPLES:    401  (8.5% of runs), across 388 projects
manual (aadm_):     3103    auto (aad_): 1617
```

Two edge cases the scan reports rather than hides: 42 folders hold an
`ordered.json` with no design (an order placed against a design made before this
store existed — outcome without proposal, so not comparable), and 31 hold a
`request.json` alone (a call that never produced a response).

## Usage

```bash
python tools/user_behavior/scan_provenance.py                  # full scan + index
python tools/user_behavior/scan_provenance.py --limit 5000     # quick look
python tools/user_behavior/scan_provenance.py --project 36025536
python tools/user_behavior/scan_provenance.py --structure 12   # re-derive the schema above
python tools/user_behavior/scan_provenance.py --ordered-only --download
```

## What users changed (`diff_report.py`)

```bash
python tools/user_behavior/diff_report.py                 # cache, diff, report
python tools/user_behavior/diff_report.py --no-download    # re-report from the cache
python tools/user_behavior/diff_report.py --download-only   # just fill the cache
```

It caches all 401 triples locally (16 parallel downloads, ~1 min) and diffs
`response.enriched` against `ordered.orderedComposition` on four axes — photos,
layout, crops, text — then drops each run into the highest bucket it reaches.
A crop is only counted for a photo present on both sides: a swapped-in photo
brings its own crop, and counting that as a re-crop would charge one edit twice.

### Telling a user edit from bookkeeping

Ordering rewrites a great deal on its own, so a naive diff calls every run
changed. The 65 orders the backend itself marked `changed: False` settle it —
whatever moves in those moved without a user: placeholder ids becoming real,
`status`, `origin`, `ptPlatform`, `alerts`, package `tags` (65/65), the
per-placement `photo` object (null in every proposal, hydrated in every order),
composition `tags` (they just mirror `designId`), and composition `styleId`
(only ever `0 ↔ -1`). Two traps beyond the field list:

- **Spreads get renumbered.** 27 of the 401 orders start `compositionId` at 1
  where the proposal started at 0. Comparing by id made every spread look
  re-templated and every photo look moved; everything per-spread is therefore
  keyed by *position*, which both sides agree on.
- **An inserted spread shifts every later position.** Counting position by
  position would call a 30-spread album fully re-laid-out because the user added
  one page, so the spread count uses a `difflib` alignment: one insert, the rest
  equal.

With those handled, the four axes account for **all** 401 orders — `changed
more` is empty, and an audit of every remaining field on kept placements and
aligned spreads turned up nothing but the bookkeeping above.

### Results (401 triples, 2026-08-18 .. 09-15)

| category | runs | share |
|---|---:|---:|
| no change | 3 | 0.7% |
| only text | 62 | 15.5% |
| only crops | 4 | 1.0% |
| changed photos | 13 | 3.2% |
| changed layout | 57 | 14.2% |
| **changed photos and layouts** | **262** | **65.3%** |
| changed more | 0 | 0.0% |

Axes, which overlap: photo set 68.6%, layout 79.6%, re-crop of a kept photo
74.6%, text 96.0%. Of the proposed photos 84.9% survive to the order, and users
add 5493 of their own.

**How much of the album changed**, over the 332 runs with at least one changed
spread. A spread counts as changed when its template differs, a photo on it was
replaced, or a photo moved to a different frame — re-cropping a photo or typing
a caption leaves the spread as the designer composed it, so neither alone moves
a spread into this count:

| spreads changed | runs | share |
|---|---:|---:|
| 1 spread | 19 | 5.7% |
| 2–3 spreads | 34 | 10.2% |
| 4–10 spreads | 80 | 24.1% |
| 11–25 spreads | 139 | 41.9% |
| 26+ spreads | 60 | 18.1% |

The median edit touches **81% of the album**, and in 20 runs (5% of all orders)
no spread survives untouched. In total 5153 spreads changed — 4931 reworked in
place, 143 added, 79 removed — of which 3469 took a different design template
and the rest changed by photo alone.

Both counts come from a `difflib` alignment of the spread sequences, scored
against the alignment's own span rather than the longer album: an alignment can
charge a delete and an insert at the same position, and measuring against
`max(len(a), len(b))` reported 45 changed spreads in a 37-spread album, calling
albums fully re-laid-out that had kept half their spreads.

`changed layout` means the photo *selection* held: nothing was added or dropped
in any of those 57 runs. It does not mean layout alone — the lower rungs ride
along, with 53 of the 57 also entering text, 40 also re-cropping a kept photo.
The layout edit itself is a re-template in 44, a spread added or removed in 26,
and in 5 it is purely photos shuffled between the frames of unchanged spreads.
Note the category axis stays template-level on purpose: if a photo swapped into
a spread also counted as a layout change there, `changed photos` would empty
into `changed photos and layouts`. The per-spread magnitude table above uses the
wider definition; the categories do not.
The photo axis compares *distinct* `photoId`s, so repeating a photo on another
spread (or dropping a repeat) counts as layout, not selection: that happens in 7
runs, visible as `placementsProposed` != `placementsOrdered`.

Two caveats worth carrying into any conclusion drawn from this. A layout change
that accompanies a photo change may be the product re-flowing pages rather than
a second decision by the user, so `changed photos and layouts` is not proof of
two independent rejections. And `changed: False` disagrees with the diff on 62
runs — all of them text-only, because the backend's flag ignores text.

### Outputs

`output/user_behavior/diff_report.csv` has one row per run with the category,
the four axis booleans, and the magnitudes: `photosAdded/Removed/Recropped/Moved`,
`spreadsChanged/Unchanged/Replaced/Added/Removed/Retemplated`,
`spreadsChangedShare`, `allSpreadsChanged`, `spreadsRearranged`,
`placementsProposed/Ordered`, plus `boxGeometryChanged` and
`productChanged` (reported but deliberately not category-driving: a resized frame
fired on a control run, and the product choice is an order-time decision rather
than a design edit). `diff_report_summary.json` holds the totals above.

## The scan index

The index is one JSON object per run at
`output/user_behavior/provenance_index.jsonl` (gitignored), carrying
`projectId`, `runId`, `requestedAtFromId`, `isManualByName`, `flag`, per-file
size and modified time, and the `isTriple` / `hasOrdered` / `designed` flags —
enough to pick a working set without listing the store again. `--download`
writes the triples under `output/user_behavior/runs/<projectId>/<runId>/`.
