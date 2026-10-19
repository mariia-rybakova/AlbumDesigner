# AlbumDesigner

A service that lays out a photo album: it reads a gallery, chooses the photos,
groups them, and packs each group onto spreads against the printlab's designs.
`main.py` runs the four queue stages; `src/pipeline` holds the substages the
read and selection stages are composed from.

## Packages that are trees

Two packages are **trees with a single entrance**, not collections of helpers.
Each has its own `CLAUDE.md` with the detail — read it before adding a file
there:

| Package | The only way in |
|---|---|
| `src/spreads_layout/` | `main.process_group` |
| `src/groups_operations/` | `groups_management.process_wedding_illegal_groups` |

**If code would be imported from outside one of those packages, it does not
belong inside it.** Put it in `src/core/` and call it from the tree.

Re-exporting an internal through an `__init__.py` is not a fix: the dependency
is unchanged, only the import path looks tidier, and the tree stops being one
while still looking like it.

Both packages have known exceptions that predate the rule, listed in their own
`CLAUDE.md`. They are not precedents.

## Where things go

| Package | What belongs there |
|---|---|
| `src/core/` | Domain vocabulary and constructors, imported from anywhere: `models`, `photos`, `group_formation`, `time_processing`, `spread_ordering` |
| `src/covers/` | The album's first and last pages — `scoring` ranks a photograph as a cover, `choosing` picks the two, `pages` lays them out. A shelf, not a tree, and it depends in one direction only |
| `src/readers/` | Libraries that turn an outside format into the album's tables — protobuf blobs, the design JSON, the query bank. Not substages; the substages that call them live in `src/pipeline/ingest` |
| `src/pipeline/` | The substage pipeline — `ingest` reads and organises but never infers; `enrich` infers; `select` chooses the photos and the spread budget |
| `src/groups_operations/`, `src/spreads_layout/` | The two trees above |
| `src/selection/` | The selection algorithms, shared by the old monolith and the new pipeline. `src/pipeline/select` may import from here; the reverse would couple the frozen path to the moving one |
| `src/stages_recorder/` | Instrumentation for the analysis PDFs. Renders in `stages_visualizer/`, which reads only the JSON on disk and imports no pipeline code |
| `utils/` | Being dissolved. Nothing new goes here |

`utils/` was a second `src` — domain code filed under a name that promised
generic helpers. Three things are left in it and each is there for a reason:
`configs.py` (the config, edited constantly, moving it would conflict with
everything), `lookup_table_tools.py`, and `selection/`. Add nothing to it.

## Tests

`pytest tests -q` — 798 passing. Tests may import any module they need; the
tree rule above constrains production code only.
