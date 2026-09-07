---
name: albumdesigner-datadog
description: How to query Datadog logs for the AlbumDesigner service - the tag that isolates it from the other pic-time AI services, the per-stage filter, the log vocabulary each stage emits, and ready-made queries for finding a successful or failed run, recovering a request payload, tracing one project end to end, and checking selection or spread behaviour. Use whenever investigating AlbumDesigner in production, reproducing a real request locally, or answering "what happened to project X".
---

# AlbumDesigner in Datadog

All queries verified against the live org (site `us3.datadoghq.com`) on 2026-09-01.

## The one thing to get right: `source:albumdesigner`

Every pic-time AI service logs under the **same** `service:ai` field, so
`service:ai` alone returns BgSegmentation, ContextCluster, ImageEmbedding and
everything else. Filtering by message text to compensate is slow and lossy.

**Always start from `source:albumdesigner`.**

```
source:albumdesigner env:production
```

Other reliable tags on the same logs:

| Tag | Value |
|-----|-------|
| `source` | `albumdesigner` |
| `service` (tag) | `albumdesigner` — note the `service` *field* is `ai` |
| `kube_deployment` | `album-designer-production-deployment` |
| `kube_namespace` | `default` |
| `image_tag` | e.g. `album-designer-260818-v585-code_freeze_hotfixes` — use to pin a release |
| `env` | `production` |
| `cluster_name` | `kubernetes-ai` |

Attributes (all under `custom.` in the response; query with `@`):

| Attribute | Query as | Value |
|-----------|----------|-------|
| `custom.threadname` | `@threadname` | the stage — see below |
| `custom.level` | `@level` | `INFO `, `DEBUG`, `ERROR` (note trailing spaces on some) |
| `custom.name` | `@name` | logger name, usually `__main__` |

## Filtering to one stage

`@threadname` carries the ptinfra worker name, which is the cleanest way to
scope to a stage:

```
source:albumdesigner @threadname:ReadStage*
source:albumdesigner @threadname:SelectionStage*
source:albumdesigner @threadname:ProcessingStage*
source:albumdesigner @threadname:ReportMessage*
```

(The workers are `<StageName>.Worker#0`, hence the trailing `*`.)

Grouping patterns by `@threadname` via `pattern_group_by` came back empty in
testing — use it as a filter, not a group-by.

## The two anchor log lines

These are the ones worth memorising; everything else hangs off them.

**Request received** — `ReadStage`, INFO. Carries the entire request payload.

```
Received message: {'replyQueueName': ..., 'projectId': 53496523, 'base_url': 'ptstorage_17://...',
 'photos': [...], 'aiMetadata': {...}, 'rating': [...], 'conditionId': 'AAD_53496523_P....', ...}
 /<ptinfra.pt_queue.Message object at 0x...>
```

The payload is a **Python repr, not JSON** (single quotes, `None`, `False`) —
parse with `ast.literal_eval`, never `json.loads`. Strip the
`Received message: ` prefix and the trailing `/<... object at 0x...>` first.

> **Large galleries are truncated.** Datadog caps log entry size; a gallery with
> thousands of photo ids and ratings overruns it and the payload will not parse.
> Fall back to another run rather than trying to repair it.

**Request succeeded** — `ReportMessage`, DEBUG. Emitted **only** on the success
path (`report_one_message`'s else-branch), so it is the definitive
"this produced an album" marker:

```
Message was reported to the queue: 53496523/AAD_53496523_P.260901-092002.71586c50-....176.967.
```

Format is `<projectId>/<conditionId>`. The failure counterpart is
`REPORT ERROR MESSAGE  <error>`.

## Ready-made queries

**Newest successful runs**

```
source:albumdesigner "Message was reported to the queue"
```
sort `-timestamp`. Gives projectId + conditionId.

**The payload for a specific run** — search a window *before* its report, and
match the conditionId, because a project is often re-run several times:

```
source:albumdesigner "Received message" "53496523"
```
from = report time − 90 min, to = report time.

**Everything for one project, in order** (the end-to-end trace):

```
source:albumdesigner "53496523"
```
sort `timestamp` ascending.

**Failures**

```
source:albumdesigner status:error
source:albumdesigner "REPORT ERROR MESSAGE"
```

**Stage timings**

```
source:albumdesigner ("READING Stage for" OR "Selection Stage for" OR "Processing Stage time")
```

**Selection behaviour** (the fallbacks firing is the interesting signal)

```
source:albumdesigner ("There are no images to select for" OR "no filtering" OR "We took out more than 80%")
source:albumdesigner "Total images:"
```

**Artificial-time galleries** (EXIF unusable, scene order used instead)

```
source:albumdesigner "Time info is not correct"
```

**Pin to a release**

```
source:albumdesigner image_tag:album-designer-260818-v585-code_freeze_hotfixes status:error
```

## Log vocabulary by stage

Useful for building filters; taken from the running service.

**ReadStage** — `Received message: {...}`, `Fetch the document from the
collection ...`, `doc found for project_id X: {'imageModelVersion': 2,
'isInVectorDatabase': True}`, `Project X has isInVectorDB = True, loading Clip
embeddings from qdrant`, `Start fetching vectors from Qdrant collection
ImageEmbedding_V2_new for projectId X`, `Fetched N vectors from Qdrant`,
`Reading Files protos for N images is: ...`, `Dropped N rows because embedding
is NaN` (warn), `Number of images before/after cleaning the nan values: N`,
`Merged user_rating into gallery_info_df: N/M photos have a rating`, `Finshed
with the photo scene` (sic), `Time info is not correct. Using artificial time.`
(warn), `Ceremony SAT: start=... end=... count=N`, `Updated N images to 'bride
and groom with parents' based on 4-person couples.`, `READING Stage for N
messages. Average time: ...`, `Reading Time Stage for one Gallery N images is:
...`

**SelectionStage** — `aiMetadata not found for message ... Continue with chosen
photos.` (the manual path), `There are no images to select for <category>`, `it
has less than needed so we select them all <category> no filtering`, `We took
out more than 80% from this cluster <category> so we get N images back from
filtering`, `No color candidates for <category>, selected N grayscale images
only`, `Total images: N`, `Selection Stage for N messages. Average time: ...`

**ProcessingStage** — `Params for this Gallery are: [...]`, `Illegal groups
processing time: Ns`, `General groups processing time: Ns`, `Condition we
split!. Using splitting to N parts`, `No singletons found to resolve`,
`Final number of groups for the album: N`, `Processed group name (t, 'ctx', i)
in Ns`, `Spread created using dummy photo for group: X`, `Pages ... - left
photos (n): [...], right photos (m): [...]`, `Added Album Cover composition
with id N`, `Added Album any page`, `Added dummy firstPage/lastPage section
with N single-box layout(s) from anyPage`, `waited for cropping process: ...`,
`Lay-outing time: ...`, `Processing Stage time: ...`

**ReportMessage** — `Message was reported to the queue: <pid>/<conditionId>`,
`REPORT ERROR MESSAGE <error>`, `deleting message id X`, `placementImg length
N`, `final_result placementImg length N`

## Known recurring failures

| Pattern | Meaning |
|---------|---------|
| `Error processing stage: cropping process not completed: . Exception in function: process_message, line 333, file /usr/app/main.py.` | The cropping subprocess missed its 200 s deadline. Paired with an `Exception raised in stage: ... in stage ProcessingStage.Worker#0`. |
| `REPORT ERROR MESSAGE  Image not found: https://.../smallres/...` | A photo in the request is gone from blob storage. High volume; usually benign. |
| `REPORT ERROR MESSAGE  Error of loading ..., file: ptstorage_N://...` | A gallery blob could not be read. |
| `Qdrant batch async batch search failed ... Received message larger than max` | Not AlbumDesigner — that is `service:serilog`. Do not chase it from here. |

## Reproducing a run locally

`process_gallery.py --from-datadog` automates the whole loop — pair the two
anchor lines, save the payload to `files/test_requests/<projectId>.json`,
download the gallery's photos, run the pipeline. See `tools/local_request.py`.

```
python process_gallery.py <input_dir> <output_dir> --from-datadog
python process_gallery.py <input_dir> <output_dir> --from-datadog --project-id 53496523
python process_gallery.py <input_dir> <output_dir> --request 53496523   # replay, no Datadog
```

**A request is only worth replaying if `aiMetadata.photoIds` is not null.** A
null one means the user assembled the album by hand: `AiHints.from_request`
reads it as `present=False`, `select.route` marks the message manual, and the
budget, preselect, pick and publish substages all skip. The whole request photo
list goes to layouting unchanged. That is deliberate -- but from the outside it
is indistinguishable from a catastrophic selection bug, and the log says
nothing: 52755795 produced a 42-page album of 346 photos, identically under
both pickers, with no "Photos selected" or "Spreads dict sum" line anywhere
because selection never ran. It finished in 18 seconds.
`find_latest_successful_request` skips these by default (`require_ai=True`); the
newest *successful* run is often one of them.

It needs `DD_API_KEY` + `DD_APP_KEY` (`DD_SITE` defaults to `us3.datadoghq.com`)
for the log lookup and the Azure network for the photos. The MCP tools need
neither — when working interactively, prefer fetching the payload over MCP and
writing it to `files/test_requests/`.

**Caveat worth knowing before promising a local repro:** requests carry
`'designInfo': None` plus a `designInfoTempLocation` pointing at
`pictures/temp/queues/aigeneratealbumdto/<hash>.json`. That is a queue temp
blob and may be purged after processing, in which case the read stage fails at
`ingest.design` and the run cannot be reproduced. Check the blob exists before
committing to a repro of an older request.

## Gotchas

- `service:ai` is shared; `source:albumdesigner` is not. Never filter by
  `service` alone.
- The success marker is at **DEBUG** level — do not add `status:info`.
- `Received message` payloads are Python reprs and are truncated for large
  galleries.
- A projectId appears many times: galleries get re-run. Always disambiguate by
  `conditionId`.
- For counting or aggregation use `analyze_datadog_logs`, and load the
  `datadog/ddsql` skill first — `search_datadog_logs` is for raw entries and
  patterns only.
