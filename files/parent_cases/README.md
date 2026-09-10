# Parent-identification cases

Galleries where `enrich.parents` gets it wrong, or gets it right for a reason
worth keeping. One JSON per gallery, named by `projectId`.

These exist so a **new approach can be judged before it ships**. The current
scorer was tuned on 47981912 and 53459898, and the results table in
`docs/parent_identification.md` records only what it named — not who it should
have named, and not who it must never name. A case file records all three.

## Match people by photo, never by identity id

Identity ids are **not stable**. Gallery 49995684 was reprocessed on
2026-09-10 and every id changed: the bride went 1 → 2, the groom 7 → 13, her
father 27 → 17. The reprocess also populated social circles for the first time
(0 → 22) and cut identity mentions by 9%, which is what finally resolved the
groom's mother — so re-reading a case gallery is expected to move the numbers.

Each labelled person therefore carries `anchorPhotoIds`: photos holding that
person and as few others as possible, chosen so the label survives renumbering.
`identityIdAtCapture` is recorded only to reproduce the capture, and is stale
the moment the gallery is reprocessed.

The same instability is why a saved request's `aiMetadata.personIds` cannot be
trusted across a reprocess — on 49995684 the three named people became ids 3,
90 and 48 while 58/100/71 went on existing as *different* people.

## What a case holds

| field | why |
|---|---|
| `request` | the fixture that reproduces the run |
| `couple` | bride and groom, with anchors |
| `people[]` | everyone whose outcome the case asserts |
| `people[].expected` | `named`, `not a parent`, `never a parent`, or `named -- CURRENTLY MISSED` |
| `people[].labelSource` | how we know. `UNVERIFIED inference` means nobody checked the photos |
| `people[].indicators` | the measured evidence, so an approach can be scored offline without re-reading the gallery |
| `outcomeAtCapture` | what the pipeline actually did, and why |

Both cases turn on the same thing: **the face-age estimate underreads older
women**, by enough to matter at a hard gate. 49995684 loses one at rank 0.36
and 53227528 keeps one at 0.72. A case set with only the failure in it would
make lowering `min_age_rank` look free.

**`labelSource` is load-bearing.** A case is only as good as its ground truth,
and some of 49995684's labels are inference rather than a person looking at the
photos. Treat an `UNVERIFIED` negative as "not yet ruled out", not as a target
to optimise against.

## Cases

| gallery | status | the interesting part |
|---|---|---|
| [49995684](49995684.json) | groom's father missed | He is separated from another old man by **0.001** (0.746 vs 0.747), inside the 0.10 margin, so the side declines. Raising him is not enough — an approach has to separate the two. The gallery also contains the trap: the identity photographed with the groom *more than anyone* (21 frames) is not his mother and has no family portrait with him at all, so anything frequency-led names her wrongly. |
| [53227528](53227528.json) | bride's side correct, by 0.02 | A true positive that nearly was not. Her mother is named on an age rank of **0.72** against a 0.70 floor, because the face model reads her as 45 and she is visibly a generation older. The same underestimate is what loses a mother on 49995684, so this case is what stops a fix there from being bought by lowering the gate. Its own trap is the brother: 19 years old and the most-photographed non-couple identity in the gallery. |

## Adding one

Capture the indicators from a real read rather than by hand — `measure()`
returns them, and `scratchpad/diag_parents.py` prints the table. Record what
the pipeline did at the time in `outcomeAtCapture` so a later change can be
diffed against it, and be explicit in `labelSource` about which labels a human
actually verified.
