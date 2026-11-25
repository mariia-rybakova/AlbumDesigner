# Running the same request twice

What makes an album reproducible, what still does not, and how to check. Written
in November 2026 after four days of chasing an ordering bug kept running into
runs that disagreed with each other.

## Why it matters more than it sounds

A pipeline that gives a different album each run is not merely untidy. It makes
every other investigation unreliable, because a real regression cannot be told
from noise. During the ceremony-ordering work this produced, in order: a fix
credited to the wrong stage, a "QA says nothing changed" that meant nothing
either way, and a local-versus-production difference chased through three wrong
explanations before the real one. None of that was the ordering code.

So: **before concluding anything from a comparison of two runs, establish that
the two runs are comparable.**

## What is pinned now

**The layout search's sampling.** Whenever a stage generates more candidates
than it may score it samples, and four sites in `src/spreads_layout` used to
draw from the global `random`. They now share one generator seeded from
`CONFIGS['layout_search_seed']`. Before: two runs of one request, same 107
photos, same 18 spreads, different page order. After: identical.

`pipeline.albums` reseeds it per album (`math_tools.reseed`) where several
albums are composed from one request -- identical variants must not produce
identical albums. The variety is deliberate, which is the point; it used to come
from whatever the thread scheduler did.

**The solver's budget.** `select.pick` carries an explicit `random_seed` and a
`max_deterministic_time` -- ortools' own unit of work. The wall clock remains
only as the last resort a queue worker needs. A result short of OPTIMAL is
logged, because that album depends on where the search was cut.

**The dependency set.** `scikit-learn` is pinned like everything else: the time
clustering is a seeded GaussianMixture and the group split a seeded
KMeansConstrained, and a seeded estimator is only reproducible against the
version it was seeded on.

## What is not pinned, deliberately

**`pick_cpsat.workers` is 8.** Eight searches race and the winner depends on the
scheduler, so the picked order is not reproducible *across machines*. Setting it
to 1 is not free: on one wedding it moved the album from 16 spreads to 18 from
the same 107 photos. Changing it changes albums, so it wants its own decision
and its own before/after, not a quiet flip.

**Row order after the readers.** `pd.merge(..., how='outer', on='image_id')`
already returns a key-sorted frame, and selection was stable across runs, so
there was nothing to fix. Left alone rather than sorted on principle.

## The trap that cost the most

**A local run is only comparable to a deployed one if the environment matches,
and for most of 2026 it did not.** The checkout's `.venv` was built in December
2025 on Python 3.10; the pins moved in September 2026 to `pandas==3.0.5`,
`numpy>=2.4.6`, `ortools==9.15.6755`. pandas 3 has no wheel below Python 3.11,
so that venv could not install the file at all and silently resolved to pandas
2.x with ortools 9.10.

A different CP-SAT build breaks ties differently, and *consistently* for a given
build. That is what made local runs reproducible locally, production runs
reproducible in production, and the two different from each other -- on one
request they agreed on barely half the photos, with nothing anywhere reporting a
problem. It reads exactly like "the fix was not deployed", and it is not.

Check `python -V` against the floor at the top of `requirements.txt` before
trusting any local result against a deployed one.

## How to check

**The suite.** `tests/test_reproducible_layout.py` drives the sampling twice and
compares, requires a reseed to still change the answer, and reads the source of
both sampling modules so a fifth site added on the global RNG fails.

**A real request, twice.** The suite cannot cover the pipeline end to end. Run
one request twice and diff the `Pages N-M - left photos ...` lines out of
`servicelog.log`; identical lines mean identical albums. That is how the sampler
was found and how the fix was verified.

**Against production.** The service logs the same `Pages` lines, so a deployed
run can be compared with a local one directly -- see
`.claude/skills/albumdesigner-datadog` for the queries. Match the commit and the
environment first, or the comparison means nothing.
