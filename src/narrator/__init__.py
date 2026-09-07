"""Vendored inference subset of the albumNarrator policy.

**This is a copy, not our code.** Source of truth is the albumNarrator project
(``src/album_narrator``); this holds the 16 modules that composing an album
actually needs, computed by walking imports from ``export.py``:

    config  export  data/{schema,preprocess}  env/{album_env,hero,masking,state}
    eval/baselines  models/{policy,set_encoder}  reward/{reward_fn,terms}

Everything else in that project is training or data plumbing and is deliberately
absent: ``train/``, ``data/mock_generator``, ``data/ingest``, ``eval/metrics``,
``eval/qualitative``, ``cli``. In particular ``data/ingest`` is *not* vendored --
the narrator's own ingest reads per-gallery protobufs from disk, whereas here the
gallery is built from the enriched frame that read + ingest + enrich already
produced. That adapter is ours: ``src/pipeline/select/narrator.py``.

``reward/`` is here because it is part of inference, not just training: composing
takes the best of k sampled rollouts and the reward is what ranks them.

The files are copied verbatim so that a diff against the source project is
meaningful. Only relative imports are used inside, so the package works unchanged
under this name. When the policy is retrained or the reward changes, re-vendor
rather than edit here -- an edit made in this copy is invisible to the project
that trained the weights, and the two would silently disagree about what the
model expects.
"""
