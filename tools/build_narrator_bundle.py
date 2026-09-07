"""Assemble the narrator model bundle the Docker image ADDs at build time.

albumNarrator is a RESEARCH repo: it trains the policy and owns the reward, and
is never a runtime dependency of this service. This script is the one-way
hand-off -- it copies the trained artefacts out of a research checkout into the
flat layout `CONFIGS['narrator']` expects, and copies NO code. (The code side
of the hand-off is `src/narrator/`, vendored separately.)

    python tools/build_narrator_bundle.py \
        --narrator-root "C:/Users/ZivRotman/PycharmProjects/albumNarrator" \
        --run pod_v29 --out narrator_models_v1.zip --verify

Then upload the zip to the `ai-models` blob container under `album-narrator/`
and put its read-only SAS URL in the Dockerfile `ADD` line. That upload is a
deliberate, credentialed step: every SAS in every service Dockerfile is `sp=r`,
so publishing needs a write credential this repo does not carry.

Why a script rather than a hand-made zip: the two files must agree with each
other and with the code. The checkpoint carries the training `cfg` that
`ActorCritic` is rebuilt from, so a policy whose `clip_dim` is not 768 cannot
consume this service's V2 embeddings, and the axes file must have one row per
attribute the adapter asks for. Both are silent failures -- a mismatched
checkpoint loads fine and composes worse albums, and a short axes file leaves
those features at a neutral 0.5 -- so they are checked here rather than
discovered in production. exemplarSelection shipped the wrong checkpoint once
for exactly this reason.

The zip's members are flat filenames, matching `bib-detection`'s bundle: the
image unzips into `files/narrator/`, so a member must be `policy.pt`, not
`narrator/policy.pt`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import sys
import zipfile

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

#: arcname (flat, inside the zip) -> path within the albumNarrator checkout.
#: The arcnames must match what `CONFIGS['narrator']` points at once the image
#: has unzipped into `files/narrator/`.
BUNDLE = [
    ('policy.pt', 'runs/{run}/best.pt'),
    ('attribute_axes.npz', 'tools/data/attribute_axes.npz'),
]

#: The embedding width this service produces (model_version 2). A policy
#: trained at any other width cannot read our frames at all.
EXPECTED_CLIP_DIM = 768

#: The four attribute axes `src/pipeline/select/narrator.py::_axis_scores`
#: projects onto. A missing one silently becomes a neutral 0.5.
EXPECTED_AXES = {'candid', 'indoor', 'lighting', 'bgcolor'}


def sha256(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def check_checkpoint(path: str, problems: list) -> dict:
    """The checkpoint loads, and its training config matches this service."""
    try:
        import torch
    except ImportError:
        problems.append("torch not importable; cannot verify the checkpoint")
        return {}

    try:
        blob = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"checkpoint will not load: {type(exc).__name__}: {exc}")
        return {}

    for key in ('cfg', 'state_dict'):
        if key not in blob:
            problems.append(f"checkpoint has no '{key}' -- the loader needs both")

    facts = {}
    cfg = blob.get('cfg') or {}
    env = cfg.get('env') if isinstance(cfg, dict) else None
    clip_dim = (env or {}).get('clip_dim') if isinstance(env, dict) else None
    facts['clip_dim'] = clip_dim
    if clip_dim is not None and int(clip_dim) != EXPECTED_CLIP_DIM:
        problems.append(f"checkpoint clip_dim={clip_dim}, but this service "
                        f"produces {EXPECTED_CLIP_DIM}-d embeddings")
    facts['update'] = blob.get('update')
    facts['best_score'] = blob.get('best_score')
    facts['tensors'] = len(blob.get('state_dict') or {})
    return facts


def check_axes(path: str, problems: list) -> dict:
    import numpy as np

    try:
        bundle = np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"axes will not load: {type(exc).__name__}: {exc}")
        return {}

    for key in ('names', 'axes'):
        if key not in bundle:
            problems.append(f"axes file has no '{key}' array")
            return {}

    names = [str(n) for n in bundle['names'].tolist()]
    axes = bundle['axes']
    missing = EXPECTED_AXES - set(names)
    if missing:
        problems.append(f"axes file is missing {sorted(missing)} -- those "
                        f"features would silently stay at 0.5")
    if axes.ndim != 2 or axes.shape[0] != len(names):
        problems.append(f"axes shape {axes.shape} does not match {len(names)} names")
    elif axes.shape[1] != EXPECTED_CLIP_DIM:
        problems.append(f"axes are {axes.shape[1]}-d, not {EXPECTED_CLIP_DIM}-d, "
                        f"so the projection against our embeddings cannot work")
    return {'names': names, 'shape': list(axes.shape)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--narrator-root', required=True,
                        help="albumNarrator checkout to take the artefacts from.")
    parser.add_argument('--run', default='pod_v29',
                        help="Run directory under runs/ whose best.pt ships.")
    parser.add_argument('--out', required=True,
                        help="Output zip. Do NOT reuse the name of a bundle "
                             "already uploaded -- the Dockerfile URL is versioned "
                             "so a rebuild cannot silently pick up new weights.")
    parser.add_argument('--verify', action='store_true',
                        help="Load and cross-check both artefacts before zipping.")
    args = parser.parse_args()

    if osp.exists(args.out):
        print(f"{args.out} already exists; pick a new version rather than "
              f"overwriting a published bundle")
        return 2

    members, problems = [], []
    for arcname, template in BUNDLE:
        source = osp.join(args.narrator_root, template.format(run=args.run))
        if not osp.exists(source):
            problems.append(f"missing {arcname}: {source}")
            continue
        members.append((arcname, source))

    if problems:
        print("cannot build the bundle:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    facts = {}
    if args.verify:
        print("verifying:")
        for arcname, source in members:
            if arcname.endswith('.pt'):
                facts[arcname] = check_checkpoint(source, problems)
            elif arcname.endswith('.npz'):
                facts[arcname] = check_axes(source, problems)
        for arcname, detail in facts.items():
            print(f"  {arcname}: {detail}")
        if problems:
            print("\nverification failed:")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print("  all checks passed")

    manifest = {
        'run': args.run,
        'narrator_root': args.narrator_root,
        'members': {},
        'facts': facts,
    }
    with zipfile.ZipFile(args.out, 'w', zipfile.ZIP_DEFLATED) as archive:
        for arcname, source in members:
            archive.write(source, arcname)
            manifest['members'][arcname] = {
                'bytes': osp.getsize(source),
                'sha256': sha256(source),
            }
        archive.writestr('bundle.json', json.dumps(manifest, indent=2))

    print(f"\nwrote {args.out} ({osp.getsize(args.out) / 1e6:.1f} MB)")
    for arcname, detail in manifest['members'].items():
        print(f"  {arcname}: {detail['bytes'] / 1e6:.1f} MB  sha256 {detail['sha256'][:16]}...")
    print("\nNext: upload to the `ai-models` container under `album-narrator/` "
          "and update the Dockerfile ADD URL.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
