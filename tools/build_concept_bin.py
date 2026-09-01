"""Build a pre-query concept bin from text phrases.

The `.bin` files under `files/pre_queries/v1|v2/` hold CLIP *text* embeddings
for a named concept — the same format `load_pre_queries_embeddings` reads:

    int32 dim, int32 n, then n*dim little-endian float32 (rows are embeddings)

v1 is the 512-d ViT-B/32 space (image model_version 1), v2 the 768-d
ViT-L-14-quickgelu space (model_version 2). A concept must exist in both,
because a gallery's embeddings can be in either space.

Vectors come from the TextEmbedding service and are L2-normalised here, so a
plain dot product against a normalised image embedding is a cosine.

    python tools/build_concept_bin.py confetti --dry-run
    python tools/build_concept_bin.py confetti
"""

from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np
import requests

#: The TextEmbedding service (see the pictime-data notes). IPs do move.
TEXT_EMBEDDING_HOSTS = ("10.0.28.215", "10.0.29.195")
TEXT_EMBEDDING_PORT = 8080

PRE_QUERIES_DIR = os.path.join("files", "pre_queries")

#: Concept phrase sets. Scoring takes the MAX over a concept's phrases, so
#: adding a phrasing broadens the concept rather than diluting it — but keep a
#: concept visually coherent, or the score stops meaning one thing.
CONCEPTS = {
    # The send-off: guests celebrating the couple as they leave the ceremony.
    # Deliberately broad — the substance varies by wedding (confetti, bubbles,
    # petals, rice, sparklers) and the tag covers the moment, not the material.
    #
    # The last two phrases describe the *formation* (a tunnel of guests) rather
    # than anything thrown. They catch send-offs with no substance at all, but
    # they are also the phrases most likely to fire on a plain recessional —
    # measure their contribution before trusting them.
    "send_off": [
        "guests throwing confetti over the bride and groom",
        "bride and groom walking through a shower of confetti",
        "soap bubbles floating around the bride and groom",
        "guests blowing bubbles as the couple walks past",
        "guests throwing flower petals at the newlyweds",
        "guests throwing rice at the bride and groom",
        "wedding send-off with sparklers at night",
        "colorful streamers and confetti in the air at a wedding celebration",
        "bride and groom exiting through a tunnel of cheering guests",
        "guests lining both sides of a path cheering the couple",
    ],
}


def embed(text: str, model_version: int) -> np.ndarray:
    last = None
    for host in TEXT_EMBEDDING_HOSTS:
        try:
            response = requests.get(
                f"http://{host}:{TEXT_EMBEDDING_PORT}/text-embedding",
                params={"text": text, "modelVersion": str(model_version)},
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("error"):
                raise RuntimeError(f"service reported an error: {payload}")
            return np.asarray(payload["results"]["text_embeddings"][0], dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 - try the next host
            last = exc
    raise RuntimeError(
        f"TextEmbedding unreachable on {TEXT_EMBEDDING_HOSTS} (needs VPN): {last}"
    )


def build_matrix(phrases, model_version: int) -> np.ndarray:
    rows = []
    for phrase in phrases:
        vector = embed(phrase, model_version)
        norm = np.linalg.norm(vector)
        rows.append(vector / norm if norm else vector)
    return np.vstack(rows).astype(np.float32)


def serialize(matrix: np.ndarray) -> bytes:
    n, dim = matrix.shape
    return struct.pack("<2i", dim, n) + matrix.astype("<f4").tobytes()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("concept", choices=sorted(CONCEPTS),
                    help="Concept to build (its phrases live in CONCEPTS).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Fetch and report, but do not write the .bin files.")
    args = ap.parse_args()

    phrases = CONCEPTS[args.concept]
    print(f"{args.concept}: {len(phrases)} phrases")

    for model_version, dim in ((1, 512), (2, 768)):
        matrix = build_matrix(phrases, model_version)
        if matrix.shape[1] != dim:
            raise SystemExit(
                f"v{model_version}: expected {dim}-d, service returned {matrix.shape[1]}-d"
            )

        # A concept whose phrases are near-identical adds nothing; report the
        # spread so a bad phrase set is visible.
        sims = matrix @ matrix.T
        off = sims[~np.eye(len(matrix), dtype=bool)]
        print(f"  v{model_version}: {matrix.shape}  "
              f"inter-phrase cosine {off.min():.3f}..{off.max():.3f} (mean {off.mean():.3f})")

        path = os.path.join(PRE_QUERIES_DIR, f"v{model_version}", f"{args.concept}.bin")
        if args.dry_run:
            print(f"       would write {path} ({len(serialize(matrix))} bytes)")
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(serialize(matrix))
        print(f"       wrote {path} ({os.path.getsize(path)} bytes)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
