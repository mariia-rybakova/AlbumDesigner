"""Export a composed album as an Album Designer ``predefinedLayout`` request block.

The two projects speak different units, and this module is the only place the translation
lives:

  * the narrator emits pages of **photo indices** into a ``PackedGallery``;
  * Album Designer's predefined-spreads bypass wants pic-time **photoIds** grouped per
    **spread**, plus cover photos in named fields (``src/predefined/models.py`` on its
    ``predefined_spreads`` branch).

Mapping (decided 2026-08-18):

  * **one narrator PAGE -> one designer SPREAD** (``photoIds``). The page is the unit the
    policy actually composes, so keeping it whole on one spread preserves what was learned.
    The designer still chooses the layout template, the left/right page split and the box
    assignment -- that is its "photos-only" granularity, the one its contract implements
    today. (``leftPhotoIds``/``rightPhotoIds`` are reserved there for a stricter mode; if
    that lands, two narrator pages can become one spread with the split fixed, and only
    ``album_to_predefined_block`` changes.)
  * **the two hero bookends -> ``firstPagePhotoIds`` / ``lastPagePhotoIds``.** They are
    single photos chosen by construction (earliest/latest 25% of the timeline, dominant
    cast), which is exactly what the designer's separate single-box cover path renders.
    They must NOT be emitted as spreads: the designer would lay them out as body spreads
    and the album would open on a one-photo spread instead of a cover.
  * **spread order is a hint only.** The designer re-sorts spreads by time downstream
    (``utils/time_processing.py::sort_groups_by_time``), and the narrator already orders
    chronologically under ``reward.order_by_time``, so the two agree rather than fight.

The block is emitted verbatim as the request's ``predefinedLayout`` key; its presence is the
single signal that makes the designer skip its own selection stage.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from .config import Config
from .data.schema import Gallery, PackedGallery
from .env import hero
from .env.album_env import AlbumNarratorEnv

# NOTE: ``eval.baselines`` (and therefore torch) is imported lazily inside ``compose``. The
# translation half of this module is pure numpy, so a caller that only needs to turn pages
# into a block -- a test, or a service that already has the album -- does not pay for torch.

# Key the designer looks for; absence of it means "run the normal selection pipeline".
REQUEST_KEY = "predefinedLayout"


def album_to_predefined_block(pg: PackedGallery, ordered_pages: list[list[int]],
                              hero_first: int = -1, hero_last: int = -1) -> dict[str, Any]:
    """Translate one composed album into the ``predefinedLayout`` block.

    ``ordered_pages`` is what ``AlbumState.ordered_pages()`` returns -- body pages with the
    hero bookends already prepended/appended when ``reward.hero_pages`` is on. Pass the hero
    indices so they can be lifted out into the cover fields; ``-1`` (the disabled value)
    leaves the covers absent and every page becomes a spread.

    Heroes are identified by CONTENT rather than by position, matching
    ``RewardEngine._local_pages``, so this holds whatever order the pages arrive in.
    """
    heroes = {i for i in (hero_first, hero_last) if i >= 0}

    def ids(page: list[int]) -> list[int]:
        # Python ints, not np.int64: json.dump cannot serialize numpy scalars.
        return [int(pg.photo_ids[i]) for i in page]

    spreads = [{"photoIds": ids(p)} for p in ordered_pages
               if p and not (len(p) == 1 and p[0] in heroes)]

    block: dict[str, Any] = {"spreads": spreads}
    if hero_first >= 0:
        block["firstPagePhotoIds"] = ids([hero_first])
    if hero_last >= 0:
        block["lastPagePhotoIds"] = ids([hero_last])
    return block


def block_stats(block: dict[str, Any]) -> dict[str, Any]:
    """Shape summary for logging -- what a reviewer wants to see before shipping a demo."""
    sizes = [len(s["photoIds"]) for s in block["spreads"]]
    n_cover = len(block.get("firstPagePhotoIds", [])) + len(block.get("lastPagePhotoIds", []))
    return {
        "spreads": len(sizes),
        "photos_in_spreads": int(sum(sizes)),
        "cover_photos": n_cover,
        "photos_total": int(sum(sizes)) + n_cover,
        "photos_per_spread": sizes,
        "mean_photos_per_spread": float(np.mean(sizes)) if sizes else 0.0,
    }


def validate_block(block: dict[str, Any], gallery_photo_ids: set[int] | None = None) -> list[str]:
    """Problems that would make the designer misbehave. Empty list = clean.

    Checked here rather than trusted because the failure modes are quiet: a duplicated id
    puts the same photo on two spreads, and an id the gallery does not contain simply
    disappears when the designer filters its dataframe to ``all_photo_ids()``.
    """
    problems: list[str] = []
    seen: dict[int, int] = {}
    for k, spread in enumerate(block["spreads"]):
        if not spread["photoIds"]:
            problems.append(f"spread {k} is empty")
        for pid in spread["photoIds"]:
            if pid in seen:
                problems.append(f"photo {pid} appears on spreads {seen[pid]} and {k}")
            seen[pid] = k
    for field in ("firstPagePhotoIds", "lastPagePhotoIds"):
        for pid in block.get(field, []):
            if pid in seen:
                problems.append(f"cover photo {pid} ({field}) is also on spread {seen[pid]}")
    if gallery_photo_ids is not None:
        # Parenthesized deliberately: `-` binds tighter than `|`, so without them only the
        # cover ids would be checked and every spread id would be reported unknown.
        referenced = ({pid for s in block["spreads"] for pid in s["photoIds"]}
                      | {pid for f in ("firstPagePhotoIds", "lastPagePhotoIds")
                         for pid in block.get(f, [])})
        unknown = sorted(referenced - gallery_photo_ids)
        if unknown:
            problems.append(f"{len(unknown)} id(s) not in the gallery, e.g. {unknown[:5]}")
    return problems


def compose(net, cfg: Config, gallery: Gallery, sample_k: int = 8, seed: int = 0,
            temperature: float = 1.0) -> tuple[PackedGallery, list[list[int]], float]:
    """Run the policy on one gallery and return (packed gallery, best album, its score).

    Best-of-``sample_k`` stochastic rollouts. Since v27 a single greedy rollout is nearly as
    good (``greedy`` ~= best-of-8), so ``sample_k=1`` is a reasonable cheap mode, but
    sampling is kept the default because it is what every reported number uses.
    """
    from .eval import baselines   # lazy: keeps torch off the pure-translation path

    env = AlbumNarratorEnv(cfg)
    rng = np.random.default_rng(seed)
    scores, ordered = baselines.policy_sample_scores(env, gallery, net, sample_k, rng, temperature)
    return env.pg, ordered, float(np.max(scores))


def export_album(net, cfg: Config, gallery: Gallery, sample_k: int = 8, seed: int = 0
                 ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compose an album for ``gallery`` and return (block, stats)."""
    pg, ordered, score = compose(net, cfg, gallery, sample_k=sample_k, seed=seed)
    # Heroes are a deterministic function of the gallery (picked at every reset), so recompute
    # them rather than reading them off whichever rollout happened to finish last.
    first, last = hero.pick_hero_photos(pg) if cfg.reward.hero_pages else (-1, -1)
    block = album_to_predefined_block(pg, ordered, first, last)
    stats = block_stats(block)
    stats["album_score"] = score
    stats["gallery_id"] = pg.gallery_id
    stats["gallery_photos"] = int(pg.n)
    return block, stats


def inject_into_request(request: dict[str, Any], block: dict[str, Any]) -> dict[str, Any]:
    """Return ``request`` with the block attached (shallow copy; input is not mutated).

    The designer reads ``content['predefinedLayout']``, where ``content`` is the request body
    itself, so the block sits at the top level next to ``projectId`` / ``photos``.
    """
    out = dict(request)
    out[REQUEST_KEY] = block
    return out


def write_block(path: str, block: dict[str, Any], request_path: str | None = None,
                indent: int = 2) -> None:
    """Write the block alone, or a full designer request with the block injected."""
    payload = block
    if request_path:
        with open(request_path, "r", encoding="utf-8") as fh:
            payload = inject_into_request(json.load(fh), block)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent)
