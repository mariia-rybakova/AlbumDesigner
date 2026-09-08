"""One gallery, one or many albums.

Phase 1 of `docs/multi_album_plan.md`. Selection mutates the state it is handed
-- it narrows the photo frame to what it chose, and that is correct, because
ProcessStage lays out the selected frame and reads it from exactly that key. It
only becomes wrong when a second album inherits the first one's output, which
is what running selection twice over one message does today:

    pass 1: saw  60 photos -> selected 5
    pass 2: saw   5 photos -> selected 5

`tests/test_multi_album_isolation.py` pins that. The fix is not to stop
narrowing; it is to stop albums sharing the thing being narrowed.

So the read's result is frozen into a :class:`GalleryBase` and every album is
composed from its own copy of it. Isolation stops being a matter of remembering
what to reset -- which is how three separate contamination bugs got into this
codebase -- and becomes a property of the shape.

At N=1 this is behaviourally identical to calling
``AlbumContext.for_message`` once, which is what it replaces.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.pipeline.contracts import AlbumContext, AiHints, GalleryFacts, Services


@dataclass(frozen=True)
class GalleryBase:
    """What the read produced, frozen, plus the means to hand out copies.

    Two kinds of state, and the split is the whole point:

    *Per-album* — everything selection writes to. Copied for each album, so one
    album cannot see another's work: the photo frame, and the small mutable
    sidecars (`facts`, `hints`, `available_photo_ids`, `key_pages`).

    *Shared* — reference data selection only ever reads. Handed over by
    reference on purpose: `designs` carries the product's layout frames, and
    the ingest sidecars are lookup tables. Copying those per album would cost
    real memory and buy nothing.

    Frozen so that "the base" cannot be quietly mutated into "the last album's
    state" — the failure mode this class exists to prevent.
    """

    message: Any
    request: Dict[str, Any]
    project_url: Optional[str]
    project_id: Any
    services: Services

    #: Per-album (copied).
    photos: pd.DataFrame
    facts: GalleryFacts
    hints: AiHints
    available_photo_ids: Tuple[int, ...]
    key_pages: Any

    #: Shared (by reference).
    designs: Any
    clip_embeddings: Optional[pd.DataFrame] = None
    ratings: Optional[pd.DataFrame] = None
    social_circles: Optional[pd.DataFrame] = None
    person_details: Optional[pd.DataFrame] = None
    is_in_vector_db: Optional[bool] = None

    @classmethod
    def capture(cls, message, logger=None, services=None) -> "GalleryBase":
        """Freeze the read's output for this message.

        Goes through `AlbumContext.for_message` once, which is where the read's
        own context (with everything INGEST and ENRICH derived) is picked up, or
        rehydrated from ``message.content`` when this process did not do the
        read. After this call the context is no longer used for composing --
        every album gets a fresh one from :meth:`album_context`.
        """
        source = AlbumContext.for_message(message, logger=logger, services=services)
        return cls(
            message=message,
            request=source.request,
            project_url=source.project_url,
            project_id=source.project_id,
            services=source.services,
            photos=source.photos,
            facts=source.facts,
            hints=source.hints,
            available_photo_ids=tuple(source.available_photo_ids or ()),
            key_pages=source.key_pages,
            designs=source.designs,
            clip_embeddings=source.clip_embeddings,
            ratings=source.ratings,
            social_circles=source.social_circles,
            person_details=source.person_details,
            is_in_vector_db=source.is_in_vector_db,
        )

    def album_context(self, logger=None) -> AlbumContext:
        """A context for one album: its own frame, the shared reference data.

        Built by construction rather than through `AlbumContext.from_message`,
        because that attaches itself to the message — so each album would
        overwrite the message's context and the last one to run would win.
        `message` is still set, so `sync_to_message` works for whichever album
        is published.

        `selection`, `predefined`, `all_photos` and the selection working state
        are deliberately left unset: they are what an album *produces*, and
        carrying them over is the contamination this class prevents.
        """
        return AlbumContext(
            message=self.message,
            logger=logger,
            services=self.services,
            request=self.request,
            project_url=self.project_url,
            project_id=self.project_id,
            available_photo_ids=list(self.available_photo_ids),
            hints=copy.copy(self.hints),
            photos=self.photos.copy(),
            all_photos=None,
            designs=self.designs,
            clip_embeddings=self.clip_embeddings,
            ratings=self.ratings,
            social_circles=self.social_circles,
            person_details=self.person_details,
            is_in_vector_db=self.is_in_vector_db,
            facts=copy.copy(self.facts),
            key_pages=copy.copy(self.key_pages),
        )


@dataclass
class AlbumRun:
    """One album's outcome: which variant produced it, and what it produced."""

    index: int
    context: AlbumContext
    variant: Optional[Any] = None

    @property
    def failed(self) -> bool:
        return bool(self.context.failed)

    @property
    def photo_ids(self) -> List[int]:
        selection = self.context.selection
        return list(selection.photo_ids) if selection is not None else []


def compose_albums(message, pipeline, count: int = 1, logger=None,
                   services=None, variants: Optional[Sequence[Any]] = None,
                   seed: Optional[int] = None) -> List[AlbumRun]:
    """Run selection once per album, each from its own copy of the gallery.

    ``count`` albums, or one per entry of ``variants`` when given. The variant
    is carried but not yet applied — `enrich.variants` and the config overlay
    are Phase 3; at N=1 with no variants and no seed this is exactly what
    ``pipeline.run(AlbumContext.for_message(message))`` did.

    Sequential on purpose. `CONFIGS` is process-global, so a variant that
    overlays config cannot safely run in parallel with another, and the crop
    subprocess in ProcessStage shares one queue.

    ``seed`` reseeds Python's and numpy's global RNGs identically before every
    album, and is **off by default** so that this change cannot move a single
    album's output. It is needed because the pipeline is stochastic in more
    places than is obvious -- `select_random_image` in the non-wedding
    selection, medoid initialisation in `time_orientation_selection`,
    combination sampling in `spreads_layout/math_tools`, cover choice in
    `album_tools` -- and several of those only fire above a size threshold, so
    a gallery can look deterministic and stop being so. Without a seed, "N
    identical variants produce identical albums" is not testable; with the
    same seed per album, difference can only come from the variant, which is
    where it should come from.
    """
    base = GalleryBase.capture(message, logger=logger, services=services)
    planned = list(variants) if variants is not None else [None] * max(1, count)

    runs: List[AlbumRun] = []
    for index, variant in enumerate(planned):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        context = pipeline.run(base.album_context(logger=logger))
        runs.append(AlbumRun(index=index, context=context, variant=variant))
    return runs
