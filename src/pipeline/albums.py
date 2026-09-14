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

from dataclasses import replace as _replace

from src.pipeline.contracts import AlbumContext, AiHints, GalleryFacts, Services


@dataclass(frozen=True)
class AlbumVariant:
    """One album's brief: what to compose differently from the request.

    A variant is an *override*, not a whole configuration, so a field left
    None means "as the request asked". That is what makes the default -- a
    single variant overriding nothing -- byte-identical to not having variants
    at all.

    ``focus`` names a column of `files/focus_csv.csv`, which is the
    per-category spread profile the budget is built from, so two focus values
    give genuinely different albums out of one gallery read. `brideAndGroom`,
    `parents` and `everyoneElse` are the columns that exist.

    ``photo_ids`` and ``person_ids`` are the *content* axis to focus's *shape*
    one. A focus changes how many spreads each category gets and leaves the
    picker to fill them, which on a gallery the couple dominates fills a
    family profile with the couple. These two override `aiMetadata`, which
    `select.preselect` honours before any ranking and `person_score` weighs
    everywhere else -- so a variant can state what an album is *of*, not only
    how it is shaped. `src.pipeline.family` builds the pair for a parents
    album.

    Overriding rather than extending is deliberate: a variant is a different
    brief, and the request's own hand picks are for the album the request
    asked for. Set `family_album.replace_user_picks` False to union instead.
    """

    name: str
    focus: Optional[Tuple[str, ...]] = None
    photo_ids: Optional[Tuple[int, ...]] = None
    person_ids: Optional[Tuple[int, ...]] = None

    def apply(self, context: AlbumContext) -> AlbumContext:
        """Overlay this variant onto a fresh album context.

        Applied to the context rather than to the request, because the request
        is shared by every album -- and `hints` is replaced rather than mutated
        for the same reason.

        ``present`` is never touched. It is what `select.route` splits manual
        from AI on, and a variant that flipped it would not steer an album, it
        would send the request down a different path entirely.
        """
        changes: Dict[str, Any] = {}
        if self.focus is not None:
            changes["focus"] = list(self.focus)
        if self.photo_ids is not None:
            changes["photo_ids"] = list(self.photo_ids)
        if self.person_ids is not None:
            changes["person_ids"] = list(self.person_ids)
        if changes:
            context.hints = _replace(context.hints, **changes)
        context.variant_name = self.name
        return context


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

    #: The album briefs `enrich.variants` planned for this gallery. Empty means
    #: nobody planned any, and one album with no overrides is composed.
    variants: Tuple[Any, ...] = ()

    #: Shared (by reference).
    designs: Any = None
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
            variants=tuple(getattr(source, "variants", None) or ()),
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


#: Attributes ProcessStage and assembly_output read off the message object
#: itself rather than out of ``content``. A sibling is useless without them,
#: and they are reference data, so they are shared rather than copied.
_SHARED_MESSAGE_ATTRS = ("designsInfo", "pagesInfo")


def sibling_message(message, index: int, count: int):
    """A message of its own for album ``index``, sharing the source.

    Albums after the first need somewhere private to put their results:
    ProcessStage writes the laid-out frame and ``album_doc`` onto the message,
    so N albums on one message would overwrite each other. ``content`` is a
    read-only property over ``body``, so the body is what gets replaced -- a
    shallow copy, which gives each album its own key mapping while the values
    it never writes (designInfo, aiMetadata, ...) stay shared.

    The source and parent queue are deliberately the *same* object: this is one
    queue message that happens to produce several albums. That makes
    ``delete()`` dangerous on a sibling -- it would delete the one underlying
    queue message N times -- so only the group's first message is ever deleted.
    :func:`album_group` is how the report side finds that out.

    Album 0 keeps the original message, so at N=1 nothing about the message
    plumbing changes at all.
    """
    if index == 0:
        original = message
        setattr(original, "album_index", 0)
        setattr(original, "album_count", count)
        return original

    # `copy.copy` rather than `type(message)(...)`: the constructor signature
    # is ptinfra's, and reconstructing would also drop any attribute a stage
    # has since hung on the message. The copy keeps `source` and `parent`, so
    # the sibling still refers to the one real queue message.
    twin = copy.copy(message)
    if not hasattr(message, "body"):
        # `content` is a read-only property over `body` on a real Message, so
        # `body` is the only place a private mapping can be installed. Refuse
        # loudly rather than hand back a sibling that silently shares content.
        raise TypeError(f"{type(message).__name__} has no `body`; a sibling "
                        "album message cannot be given its own content")
    twin.body = dict(message.body)
    for attr in _SHARED_MESSAGE_ATTRS:
        if hasattr(message, attr):
            setattr(twin, attr, getattr(message, attr))
    twin.error = None
    setattr(twin, "album_index", index)
    setattr(twin, "album_count", count)
    setattr(twin, "album_sibling_of", message)
    return twin


def album_group(messages) -> List[List[Any]]:
    """Group messages by the queue message they came from, order preserved.

    Siblings share ``source``, so this is what lets the report side send one
    result and delete one queue message however many albums were produced.
    Messages with no siblings group alone, which is every message today.
    """
    groups: List[List[Any]] = []
    seen: Dict[int, int] = {}
    for message in messages:
        source = getattr(message, "source", None)
        key = id(source) if source is not None else id(message)
        if key in seen:
            groups[seen[key]].append(message)
        else:
            seen[key] = len(groups)
            groups.append([message])
    return groups


@dataclass
class AlbumRun:
    """One album's outcome: which variant produced it, and what it produced."""

    index: int
    context: AlbumContext
    variant: Optional[Any] = None

    @property
    def message(self):
        """The message this album publishes to (its sibling, or the original)."""
        return self.context.message

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
    if variants is not None:
        planned = list(variants)
    elif base.variants:
        # What `enrich.variants` decided for this gallery. The count comes from
        # the plan, not from config: how many albums are worth making is a fact
        # about the gallery.
        planned = list(base.variants)
    else:
        planned = [None] * max(1, count)

    runs: List[AlbumRun] = []
    for index, variant in enumerate(planned):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        context = base.album_context(logger=logger)
        if variant is not None and hasattr(variant, "apply"):
            variant.apply(context)
        # Each album publishes to its own message, so ProcessStage's writes --
        # the laid-out frame, `album_doc` -- cannot collide. Album 0 keeps the
        # original, so N=1 is unchanged.
        context.message = sibling_message(message, index, len(planned))
        # The report side labels each album from its message, because that is
        # all it has by then -- the contexts do not travel between stages.
        setattr(context.message, "variant_name", context.variant_name)
        runs.append(AlbumRun(index=index, context=pipeline.run(context),
                             variant=variant))
    return runs
