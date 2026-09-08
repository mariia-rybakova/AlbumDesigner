"""The AlbumDesigner substage pipeline.

The service used to be four coarse stages, two of which were monoliths: the
read stage both fetched data and derived a great deal from it, and the
selection stage was a single loop that scored, filtered, diversified and
allocated all at once. Neither could be changed a piece at a time.

This package breaks that into named substages that share one data-transfer
object (:class:`~src.pipeline.contracts.AlbumContext`) and declare what they
consume and produce. Composition is data — a list of names in
:mod:`src.pipeline.registry` — so any single substage can be replaced without
touching its neighbours.

    from src.pipeline import build_read, build_select

    read = build_read(logger)
    select = build_select(logger)
    context = select.run(read.run(AlbumContext.from_message(msg, logger, services)))

The dividing line the packages enforce:

* :mod:`src.pipeline.ingest` — reads bytes and organises them. Never infers.
* :mod:`src.pipeline.enrich` — infers: classification, CLIP projection,
  identity resolution, time normalisation, event detection.
* :mod:`src.pipeline.select` — chooses the photos and the spread budget.
"""

from src.pipeline.contracts import (
    AiHints,
    AlbumContext,
    Col,
    DesignSpec,
    GalleryFacts,
    KeyPages,
    SelectionOutcome,
    Services,
    StageRecord,
    ctx,
    photo,
)
from src.pipeline.albums import (AlbumRun, AlbumVariant, GalleryBase,
                                 album_group, compose_albums,
                                 sibling_message)
from src.pipeline.runner import Pipeline, chain
from src.pipeline.substage import FunctionSubStage, SubStage

# Importing the packages registers their substages, which the builders below
# resolve by name.
from src.pipeline import ingest as _ingest  # noqa: F401,E402
from src.pipeline import enrich as _enrich  # noqa: F401,E402
from src.pipeline import select as _select  # noqa: F401,E402

from src.pipeline.registry import (  # noqa: E402
    ENRICH,
    INGEST,
    SELECT,
    build,
    build_enrich,
    build_ingest,
    build_read,
    build_select,
    known,
    override,
    register,
)

__all__ = [
    "AlbumRun",
    "AlbumVariant",
    "GalleryBase",
    "album_group",
    "compose_albums",
    "sibling_message",
    "AiHints",
    "AlbumContext",
    "Col",
    "DesignSpec",
    "GalleryFacts",
    "KeyPages",
    "SelectionOutcome",
    "Services",
    "StageRecord",
    "ctx",
    "photo",
    "Pipeline",
    "chain",
    "SubStage",
    "FunctionSubStage",
    "INGEST",
    "ENRICH",
    "SELECT",
    "build",
    "build_ingest",
    "build_enrich",
    "build_read",
    "build_select",
    "known",
    "override",
    "register",
]
