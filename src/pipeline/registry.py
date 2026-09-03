"""Substage registry and the default pipeline compositions.

Substages are looked up by name, so replacing one is a registration rather than
an edit to the call site::

    from src.pipeline.registry import override
    from my_experiments import BetterScorer

    override("select.score", BetterScorer)

Every pipeline built afterwards picks up the replacement. ``CONFIGS`` can carry
the same thing declaratively via ``CONFIGS['pipeline_overrides']``, which maps a
substage name to a ``"module:ClassName"`` string.
"""

from __future__ import annotations

import importlib
from typing import Dict, List, Optional, Sequence, Type

from src.pipeline.runner import Pipeline
from src.pipeline.substage import SubStage
from utils.configs import CONFIGS

_REGISTRY: Dict[str, Type[SubStage]] = {}


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------


def register(substage_cls: Type[SubStage]) -> Type[SubStage]:
    """Class decorator. Registers under ``substage_cls.name``."""
    name = substage_cls.name
    if name in (None, "", "unnamed"):
        raise ValueError(f"{substage_cls!r} must set a class-level `name`")
    _REGISTRY[name] = substage_cls
    return substage_cls


def override(name: str, substage_cls: Type[SubStage]) -> None:
    """Point an existing slot at a different implementation."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown substage {name!r}. Known: {sorted(_REGISTRY)}")
    _REGISTRY[name] = substage_cls


def get(name: str) -> Type[SubStage]:
    _load_configured_overrides()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown substage {name!r}. Known: {sorted(_REGISTRY)}") from None


def known() -> List[str]:
    return sorted(_REGISTRY)


_overrides_loaded = False


def _load_configured_overrides() -> None:
    """Apply ``CONFIGS['pipeline_overrides']`` once, lazily."""
    global _overrides_loaded
    if _overrides_loaded:
        return
    _overrides_loaded = True

    for name, target in (CONFIGS.get("pipeline_overrides") or {}).items():
        module_name, _, class_name = target.partition(":")
        module = importlib.import_module(module_name)
        override(name, getattr(module, class_name))


# --------------------------------------------------------------------------
# Default compositions
# --------------------------------------------------------------------------
#
# Order is behaviour. These sequences reproduce, step for step, what
# `read_messages` and `SelectionStage.get_selection` did as monoliths.

#: Pure acquisition: read bytes, decode them, organise them into the photo
#: table. No classification, no projection, no derived metadata.
INGEST: Sequence[str] = (
    "ingest.request",
    "ingest.design",
    "ingest.rating",
    "ingest.project_registry",
    "ingest.embeddings",
    "ingest.gallery_assets",
)

#: Everything the read stage used to do that is not reading: classification,
#: CLIP projection, identity resolution, temporal normalisation and the
#: event/relationship detectors.
ENRICH: Sequence[str] = (
    "enrich.duplicate_shots",
    "enrich.gallery_type",
    "enrich.content_class",
    "enrich.identities",
    "enrich.same_sex_couple",
    "enrich.semantic_tags",
    "enrich.require_cluster_data",
    "enrich.people_cluster",
    "ingest.merge_ratings",
    "ingest.scenes",
    "enrich.temporal",
    "enrich.parents",
    "enrich.ceremony_anchor",
    "enrich.key_pages",
)

#: Choosing the photos and the per-category spread budget.
SELECT: Sequence[str] = (
    "select.route",
    "select.budget",
    "select.preselect",
    "select.pick",
    "select.publish",
)


def build(names: Sequence[str], name: str = "pipeline", logger=None, **per_substage) -> Pipeline:
    """Instantiate a pipeline from substage names.

    ``per_substage`` passes options to individual substages, keyed by name::

        build(SELECT, options={"select.pick": {"strategies": my_registry}})
    """
    options: Dict[str, dict] = per_substage.get("options") or {}
    substages = [get(n)(**options.get(n, {})) for n in names]
    return Pipeline(name, substages, logger=logger)


def build_read(logger=None, **per_substage) -> Pipeline:
    """Ingest + enrich — the decomposed replacement for ``read_messages``."""
    return build(tuple(INGEST) + tuple(ENRICH), name="read", logger=logger, **per_substage)


def build_ingest(logger=None, **per_substage) -> Pipeline:
    return build(INGEST, name="ingest", logger=logger, **per_substage)


def build_enrich(logger=None, **per_substage) -> Pipeline:
    return build(ENRICH, name="enrich", logger=logger, **per_substage)


def build_select(logger=None, **per_substage) -> Pipeline:
    """The decomposed replacement for ``SelectionStage.get_selection``."""
    return build(SELECT, name="select", logger=logger, **per_substage)
