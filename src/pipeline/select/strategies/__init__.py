"""Per-category selection strategies.

One class per way of choosing photos. ``default_registry()`` wires them to the
categories they handle; everything unclaimed falls through to
:class:`~src.pipeline.select.strategies.default.ContentClusterStrategy`.

To change how one category behaves, register a different strategy for it rather
than editing the driver::

    from src.pipeline.select.strategies import default_registry
    registry = default_registry().set("dancing", MyDancingStrategy())
    pipeline = build_select(logger, options={"select.pick": {"strategies": registry}})
"""

from src.pipeline.select.strategies.base import CategoryStrategy, StrategyRegistry
from src.pipeline.select.strategies.bride_prep import BridePrepStrategy
from src.pipeline.select.strategies.couple_time import (
    ORIENTATION_TIME_CATEGORIES,
    CoupleTimelineStrategy,
)
from src.pipeline.select.strategies.default import ContentClusterStrategy
from src.pipeline.select.strategies.parents import ParentsPortraitStrategy
from src.pipeline.select.strategies.persons import PERSONS_CATEGORIES, PersonCoverageStrategy


def default_registry() -> StrategyRegistry:
    """The stock category -> strategy wiring."""
    return (
        StrategyRegistry(fallback=ContentClusterStrategy())
        .add(BridePrepStrategy())
        .add(CoupleTimelineStrategy())
        .add(PersonCoverageStrategy())
        .add(ParentsPortraitStrategy())
    )


__all__ = [
    "CategoryStrategy",
    "StrategyRegistry",
    "BridePrepStrategy",
    "CoupleTimelineStrategy",
    "ContentClusterStrategy",
    "ParentsPortraitStrategy",
    "PersonCoverageStrategy",
    "ORIENTATION_TIME_CATEGORIES",
    "PERSONS_CATEGORIES",
    "default_registry",
]
