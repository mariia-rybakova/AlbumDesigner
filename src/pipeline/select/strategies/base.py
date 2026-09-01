"""The per-category strategy contract and its registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional, Sequence, Tuple, Type

from src.pipeline.select.contracts import CategoryPicks, CategoryRequest


class CategoryStrategy(ABC):
    """How one content category chooses its photos.

    Implementations receive a fully prepared :class:`CategoryRequest` and return
    a :class:`CategoryPicks`. They must not touch anything else — no globals, no
    reaching back into the driver — which is what makes them swappable.
    """

    #: Categories this strategy claims. Empty means "the fallback".
    handles: Tuple[str, ...] = ()

    @abstractmethod
    def pick(self, request: CategoryRequest) -> CategoryPicks:
        ...


class StrategyRegistry:
    """Category name -> strategy instance, with a fallback.

    Swap one category's behaviour without touching any other::

        registry = default_registry()
        registry.set("dancing", MyDancingStrategy())
    """

    def __init__(self, fallback: CategoryStrategy):
        self._by_category: Dict[str, CategoryStrategy] = {}
        self._fallback = fallback

    def add(self, strategy: CategoryStrategy) -> "StrategyRegistry":
        for category in strategy.handles:
            self._by_category[category] = strategy
        return self

    def set(self, category: str, strategy: CategoryStrategy) -> "StrategyRegistry":
        self._by_category[category] = strategy
        return self

    def for_category(self, category: str) -> CategoryStrategy:
        return self._by_category.get(category, self._fallback)

    @property
    def fallback(self) -> CategoryStrategy:
        return self._fallback

    def categories(self) -> Sequence[str]:
        return sorted(self._by_category)
