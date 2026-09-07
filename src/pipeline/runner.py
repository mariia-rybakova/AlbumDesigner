"""The pipeline runner: an ordered list of substages, executed against one context."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from src.pipeline.contracts import AlbumContext
from src.pipeline.substage import SubStage


class Pipeline:
    """Runs substages in order, stopping at the first non-optional failure.

    The runner owns no domain logic. It exists so that composition (which
    substages, in what order) is data rather than control flow, which is what
    makes an individual substage swappable.
    """

    def __init__(self, name: str, substages: Sequence[SubStage], logger=None):
        self.name = name
        self.substages: List[SubStage] = list(substages)
        self.logger = logger

    # -- composition --------------------------------------------------------

    def __iter__(self):
        return iter(self.substages)

    def __len__(self) -> int:
        return len(self.substages)

    def index_of(self, name: str) -> int:
        for i, substage in enumerate(self.substages):
            if substage.name == name:
                return i
        raise KeyError(f"{self.name}: no substage named {name!r}")

    def replace(self, name: str, substage: SubStage) -> "Pipeline":
        """Swap one substage for another, in place. The whole point of the
        decomposition::

            pipeline.replace("select.score", MyNewScorer())
        """
        self.substages[self.index_of(name)] = substage
        return self

    def insert_after(self, name: str, substage: SubStage) -> "Pipeline":
        self.substages.insert(self.index_of(name) + 1, substage)
        return self

    def remove(self, name: str) -> "Pipeline":
        del self.substages[self.index_of(name)]
        return self

    # -- execution ----------------------------------------------------------

    def run(self, context: AlbumContext) -> AlbumContext:
        if context.logger is None:
            context.logger = self.logger

        for substage in self.substages:
            if context.failed:
                break
            context = substage(context)

        self.log_trace(context)
        return context

    def log_trace(self, context: AlbumContext, level: str = "info") -> None:
        if self.logger is None:
            return
        emit = getattr(self.logger, level, None)
        if emit is None:
            return
        lines = [f"{self.name} trace:"]
        lines += [f"  {record}" for record in context.diagnostics]
        total = sum(r.seconds for r in context.diagnostics)
        lines.append(f"  {'total':<30} {total:7.3f}s")
        emit("\n".join(lines))

    # -- introspection ------------------------------------------------------

    def describe(self) -> str:
        """Human-readable contract listing. Useful when writing a replacement:
        it shows exactly what the slot is expected to consume and produce."""
        rows = [f"{self.name} ({len(self.substages)} substages)"]
        for substage in self.substages:
            rows.append(f"  {substage.name}")
            if substage.requires:
                rows.append(f"      requires: {', '.join(sorted(substage.requires))}")
            if substage.provides:
                rows.append(f"      provides: {', '.join(sorted(substage.provides))}")
        return "\n".join(rows)

    def unsatisfied(self) -> List[str]:
        """Static check: substages whose requirements no earlier substage
        provides. Catches ordering mistakes without running anything.

        Only tokens that *some* substage in this pipeline provides are treated
        as pipeline-internal; anything else is assumed to come in on the
        context from an earlier pipeline and is not reported.
        """
        producible = set()
        for substage in self.substages:
            producible |= set(substage.provides)

        available: set = set()
        problems: List[str] = []
        for substage in self.substages:
            for token in sorted(substage.requires):
                if token in producible and token not in available:
                    problems.append(f"{substage.name} requires {token} before it is provided")
            available |= set(substage.provides)
        return problems


def chain(name: str, *pipelines: Pipeline, logger=None) -> Pipeline:
    """Flatten several pipelines into one."""
    substages: List[SubStage] = []
    for pipeline in pipelines:
        substages.extend(pipeline.substages)
    return Pipeline(name, substages, logger=logger)
