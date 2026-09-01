"""The substage contract.

A substage is one replaceable unit of work. It declares what it needs and what
it produces, and it transforms an :class:`~src.pipeline.contracts.AlbumContext`
in place.

Subclasses implement :meth:`SubStage.execute`. The base class handles the
uniform parts — timing, requirement checking, error containment — so every
substage behaves the same way regardless of who wrote it.

    class MySubStage(SubStage):
        name = "enrich.my_thing"
        requires = {photo(Col.EMBEDDING)}
        provides = {photo("my_column")}

        def execute(self, ctx):
            ctx.photos["my_column"] = ...
            return ctx
"""

from __future__ import annotations

import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from typing import ClassVar, FrozenSet, Set

from src.pipeline.contracts import AlbumContext, StageRecord


class SubStage(ABC):
    """Base class for every replaceable unit in the pipeline."""

    #: Dotted identifier, e.g. ``"enrich.semantic_tags"``. Also the registry key.
    name: ClassVar[str] = "unnamed"

    #: Requirement tokens (see :mod:`src.pipeline.contracts`) that must hold
    #: before this substage runs.
    requires: ClassVar[Set[str]] = frozenset()

    #: Tokens this substage guarantees afterwards. Checked only when the
    #: substage actually did work (see :meth:`applies_to`).
    provides: ClassVar[Set[str]] = frozenset()

    #: When True a failure is logged and the pipeline continues. Use for
    #: best-effort enrichment whose absence downstream code already tolerates.
    optional: ClassVar[bool] = False

    def __init__(self, **options):
        #: Per-instance configuration, so the same class can be registered twice
        #: with different settings.
        self.options = options

    # -- to implement -------------------------------------------------------

    @abstractmethod
    def execute(self, context: AlbumContext) -> AlbumContext:
        """Do the work. Raise to signal failure; the runner contains it."""

    def applies_to(self, context: AlbumContext) -> bool:
        """Whether this substage should run at all for this request.

        Override for conditional substages (wedding-only enrichment, the manual
        vs AI selection split, ...). A skipped substage is not held to its
        ``provides`` contract.
        """
        return True

    # -- uniform wrapper ----------------------------------------------------

    def __call__(self, context: AlbumContext) -> AlbumContext:
        started = datetime.now()
        photos_in = len(context.photos) if context.photos is not None else 0

        def record(ok: bool, note: str = "") -> None:
            context.diagnostics.append(
                StageRecord(
                    name=self.name,
                    ok=ok,
                    seconds=(datetime.now() - started).total_seconds(),
                    photos_in=photos_in,
                    photos_out=len(context.photos) if context.photos is not None else 0,
                    note=note,
                )
            )

        if not self.applies_to(context):
            record(True, "skipped")
            return context

        missing = context.missing(self.requires)
        if missing:
            message = f"{self.name}: unmet requirements {missing}"
            record(False, "unmet requirements")
            if self.optional:
                if context.logger:
                    context.logger.warning(message)
                return context
            return context.fail(message)

        try:
            context = self.execute(context) or context
        except Exception as exc:  # noqa: BLE001 - boundary: contain and report
            where = _origin(exc)
            message = f"{self.name} failed: {exc}{where}"
            record(False, str(exc))
            if self.optional:
                if context.logger:
                    context.logger.warning(message)
                return context
            return context.fail(message)

        if context.failed:
            # execute() called ctx.fail() itself; its message is the specific
            # one, so don't paper over it with a contract complaint.
            record(False, context.error)
            return context

        unmet = context.missing(self.provides)
        if unmet:
            message = f"{self.name}: declared but did not provide {unmet}"
            record(False, "broken provides contract")
            if self.optional:
                if context.logger:
                    context.logger.warning(message)
                return context
            return context.fail(message)

        record(True)
        return context


def _origin(exc: Exception) -> str:
    """`. In <func>, line <n>, <file>` for the innermost frame, or ''."""
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return ""
    filename, lineno, func, _ = frames[-1]
    return f". In {func}, line {lineno}, {filename}"


class FunctionSubStage(SubStage):
    """Adapter that turns a plain ``fn(context) -> context`` into a substage.

    Handy for wrapping an existing function without writing a class::

        FunctionSubStage.build("enrich.thing", my_fn, requires={...})
    """

    def __init__(self, fn, **options):
        super().__init__(**options)
        self._fn = fn

    def execute(self, context: AlbumContext) -> AlbumContext:
        return self._fn(context)

    @classmethod
    def build(
        cls,
        name: str,
        fn,
        requires: FrozenSet[str] = frozenset(),
        provides: FrozenSet[str] = frozenset(),
        optional: bool = False,
    ) -> "FunctionSubStage":
        subclass = type(
            f"{name.replace('.', '_')}_SubStage",
            (cls,),
            {
                "name": name,
                "requires": frozenset(requires),
                "provides": frozenset(provides),
                "optional": optional,
            },
        )
        return subclass(fn)
