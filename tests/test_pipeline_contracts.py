"""Structural tests for the substage pipeline.

These guard the properties that make substages replaceable: the contracts line
up, composition is by name, and the ingest/enrich boundary does not erode.

    python -m pytest tests/test_pipeline_contracts.py -v
    python tests/test_pipeline_contracts.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import (  # noqa: E402
    ENRICH,
    INGEST,
    SELECT,
    AlbumContext,
    Col,
    SubStage,
    build_enrich,
    build_ingest,
    build_read,
    build_select,
    known,
    photo,
)
from src.pipeline.registry import get  # noqa: E402

#: Columns that are *inferred*, never read off a protobuf. No ingest substage
#: may claim to produce one of these — that is the whole point of the split.
DERIVED_COLUMNS = {
    Col.CLUSTER_CONTEXT,
    Col.IMAGE_QUERY_CONTENT,
    Col.IMAGE_SUBQUERY_CONTENT,
    Col.BRIDE_ID,
    Col.GROOM_ID,
    Col.PEOPLE_CLUSTER,
    Col.PARENT_CATEGORY,
    Col.GENERAL_TIME,
    Col.IMAGE_TIME_DATE,
    Col.KEY_PAGE,
}


def test_every_named_substage_resolves():
    for name in tuple(INGEST) + tuple(ENRICH) + tuple(SELECT):
        assert get(name).name == name, f"{name} is registered under a different name"


def test_pipelines_have_no_ordering_violations():
    for pipeline in (build_ingest(), build_enrich(), build_read(), build_select()):
        assert pipeline.unsatisfied() == [], (
            f"{pipeline.name}: {pipeline.unsatisfied()}"
        )


def test_ingest_never_derives():
    """The line between reading and inferring."""
    for substage in build_ingest():
        provided = {token.split(":", 1)[1] for token in substage.provides if token.startswith("photo:")}
        leaked = provided & DERIVED_COLUMNS
        assert not leaked, (
            f"{substage.name} claims to provide derived column(s) {sorted(leaked)}; "
            f"inference belongs in src/pipeline/enrich"
        )


def test_enrich_runs_after_ingest():
    """Every enrich substage's photo requirements are met by ingest + earlier
    enrich, so the read pipeline is orderable as written."""
    available = set()
    for substage in build_read():
        unmet = {
            token for token in substage.requires
            if token.startswith("photo:") and token not in available
        }
        # Columns no substage declares (e.g. number_bodies, main_persons) come
        # in with the protobuf read; only flag ones we claim to produce later.
        producible = set()
        for other in build_read():
            producible |= set(other.provides)
        assert not (unmet & producible), (
            f"{substage.name} requires {sorted(unmet & producible)} before it is provided"
        )
        available |= set(substage.provides)


def test_substage_can_be_replaced_by_name():
    class Stub(SubStage):
        name = "enrich.semantic_tags"

        def execute(self, context):
            return context

    pipeline = build_read()
    original = pipeline.substages[pipeline.index_of("enrich.semantic_tags")]
    pipeline.replace("enrich.semantic_tags", Stub())

    assert pipeline.substages[pipeline.index_of("enrich.semantic_tags")] is not original
    assert len(pipeline) == len(build_read()), "replacement changed the pipeline length"


def test_unmet_requirement_fails_loudly():
    """A substage whose inputs are missing must stop the pipeline, not run on
    an empty frame and quietly produce nothing."""

    class NeedsMissingColumn(SubStage):
        name = "test.needs_missing"
        requires = frozenset({photo("a_column_that_does_not_exist")})

        def execute(self, context):
            raise AssertionError("must not run")

    context = AlbumContext(photos=pd.DataFrame({Col.IMAGE_ID: [1, 2]}))
    context = NeedsMissingColumn()(context)

    assert context.failed
    assert "unmet requirements" in context.error or "a_column_that_does_not_exist" in context.error


def test_broken_provides_contract_fails_loudly():
    """A replacement that forgets to produce what it promised is caught at the
    boundary rather than downstream."""

    class Liar(SubStage):
        name = "test.liar"
        provides = frozenset({photo("promised")})

        def execute(self, context):
            return context

    context = AlbumContext(photos=pd.DataFrame({Col.IMAGE_ID: [1]}))
    context = Liar()(context)

    assert context.failed
    assert "did not provide" in context.error


def test_diagnostics_record_every_substage():
    class Noop(SubStage):
        name = "test.noop"

        def execute(self, context):
            return context

    context = AlbumContext(photos=pd.DataFrame({Col.IMAGE_ID: [1]}))
    context = Noop()(context)

    assert len(context.diagnostics) == 1
    record = context.diagnostics[0]
    assert record.name == "test.noop" and record.ok


def test_context_round_trips_through_a_message():
    """A later stage picks up the context the earlier stage left, and can also
    rebuild one from a message assembled elsewhere."""

    class FakeMessage:
        def __init__(self, content):
            self.content = content
            self.pagesInfo = {}
            self.designsInfo = {}

    photos = pd.DataFrame({Col.IMAGE_ID: [1, 2, 3]})
    message = FakeMessage({
        "base_url": "x://y", "projectId": 1, "photos": [1, 2],
        "gallery_photos_info": photos, "is_wedding": True, "is_artificial_time": False,
    })

    first = AlbumContext.from_message(message)
    first.photos = photos
    first.facts.is_wedding = True

    # Same process, straight from the previous stage: same object.
    second = AlbumContext.for_message(message)
    assert second is first

    # A message with no attached context is rebuilt from its content.
    fresh = FakeMessage(dict(message.content))
    rebuilt = AlbumContext.for_message(fresh)
    assert rebuilt.facts.is_wedding is True
    assert list(rebuilt.photos[Col.IMAGE_ID]) == [1, 2, 3]
    assert rebuilt.available_photo_ids == [1, 2]


def test_registry_lists_all_packages():
    names = known()
    assert any(n.startswith("ingest.") for n in names)
    assert any(n.startswith("enrich.") for n in names)
    assert any(n.startswith("select.") for n in names)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} structural tests passed")
