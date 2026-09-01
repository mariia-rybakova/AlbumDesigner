"""Ingest substages: read bytes, decode them, organise them.

Nothing in this package derives anything. No classification, no embedding
projection, no identity resolution, no event detection — those all live in
:mod:`src.pipeline.enrich`. If a substage here starts *inferring* something
about a photo rather than reading it, it belongs on the other side of the line.

Importing this package registers every substage it defines.
"""

from src.pipeline.ingest import (  # noqa: F401  (imported for registration)
    design,
    embeddings,
    gallery_assets,
    project_registry,
    rating,
    request,
    scenes,
)

__all__ = [
    "design",
    "embeddings",
    "gallery_assets",
    "project_registry",
    "rating",
    "request",
    "scenes",
]
