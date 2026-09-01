"""Enrichment substages: everything the read stage used to derive.

These all used to be interleaved with the protobuf reads inside
``get_info_protobufs`` and ``read_messages``. They are separated here because
none of them is reading — each one *infers* something:

===========================  =========================================
``enrich.gallery_type``      wedding or not, from the classifier output
``enrich.content_class``     ``cluster_class`` int -> category name
``enrich.identities``        which identity is the bride, which the groom
``enrich.semantic_tags``     CLIP projection against a text query bank
``enrich.people_cluster``    people-composition key
``enrich.temporal``          usable timeline + artificial-time detection
``enrich.parents``           couple-with-parents portraits
``enrich.ceremony_anchor``   the kiss and the send-off, from one anchor
``enrich.key_pages``         the photos that open and close the album
===========================  =========================================

Each is a candidate for replacement by a better model without touching
anything else, which is the reason for pulling them apart.

Importing this package registers every substage it defines.
"""

from src.pipeline.enrich import (  # noqa: F401  (imported for registration)
    ceremony_anchor,
    classification,
    hygiene,
    identities,
    key_pages,
    temporal,
)

__all__ = ["ceremony_anchor", "classification", "hygiene", "identities",
           "key_pages", "temporal", "timeline"]
