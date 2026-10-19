"""Turning an outside format into the album's own tables.

Everything that reads something the service did not write -- protobuf blobs,
the printlab's design JSON, a pickled query bank -- and hands back a table the
rest of the code can use:

* `protos`           the gallery's protobuf blobs -> the photo table
* `design_templates` the printlab's design JSON -> the layout and box tables
* `query_bank`       photo embeddings matched against the query bank
* `proto_labels`     what the blobs' coded fields mean

These are libraries, not substages: `src/pipeline/ingest` holds the substages
that call them, and every file there carries a `@register`. The split keeps
that package readable -- a substage is a step in the pipeline, a reader is a
format it knows how to read.

Importing this package has no side effects on purpose. Import the module you
want; nothing is re-exported here.
"""
