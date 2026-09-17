"""The reply payload, for one album or several.

Phase 4 of `docs/multi_album_plan.md`.

One album has always gone back as::

    {"requestId": ..., "error": ..., "composition": {...}}

and `assembly_output` still builds exactly that per album. Several albums are
carried by *adding* a plural key rather than changing the singular one::

    {"requestId": ..., "error": ...,
     "composition": {...first album...},
     "albums": [{"albumIndex": 0, "variant": "brideAndGroom", "composition": {...}},
                {"albumIndex": 1, "variant": "parents",       "composition": {...}}]}

``composition`` keeps pointing at the first album, so a consumer that has never
heard of ``albums`` still gets a working album out of a multi-album request.
And ``albums`` is absent entirely for a single album, so the common payload is
byte-identical to what it was.

Why not extend the existing ``compositions`` list inside one composition:
``compositionId`` and ``placementImgId`` both restart at 0 on every
`assembly_output` call, so merged albums make ``placementsImg`` ambiguous about
which album a placement belongs to. The ids are only unique *within* an album,
which is why an album is the unit here.

``albumIndex`` and ``variant`` are additive discriminators, and they exist
because of a question this module cannot answer: ``userJobId`` and
``compositionPackageId`` are per **request**, so N albums share them. If the
consumer keys a stored album by ``userJobId``, N albums collide on write no
matter what this payload looks like. That has to be settled with whoever owns
the consumer; meanwhile these two fields give it something to tell the albums
apart with, and nothing existing is renamed or moved.

The size guard
--------------
The payload is gzipped and base64'd into a single Azure Storage queue message,
which caps the encoded string at 64 KiB. There has never been a check: today
`push_report_msg` json-dumps, compresses, encodes and sends, and an oversized
payload fails at the Azure boundary with nothing in our logs to say why.

Measured on a real 27-spread / 72-photo album: 39,108 bytes of JSON, 4,888
encoded -- so roughly 13 albums of that size fit. Two albums are nowhere near
it, but album size varies with spread count, so the limit is enforced rather
than assumed: :func:`fit_to_limit` drops the albums that do not fit, says so in
``albumsOmitted``, and leaves a payload that sends. Dropping visibly beats
failing opaquely, and beats truncating silently.

The proper fix is the offload the *incoming* side already uses --
``designInfoTempLocation`` and ``ratingTempLocation`` put an oversized payload
in blob and pass a location. Doing that outbound needs the consumer to read a
location, so it is not enabled here.
"""

from __future__ import annotations

import base64
import gzip
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Azure Storage queue message cap, on the base64 text that gets sent.
QUEUE_MESSAGE_LIMIT = 64 * 1024

#: Left free for the queue's own overhead, so a payload that just fits our
#: measurement does not fail on theirs.
SAFETY_MARGIN = 2 * 1024


def encode(payload: Dict[str, Any]) -> str:
    """json -> gzip -> base64, exactly as `push_report_msg` sends it.

    Kept here so the size guard measures the same bytes that go on the wire
    rather than an approximation of them.
    """
    raw = json.dumps(payload)
    return base64.b64encode(gzip.compress(raw.encode("ascii"))).decode("ascii")


def encoded_size(payload: Dict[str, Any]) -> int:
    return len(encode(payload))


def album_entry(index: int, doc: Optional[Dict[str, Any]],
                variant: Optional[str] = None,
                album_request_id: Optional[str] = None,
                derived_from: Optional[str] = None) -> Dict[str, Any]:
    """One element of ``albums``.

    ``doc`` is a whole single-album reply as `assembly_output` returns it; only
    its ``composition`` travels, because ``requestId`` is the same for every
    album and ``error`` is carried per album in its own field.

    ``albumRequestId`` is the brief this album answers, echoed exactly as it
    arrived -- the caller keys its own tracking record on it, and a reply it
    cannot match is a reply it drops. ``derivedFrom`` is set instead on an
    album nobody asked for, naming the brief it grew out of, so an extra album
    is still traceable to a request even though it answers none.
    """
    entry: Dict[str, Any] = {"albumIndex": index}
    if variant:
        entry["variant"] = variant
    if album_request_id:
        entry["albumRequestId"] = album_request_id
    if derived_from:
        entry["derivedFrom"] = derived_from
    entry["composition"] = (doc or {}).get("composition")
    error = (doc or {}).get("error")
    if error is not None:
        entry["error"] = str(error)
    return entry


def combine(album_docs: Sequence[Optional[Dict[str, Any]]],
            variants: Optional[Sequence[Optional[str]]] = None,
            album_request_ids: Optional[Sequence[Optional[str]]] = None,
            derived_from: Optional[Sequence[Optional[str]]] = None,
            unfulfilled: Optional[Sequence[Dict[str, Any]]] = None
            ) -> Optional[Dict[str, Any]]:
    """One reply for however many albums were composed.

    A single album returns its own doc untouched -- no ``albums`` key, nothing
    renamed -- so the ordinary payload does not change shape at all. The one
    exception is ``unfulfilled``: a caller that sent a list of briefs is owed
    an answer for each of them even when only one album came back, and even
    when none did.

    ``unfulfilled`` is kept separate from ``albumsOmitted`` on purpose. Omitted
    means the album exists and did not fit the queue message, which is worth
    retrying; unfulfilled means it was never composed and retrying changes
    nothing. Collapsing them would tell the caller to retry what cannot
    succeed.
    """
    docs = [doc for doc in album_docs if doc is not None]
    if not docs:
        return None

    refused = [dict(item) for item in (unfulfilled or [])]

    if len(docs) == 1 and not refused:
        return docs[0]

    names = list(variants or [])
    ids = list(album_request_ids or [])
    derived = list(derived_from or [])
    payload = dict(docs[0])

    def at(seq, index):
        return seq[index] if index < len(seq) else None

    if len(docs) > 1 or ids or derived:
        payload["albums"] = [
            album_entry(index, doc, at(names, index), at(ids, index), at(derived, index))
            for index, doc in enumerate(docs)
        ]
    if refused:
        payload["unfulfilled"] = refused
    return payload


def fit_to_limit(payload: Dict[str, Any], limit: int = QUEUE_MESSAGE_LIMIT,
                 margin: int = SAFETY_MARGIN, logger=None
                 ) -> Tuple[Dict[str, Any], int]:
    """Drop trailing albums until the encoded payload fits. Returns (payload, omitted).

    Only ``albums`` is ever shortened; ``composition`` is left alone, so the
    first album survives whatever happens. A single-album payload that is
    somehow too big is returned unchanged -- there is nothing to drop, and
    failing at the queue with a logged size beats pretending it fit.
    """
    budget = max(0, limit - margin)
    size = encoded_size(payload)
    if size <= budget:
        return payload, 0

    albums: List[Dict[str, Any]] = list(payload.get("albums") or [])
    if not albums:
        if logger:
            logger.error(
                f"Reply is {size} bytes encoded, over the {budget} byte budget, "
                f"and has a single album -- nothing can be dropped")
        return payload, 0

    omitted = 0
    trimmed = dict(payload)
    while len(albums) > 1 and size > budget:
        albums = albums[:-1]
        omitted += 1
        trimmed["albums"] = albums
        trimmed["albumsOmitted"] = omitted
        size = encoded_size(trimmed)

    if logger:
        logger.warning(
            f"Reply exceeded the queue budget; dropped {omitted} album(s), "
            f"{len(albums)} sent, {size} bytes encoded of {budget}")
    return trimmed, omitted


def build_reply(album_docs: Sequence[Optional[Dict[str, Any]]],
                variants: Optional[Sequence[Optional[str]]] = None,
                album_request_ids: Optional[Sequence[Optional[str]]] = None,
                derived_from: Optional[Sequence[Optional[str]]] = None,
                unfulfilled: Optional[Sequence[Dict[str, Any]]] = None,
                logger=None) -> Optional[Dict[str, Any]]:
    """The reply to send: combined, then trimmed to what the queue accepts."""
    payload = combine(album_docs, variants, album_request_ids, derived_from, unfulfilled)
    if payload is None:
        return None
    payload, _ = fit_to_limit(payload, logger=logger)
    return payload
