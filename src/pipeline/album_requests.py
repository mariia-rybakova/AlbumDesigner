"""The albums a caller asked for, parsed off the request.

One call, a list of briefs. `aiAutocompose` used to dispatch one message per
product and read the gallery once per album; now it sends the list and the
gallery is read once for all of them.

The list **guides** rather than dictates. Which albums are actually worth
composing is a fact about the gallery -- whether the parents were resolved,
whether there is enough distinct content -- so `enrich.variants` decides, and
answers for every brief it was given: composed, or declined with a reason. It
may also compose an album nobody asked for, carrying `derived_from` to say
which brief it grew out of.

A malformed entry declines *itself* and leaves the rest of the list to run.
That matches the sender's own rule -- "a bad option fails only its own
product" -- and is the difference between one bad brief costing one album and
costing the whole gallery.

Absent or empty means the legacy shape: one album, exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: The request key carrying the list.
ALBUM_REQUESTS = "albumRequests"

#: Where the declines ride back to the report stage. The context's `request`
#: IS the message content dict (`AlbumContext.for_message` passes it straight
#: through), so writing it there is how a fact derived in ENRICH reaches a
#: reply built in ReportStage -- the contexts themselves do not travel.
UNFULFILLED_KEY = "albumsUnfulfilled"

#: What the caller keys its own tracking record on, so it must travel back
#: untouched on whatever the designer composed for it.
ID_KEY = "albumRequestId"


@dataclass(frozen=True)
class AlbumRequest:
    """One brief: what the caller would like, and how to answer it.

    ``focus``, ``photo_ids`` and ``person_ids`` are read out of the brief's own
    ``aiMetadata``, which is the vocabulary the composer already speaks -- it
    maps its product names onto these before sending. Nothing here is the
    composer's *product name*: `logical_name` travels for logging and for the
    reply, and no decision is keyed on it, so a product the composer invents
    tomorrow needs no change here.

    The page and composition counts are recorded and deliberately not applied:
    the designer derives its own spread budget from the gallery, and these are
    the caller's expectation rather than an instruction.
    """

    album_request_id: str
    logical_name: Optional[str] = None
    focus: Optional[Tuple[str, ...]] = None
    photo_ids: Optional[Tuple[int, ...]] = None
    person_ids: Optional[Tuple[int, ...]] = None
    compositions_count: Optional[int] = None
    min_pages: Optional[int] = None
    max_pages: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decline:
    """A brief that will not be composed, and why.

    ``reason`` is a stable token for the caller to branch on; ``detail`` is for
    a human reading the logs. Both, because a caller that can only show a
    sentence cannot tell "we chose not to" from "it broke".
    """

    album_request_id: Optional[str]
    reason: str
    detail: str = ""

    def as_content(self) -> Dict[str, Any]:
        entry = {ID_KEY: self.album_request_id, "reason": self.reason}
        if self.detail:
            entry["detail"] = self.detail
        return entry


def _ids(value) -> Optional[Tuple[int, ...]]:
    if value is None:
        return None
    out = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(out)


def _strings(value) -> Optional[Tuple[str, ...]]:
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    return tuple(str(item) for item in value if str(item).strip())


def _int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse(request: Optional[Dict[str, Any]]
          ) -> Tuple[Tuple[AlbumRequest, ...], Tuple[Decline, ...]]:
    """``(briefs, declines)`` off a request.

    Both empty means the legacy single-album shape, which every caller sends
    today and which must keep behaving exactly as it did.
    """
    raw_list = (request or {}).get(ALBUM_REQUESTS)
    if not raw_list:
        return (), ()

    if not isinstance(raw_list, (list, tuple)):
        return (), (Decline(None, "malformed_request",
                            f"{ALBUM_REQUESTS} is {type(raw_list).__name__}, not a list"),)

    briefs: List[AlbumRequest] = []
    declines: List[Decline] = []
    seen = set()

    for index, entry in enumerate(raw_list):
        if not isinstance(entry, dict):
            declines.append(Decline(None, "malformed_brief",
                                    f"entry {index} is {type(entry).__name__}, not an object"))
            continue

        request_id = entry.get(ID_KEY)
        if request_id is None or not str(request_id).strip():
            declines.append(Decline(None, "missing_id",
                                    f"entry {index} has no {ID_KEY}"))
            continue
        request_id = str(request_id)

        if request_id in seen:
            # Two briefs with one id would make the answer ambiguous: the
            # caller could not tell which of its records either album is for.
            declines.append(Decline(request_id, "duplicate_id",
                                    f"entry {index} repeats {request_id}"))
            continue
        seen.add(request_id)

        hints = entry.get("aiMetadata") or {}
        if not isinstance(hints, dict):
            hints = {}

        briefs.append(AlbumRequest(
            album_request_id=request_id,
            logical_name=entry.get("logicalName"),
            focus=_strings(hints.get("focus")),
            photo_ids=_ids(hints.get("photoIds")),
            person_ids=_ids(hints.get("personIds")),
            compositions_count=_int(entry.get("compositionsCount")),
            min_pages=_int(entry.get("minPages")),
            max_pages=_int(entry.get("maxPages")),
            raw=entry,
        ))

    return tuple(briefs), tuple(declines)
