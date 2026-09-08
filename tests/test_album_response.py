"""Tests for the reply payload — one album or several.

Phase 4 of `docs/multi_album_plan.md`. Two things are load-bearing and neither
is visible from the outside:

* a single album's payload must not change shape at all, or every existing
  consumer is affected by a feature it did not ask for;
* the payload must fit an Azure queue message, which has never been checked --
  today an oversized reply fails at the queue with nothing in our logs.

    python -m pytest tests/test_album_response.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.album_response import (  # noqa: E402
    QUEUE_MESSAGE_LIMIT, SAFETY_MARGIN, album_entry, build_reply, combine,
    encode, encoded_size, fit_to_limit)


def composition(spreads: int = 3, placements: int = 8, filler: str = "x"):
    """A composition shaped like `assembly_output`'s, sized to order."""
    return {
        "userJobId": 1360313992,
        "compositionPackageId": -1,
        "productId": 919,
        "projectId": 52711892,
        "compositions": [
            {"compositionId": i, "designId": 1518200 + i, "boxes": None,
             "styleId": 0, "copies": 1, "revisionCounter": 0}
            for i in range(spreads)
        ],
        "placementsImg": [
            {"placementImgId": i, "compositionId": i % max(spreads, 1),
             "boxId": 11840 + i, "photoId": 12146670441 + i,
             "cropX": 0.0, "cropY": 0.33, "cropWidth": 1.0, "cropHeight": 0.34,
             "rotate": 0, "photoFilter": 0, "photo": None, "pad": filler * 64}
            for i in range(placements)
        ],
        "placementsTxt": [],
    }


def album_doc(spreads: int = 3, placements: int = 8, error=None, filler="x"):
    return {"requestId": "AAD_52711892_P.test", "error": error,
            "composition": composition(spreads, placements, filler)}


# -- one album must not change ------------------------------------------


def test_a_single_album_is_returned_untouched():
    """The ordinary payload has to stay exactly what it was: no plural key, no
    renaming, the same object."""
    doc = album_doc()

    reply = combine([doc])

    assert reply is doc
    assert "albums" not in reply
    assert "albumsOmitted" not in reply


def test_no_albums_at_all_gives_nothing_to_send():
    assert combine([]) is None
    assert combine([None, None]) is None


def test_a_failed_album_is_skipped_not_sent_as_a_hole():
    reply = combine([None, album_doc()])

    assert reply is not None
    assert "albums" not in reply, "one surviving album is a single-album reply"


# -- several albums -----------------------------------------------------


def test_composition_still_holds_the_first_album():
    """A consumer that has never heard of `albums` must still get an album."""
    first, second = album_doc(spreads=3), album_doc(spreads=9)

    reply = combine([first, second])

    assert reply["composition"] == first["composition"]
    assert reply["composition"] != second["composition"]


def test_albums_carries_every_album_in_order():
    reply = combine([album_doc(spreads=2), album_doc(spreads=5)],
                    ["brideAndGroom", "parents"])

    assert [a["albumIndex"] for a in reply["albums"]] == [0, 1]
    assert [a["variant"] for a in reply["albums"]] == ["brideAndGroom", "parents"]
    assert len(reply["albums"][0]["composition"]["compositions"]) == 2
    assert len(reply["albums"][1]["composition"]["compositions"]) == 5


def test_the_request_id_is_not_repeated_per_album():
    """It identifies the request, and every album shares it."""
    reply = combine([album_doc(), album_doc()])

    assert "requestId" in reply
    assert all("requestId" not in entry for entry in reply["albums"])


def test_an_albums_own_error_travels_with_it():
    reply = combine([album_doc(), album_doc(error="layout failed")])

    assert reply["albums"][0].get("error") is None
    assert reply["albums"][1]["error"] == "layout failed"


def test_a_variant_name_is_optional():
    reply = combine([album_doc(), album_doc()])

    assert all("variant" not in entry for entry in reply["albums"])


def test_album_ids_are_only_unique_within_an_album():
    """Why an album is the unit and `compositions` is not extended:
    `compositionId` and `placementImgId` restart at 0 per album, so merging the
    lists would make `placementsImg` ambiguous about which album a placement
    belongs to."""
    reply = combine([album_doc(spreads=3), album_doc(spreads=3)])

    ids = [[c["compositionId"] for c in entry["composition"]["compositions"]]
           for entry in reply["albums"]]
    assert ids[0] == ids[1] == [0, 1, 2], (
        "ids repeat across albums, which is exactly why they stay separated")


# -- the size guard -----------------------------------------------------


def test_a_payload_that_fits_is_untouched():
    reply = combine([album_doc(), album_doc()])

    fitted, omitted = fit_to_limit(reply)

    assert omitted == 0
    assert fitted is reply


def test_albums_are_dropped_until_the_payload_fits():
    """Dropping visibly beats failing opaquely at the queue."""
    big = album_doc(spreads=200, placements=900, filler="q")
    reply = combine([big] * 8)
    assert encoded_size(reply) > QUEUE_MESSAGE_LIMIT, "fixture is not big enough"

    fitted, omitted = fit_to_limit(reply)

    assert omitted > 0
    assert len(fitted["albums"]) == 8 - omitted
    assert fitted["albumsOmitted"] == omitted
    assert encoded_size(fitted) <= QUEUE_MESSAGE_LIMIT - SAFETY_MARGIN


def test_the_first_album_always_survives():
    """`composition` is never trimmed, so the request still gets an album."""
    big = album_doc(spreads=400, placements=2000, filler="z")
    reply = combine([big] * 4)

    fitted, _ = fit_to_limit(reply)

    assert fitted["composition"] == big["composition"]
    assert len(fitted["albums"]) >= 1


def test_a_single_oversized_album_is_reported_not_silently_mangled():
    """Nothing can be dropped, so it goes out and the size is logged. Failing
    at the queue with a logged size beats pretending it fit."""
    class Recorder:
        def __init__(self):
            self.errors = []
        def error(self, message):
            self.errors.append(message)
        def warning(self, message):
            pass

    huge = album_doc(spreads=500, placements=4000, filler="w")
    log = Recorder()

    fitted, omitted = fit_to_limit(huge, logger=log)

    assert omitted == 0
    assert fitted is huge
    assert log.errors and "single album" in log.errors[0]


def test_the_guard_measures_the_bytes_that_are_actually_sent():
    """json -> gzip -> base64, the same three steps `push_report_msg` does. A
    guard measuring anything else would pass payloads the queue rejects."""
    reply = combine([album_doc(), album_doc()])

    assert encoded_size(reply) == len(encode(reply))
    assert encode(reply).isascii()


def test_build_reply_combines_and_fits_in_one_step():
    big = album_doc(spreads=200, placements=900, filler="q")

    reply = build_reply([big] * 8, ["a"] * 8)

    assert encoded_size(reply) <= QUEUE_MESSAGE_LIMIT - SAFETY_MARGIN
    assert reply["albumsOmitted"] > 0


def test_two_real_sized_albums_are_nowhere_near_the_limit():
    """The measured case: a 27-spread album is ~4.9 KB encoded, so `autoAlbums`
    has ample headroom. Pinned so a regression in payload size is visible."""
    reply = combine([album_doc(spreads=27, placements=72),
                     album_doc(spreads=27, placements=74)])

    size = encoded_size(reply)
    assert size < QUEUE_MESSAGE_LIMIT // 4, f"two albums encoded to {size} bytes"
