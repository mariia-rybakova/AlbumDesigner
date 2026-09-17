"""One call, a list of briefs, an answer for every one of them.

The composer used to dispatch a message per product and pay for the gallery
read each time. Now it sends the list and the designer decides: the list
guides, the gallery rules. What must hold is that no brief disappears -- every
one comes back composed or declined by name -- because the caller keys its own
tracking record on that id and cannot match what it never hears about.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.album_response import build_reply, combine  # noqa: E402
from src.pipeline import album_requests  # noqa: E402
from src.pipeline.contracts import AlbumContext, Col, GalleryFacts  # noqa: E402
from src.pipeline.enrich.variants import AS_REQUESTED, VariantsSubStage  # noqa: E402


class _Quiet:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass
    def debug(self, *a, **k): pass


_QUIET = _Quiet()


def brief(album_request_id, logical_name=None, focus=None, **extra):
    entry = {"albumRequestId": album_request_id}
    if logical_name:
        entry["logicalName"] = logical_name
    if focus is not None:
        entry["aiMetadata"] = {"focus": list(focus)}
    entry.update(extra)
    return entry


def plan(request, facts=None):
    """`(variants, declines)` for a request, as `enrich.variants` decides."""
    context = AlbumContext(
        photos=pd.DataFrame({Col.IMAGE_ID: [1, 2, 3]}),
        request=request,
        logger=_QUIET,
    )
    if facts is not None:
        context.facts = facts
    VariantsSubStage()(context)
    return context.variants, context.unfulfilled


def with_parents():
    facts = GalleryFacts()
    facts.bride_parents = (11,)
    facts.groom_parents = (12,)
    return facts


# -- the contract ---------------------------------------------------------

def test_no_list_is_the_legacy_shape():
    """Every caller sends this today; it must not move."""
    briefs, declines = album_requests.parse({'projectId': 1})

    assert briefs == () and declines == ()


def test_a_brief_carries_the_objective_not_the_product_name():
    briefs, declines = album_requests.parse({'albumRequests': [
        brief("AACP_1#0", "wedding_main_album"),
        brief("AACP_1#1", "wedding_parents_album", focus=["parents"]),
    ]})

    assert declines == ()
    assert [b.album_request_id for b in briefs] == ["AACP_1#0", "AACP_1#1"]
    assert briefs[0].focus is None, "no focus override is 'the full story'"
    assert briefs[1].focus == ("parents",)
    assert briefs[1].logical_name == "wedding_parents_album"


def test_a_bad_brief_costs_only_itself():
    """The sender's own rule: a bad option fails that product, not the job."""
    briefs, declines = album_requests.parse({'albumRequests': [
        brief("AACP_1#0"),
        {"logicalName": "no_id_here"},
        "not an object",
        brief("AACP_1#0"),
    ]})

    assert [b.album_request_id for b in briefs] == ["AACP_1#0"]
    assert [d.reason for d in declines] == ["missing_id", "malformed_brief", "duplicate_id"]


def test_page_counts_are_recorded_and_not_obeyed():
    briefs, _ = album_requests.parse({'albumRequests': [
        brief("AACP_1#0", compositionsCount=20, minPages=20, maxPages=20)]})

    assert briefs[0].compositions_count == 20 and briefs[0].min_pages == 20
    # The designer derives its own budget; nothing here overrides the request.
    assert briefs[0].focus is None


# -- the decision ---------------------------------------------------------

def test_every_brief_becomes_an_album_when_the_gallery_can_meet_it():
    variants, declines = plan({'albumRequests': [
        brief("AACP_1#0", "wedding_main_album"),
        brief("AACP_1#1", "wedding_couple_album", focus=["brideAndGroom"]),
    ]})

    assert declines == ()
    assert [v.fulfils for v in variants] == ["AACP_1#0", "AACP_1#1"]
    assert [v.name for v in variants] == ["wedding_main_album", "wedding_couple_album"]
    assert variants[1].focus == ("brideAndGroom",)


def test_a_parents_album_is_declined_when_there_are_no_parents():
    """Composed anyway it is a second couple album wearing the wrong label."""
    variants, declines = plan({'albumRequests': [
        brief("AACP_1#0", "wedding_main_album"),
        brief("AACP_1#1", "wedding_parents_album", focus=["parents"]),
    ]}, facts=GalleryFacts())

    assert [v.fulfils for v in variants] == ["AACP_1#0"], "the other album still runs"
    assert [(d.album_request_id, d.reason) for d in declines] == [
        ("AACP_1#1", "no_parents_resolved")]
    assert declines[0].detail, "a reason the caller can show a human"


def test_a_parents_album_is_composed_when_parents_were_resolved():
    variants, declines = plan({'albumRequests': [
        brief("AACP_1#1", "wedding_parents_album", focus=["parents"])]},
        facts=with_parents())

    assert declines == ()
    assert [v.fulfils for v in variants] == ["AACP_1#1"]


def test_declining_everything_still_returns_an_album():
    """An empty reply is worse than one album plus the reasons."""
    variants, declines = plan({'albumRequests': [
        brief("AACP_1#0", "wedding_parents_album", focus=["parents"])]},
        facts=GalleryFacts())

    assert variants == [AS_REQUESTED]
    assert [d.reason for d in declines] == ["no_parents_resolved"]


def test_the_declines_ride_back_on_the_request():
    """ReportStage builds the reply and the contexts never reach it."""
    request = {'albumRequests': [
        brief("AACP_1#1", "wedding_parents_album", focus=["parents"])]}
    plan(request, facts=GalleryFacts())

    assert request[album_requests.UNFULFILLED_KEY] == [
        {"albumRequestId": "AACP_1#1", "reason": "no_parents_resolved",
         "detail": request[album_requests.UNFULFILLED_KEY][0]["detail"]}]


# -- the answer -----------------------------------------------------------

def _doc(n):
    return {"requestId": "AACP_1", "error": None, "composition": {"n": n}}


def test_each_album_says_which_brief_it_answers():
    payload = combine([_doc(0), _doc(1)],
                      variants=["wedding_main_album", "wedding_parents_album"],
                      album_request_ids=["AACP_1#0", "AACP_1#1"])

    assert [a["albumRequestId"] for a in payload["albums"]] == ["AACP_1#0", "AACP_1#1"]


def test_an_album_nobody_asked_for_names_the_brief_it_grew_from():
    payload = combine([_doc(0), _doc(1)],
                      variants=["wedding_main_album", "wedding_main_album#2"],
                      album_request_ids=["AACP_1#0", None],
                      derived_from=[None, "AACP_1#0"])

    assert "albumRequestId" not in payload["albums"][1]
    assert payload["albums"][1]["derivedFrom"] == "AACP_1#0"


def test_declines_travel_even_when_a_single_album_came_back():
    payload = combine([_doc(0)],
                      album_request_ids=["AACP_1#0"],
                      unfulfilled=[{"albumRequestId": "AACP_1#1",
                                    "reason": "no_parents_resolved"}])

    assert payload["unfulfilled"] == [{"albumRequestId": "AACP_1#1",
                                       "reason": "no_parents_resolved"}]
    assert payload["albums"][0]["albumRequestId"] == "AACP_1#0"


def test_one_album_and_nothing_refused_is_the_payload_it_always_was():
    """The common case must not grow a key."""
    assert combine([_doc(0)]) == _doc(0)
    assert build_reply([_doc(0)]) == _doc(0)


def test_unfulfilled_is_not_albums_omitted():
    """Omitted is worth a retry; unfulfilled never is. They cannot share a key."""
    payload = build_reply([_doc(0)], unfulfilled=[{"albumRequestId": "x",
                                                   "reason": "no_parents_resolved"}])

    assert "unfulfilled" in payload
    assert payload.get("albumsOmitted") in (None, 0)


# -- one album's failure is one album's --------------------------------------
#
# One queue message now carries every album of a gallery, so an exception
# composing one of them could take its siblings with it -- the failure shape
# that cost 53009168 two albums for one bad group, scaled up to the whole
# request.

def test_a_lost_album_is_declined_rather_than_silent():
    """An album that vanishes from `albums` leaves its brief unanswered, and
    the caller cannot tell a product that failed from one we forgot."""
    lead = {}

    album_requests.record_failure(lead, "AACP_1#1", "Error processing stage: boom")

    assert lead[album_requests.UNFULFILLED_KEY] == [
        {"albumRequestId": "AACP_1#1", "reason": "error",
         "detail": "Error processing stage: boom"}]


def test_a_lost_album_joins_the_briefs_the_gallery_declined():
    """Both are answers to a brief, and both travel on the same key."""
    lead = {album_requests.UNFULFILLED_KEY: [
        {"albumRequestId": "AACP_1#2", "reason": "no_parents_resolved"}]}

    album_requests.record_failure(lead, "AACP_1#1", "boom")

    assert [d["reason"] for d in lead[album_requests.UNFULFILLED_KEY]] == [
        "no_parents_resolved", "error"]


def test_an_album_that_answers_no_brief_still_reports_its_failure():
    """The legacy path has no `albumRequestId`; the reply is still owed one."""
    lead = {}

    album_requests.record_failure(lead, None, "boom")

    assert lead[album_requests.UNFULFILLED_KEY][0]["albumRequestId"] is None
    assert lead[album_requests.UNFULFILLED_KEY][0]["reason"] == "error"


# -- how many albums a wedding is worth --------------------------------------
#
# The caller sends a brief per product it would like; how many of those get
# composed is a product decision, and it lives in `CONFIGS['album_policy']` so
# that changing it is an edit rather than a change to the planner.

def wedding():
    facts = GalleryFacts()
    facts.is_wedding = True
    return facts


def wedding_with_parents():
    facts = wedding()
    facts.bride_parents = (11,)
    facts.groom_parents = (12,)
    return facts


def test_a_wedding_composes_one_album_today():
    variants, declines = plan({'albumRequests': [
        brief("AACP_1#0", "wedding_main_album"),
        brief("AACP_1#1", "wedding_parents_album", focus=["parents"]),
    ]}, facts=wedding_with_parents())

    assert [v.fulfils for v in variants] == ["AACP_1#0"]
    assert [(d.album_request_id, d.reason) for d in declines] == [
        ("AACP_1#1", "focus_overridden")]


def test_the_parents_album_is_refused_even_where_it_could_be_composed():
    """The parents were resolved, so the requirement is met and this is purely
    the policy talking. Refused and not quietly turned into a couple album:
    the two are indistinguishable at the caller, and only one was asked for."""
    variants, declines = plan({'albumRequests': [
        brief("AACP_1#1", "wedding_parents_album", focus=["parents"])]},
        facts=wedding_with_parents())

    assert [d.reason for d in declines] == ["focus_overridden"]
    # Something still comes back -- the album the request itself asked for.
    assert variants == [AS_REQUESTED]


def test_a_third_wedding_album_is_refused_by_the_limit_not_the_focus():
    """Two reasons the caller can tell apart: we do not make that kind of
    album, and we are not making that many."""
    variants, declines = plan({'albumRequests': [
        brief("AACP_1#0", "wedding_main_album"),
        brief("AACP_1#1", "wedding_couple_album", focus=["brideAndGroom"]),
    ]}, facts=wedding())

    assert [v.fulfils for v in variants] == ["AACP_1#0"]
    assert [(d.album_request_id, d.reason) for d in declines] == [
        ("AACP_1#1", "album_limit")]


def test_the_policy_is_keyed_on_the_gallery_not_the_request():
    """A non-wedding gallery composes what it was asked for."""
    variants, declines = plan({'albumRequests': [
        brief("AACP_1#0", "family_album"),
        brief("AACP_1#1", "event_album"),
    ]}, facts=GalleryFacts())

    assert [v.fulfils for v in variants] == ["AACP_1#0", "AACP_1#1"]
    assert declines == ()


def test_the_wedding_limit_is_config_and_moves_with_it():
    from utils.configs import CONFIGS

    original = CONFIGS['album_policy']
    CONFIGS['album_policy'] = {**original,
                               'wedding': {'max_albums': 2,
                                           'overridden_focus': ()}}
    try:
        variants, declines = plan({'albumRequests': [
            brief("AACP_1#0", "wedding_main_album"),
            brief("AACP_1#1", "wedding_parents_album", focus=["parents"]),
        ]}, facts=wedding_with_parents())
    finally:
        CONFIGS['album_policy'] = original

    assert [v.fulfils for v in variants] == ["AACP_1#0", "AACP_1#1"]
    assert declines == ()
