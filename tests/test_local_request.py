"""Tests for reproducing a production request locally (tools/local_request.py).

The Datadog client and PTFile are faked, so these run offline. The log fixtures
are copied verbatim from real production entries.

    python -m pytest tests/test_local_request.py -v
    python tests/test_local_request.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import local_request as lr  # noqa: E402

PROJECT_ID = 53496523
CONDITION_ID = "AAD_53496523_P.260901-092002.71586c50-6ff0-4155-b996-9affc9b04972.176.967"

SUCCESS_LOG = f"Message was reported to the queue: {PROJECT_ID}/{CONDITION_ID}."

REQUEST_LOG = (
    "Received message: {'replyQueueName': 'aigeneratealbumresponsedto', 'storeId': 4, "
    f"'accountId': 261682, 'projectId': {PROJECT_ID}, 'fulfillerId': 0, 'userId': 763939644, "
    "'userJobId': 1447421490, "
    "'base_url': 'ptstorage_17://pictures/53/496/53496523/7eg0nfck686zpfooqs', "
    "'photos': [12391672742, 12391672750, 12391672760], 'projectCategory': 1, "
    "'compositionPackageId': -1, 'designInfo': None, 'designInfoTempLocation': "
    "'pictures/temp/queues/aigeneratealbumdto/hivihcnejsg4196p05kub19r53.json', "
    "'aiMetadata': {'photoIds': None, 'focus': [], 'personIds': [], 'subjects': [], 'density': 3}, "
    "'rating': [{'photoId': 12391673031, 'rating': 4.0}], 'ratingTempLocation': None, "
    f"'conditionId': '{CONDITION_ID}', "
    "'timedOut': False, 'dependencyDeleted': False, 'retryCount': 0}"
    "/<ptinfra.pt_queue.Message object at 0x7f0fa5469ee0>"
)


def _entry(message, when="2026-09-01T09:20:04Z"):
    return {"attributes": {"message": message, "timestamp": when, "service": "ai"}}


class FakeClient:
    """Stands in for DatadogLogs; routes by which marker the query contains."""

    def __init__(self, success=(), requests=()):
        self.success = list(success)
        self.requests = list(requests)
        self.queries = []

    def search(self, query, frm, to="now", limit=50):
        self.queries.append(query)
        if lr.SUCCESS_MARKER in query:
            return self.success
        if lr.REQUEST_MARKER in query:
            return self.requests
        return []


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_parses_a_real_request_log():
    request = lr.parse_logged_request(REQUEST_LOG)

    assert request["projectId"] == PROJECT_ID
    assert request["base_url"].startswith("ptstorage_17://")
    assert request["photos"] == [12391672742, 12391672750, 12391672760]
    # Python repr literals must survive: None -> None, False -> False, floats.
    assert request["designInfo"] is None
    assert request["timedOut"] is False
    assert request["rating"][0]["rating"] == 4.0
    assert request["aiMetadata"]["photoIds"] is None


def test_request_survives_a_json_round_trip():
    """The saved file must reproduce the request the service would receive."""
    request = lr.parse_logged_request(REQUEST_LOG)
    assert json.loads(json.dumps(request)) == request


def test_truncated_log_is_reported_clearly():
    truncated = REQUEST_LOG[: len(REQUEST_LOG) // 2]
    try:
        lr.parse_logged_request(truncated)
    except lr.TruncatedLogError as exc:
        assert "truncated" in str(exc).lower()
    else:
        raise AssertionError("a truncated payload must raise TruncatedLogError")


def test_success_marker_regex():
    match = lr._SUCCESS_RE.search(SUCCESS_LOG)
    assert match and int(match.group("project_id")) == PROJECT_ID
    assert match.group("condition_id") == CONDITION_ID


# --------------------------------------------------------------------------
# Finding a run
# --------------------------------------------------------------------------


def test_finds_the_newest_successful_run():
    client = FakeClient(success=[_entry(SUCCESS_LOG)], requests=[_entry(REQUEST_LOG)])
    run = lr.find_latest_successful_request(client)

    assert run.project_id == PROJECT_ID
    assert run.condition_id == CONDITION_ID
    assert run.base_url.startswith("ptstorage_17://")
    assert run.photo_ids == [12391672742, 12391672750, 12391672760]
    assert "manual selection" in run.summary()
    # both markers must be scoped to the shared service tag
    assert all(f"service:{lr.SERVICE}" in q for q in client.queries)


def test_skips_a_run_whose_payload_is_truncated():
    """A big gallery's request log overruns Datadog's limit; move on rather
    than fail."""
    other_id = 52337727
    other_condition = "AAD_52337727_P.260901-082543.078ecd1c.102.72"
    truncated = REQUEST_LOG.replace(str(PROJECT_ID), str(other_id))[:200]

    class Routed(FakeClient):
        def search(self, query, frm, to="now", limit=50):
            self.queries.append(query)
            if lr.SUCCESS_MARKER in query:
                return [
                    _entry(f"Message was reported to the queue: {other_id}/{other_condition}."),
                    _entry(SUCCESS_LOG),
                ]
            return [_entry(truncated)] if str(other_id) in query else [_entry(REQUEST_LOG)]

    run = lr.find_latest_successful_request(Routed())
    assert run.project_id == PROJECT_ID, "should fall through to the intact run"


def test_prefers_the_request_matching_the_condition_id():
    """A project is often re-run; the payload must be the one for this run."""
    stale = REQUEST_LOG.replace(CONDITION_ID, "AAD_53496523_P.260901-000000.stale.1.1")
    stale = stale.replace("12391672742, 12391672750, 12391672760", "999")

    client = FakeClient(success=[_entry(SUCCESS_LOG)],
                        requests=[_entry(stale), _entry(REQUEST_LOG)])
    run = lr.find_latest_successful_request(client)
    assert run.photo_ids == [12391672742, 12391672750, 12391672760]


def test_no_runs_raises():
    try:
        lr.find_latest_successful_request(FakeClient())
    except lr.DatadogError as exc:
        assert "No successful album runs" in str(exc)
    else:
        raise AssertionError("expected DatadogError")


def test_missing_credentials_points_at_the_alternative():
    with mock.patch.dict(os.environ, {"DD_API_KEY": "", "DD_APP_KEY": ""}, clear=False):
        os.environ.pop("DD_API_KEY", None)
        os.environ.pop("DD_APP_KEY", None)
        try:
            lr.DatadogLogs()
        except lr.DatadogError as exc:
            assert "DD_API_KEY" in str(exc) and "--request" in str(exc)
        else:
            raise AssertionError("expected DatadogError")


# --------------------------------------------------------------------------
# Saving / loading
# --------------------------------------------------------------------------


def test_save_and_load_round_trip():
    request = lr.parse_logged_request(REQUEST_LOG)
    with tempfile.TemporaryDirectory() as tmp:
        path = lr.save_request(request, "unit-test", requests_dir=tmp)
        assert os.path.isfile(path)
        with open(path, encoding="utf-8") as handle:
            assert json.load(handle) == request


def test_request_path_accepts_name_or_path():
    assert lr.request_path("request0").endswith(os.path.join("test_requests", "request0.json"))
    assert lr.request_path("a/b/c.json") == "a/b/c.json"
    assert lr.request_path("c.json") == "c.json"


def test_the_seeded_production_request_is_loadable():
    """files/test_requests/53496523.json is a real captured request; the
    default saved-request mode must be able to run it."""
    request = lr.load_request("53496523")
    assert request["projectId"] == 53496523
    assert len(request["photos"]) == 44
    assert request["base_url"].startswith("ptstorage_17://")


# --------------------------------------------------------------------------
# Photo download
# --------------------------------------------------------------------------


class FakePhoto:
    def __init__(self, photo_id, filename=None):
        self.photoId = photo_id
        self.filename = filename or f"{photo_id}.jpg"


class FakeScene:
    def __init__(self, photos):
        self.photos = photos


class FakeGallery:
    photos = [FakePhoto(1), FakePhoto(2), FakePhoto(3, "3_v2.jpg")]

    def __init__(self, root_path):
        self.root_path = root_path
        self.scenes = [FakeScene(self.photos)]


class FakePTFile:
    missing = set()

    def __init__(self, url):
        self.url = url

    def read_blob(self):
        name = self.url.rsplit("/", 1)[-1]
        if name in self.missing:
            raise FileNotFoundError(self.url)
        assert "/smallres/" in self.url, f"photos must come from the size folder: {self.url}"
        return b"\xff\xd8jpegbytes"


def _download(dest, **kwargs):
    with mock.patch("ptinfra.utils.gallery.Gallery", FakeGallery), \
         mock.patch("ptinfra.azure.pt_file.PTFile", FakePTFile):
        return lr.download_gallery_photos("ptstorage_17://base", dest, log=lambda *_: None, **kwargs)


def test_downloads_only_the_requested_photos():
    with tempfile.TemporaryDirectory() as tmp:
        counts = _download(tmp, photo_ids=[1, 3])
        assert sorted(os.listdir(tmp)) == ["1.jpg", "3_v2.jpg"]
        assert counts["downloaded"] == 2 and counts["requested"] == 2


def test_downloads_whole_gallery_when_no_ids_given():
    with tempfile.TemporaryDirectory() as tmp:
        counts = _download(tmp, photo_ids=None)
        assert counts["downloaded"] == 3


def test_skips_photos_already_on_disk():
    with tempfile.TemporaryDirectory() as tmp:
        _download(tmp, photo_ids=[1, 2])
        counts = _download(tmp, photo_ids=[1, 2])
        assert counts["skipped"] == 2 and counts["downloaded"] == 0


def test_a_purged_photo_does_not_stop_the_run():
    FakePTFile.missing = {"2.jpg"}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            counts = _download(tmp, photo_ids=[1, 2, 3])
            assert counts["downloaded"] == 2 and counts["missing"] == 1
            assert "2.jpg" not in os.listdir(tmp)
            assert not [f for f in os.listdir(tmp) if f.endswith(".part")]
    finally:
        FakePTFile.missing = set()


def test_reports_ids_absent_from_the_gallery():
    with tempfile.TemporaryDirectory() as tmp:
        counts = _download(tmp, photo_ids=[1, 999])
        assert counts["not_in_gallery"] == 1


def test_max_photos_caps_the_download():
    with tempfile.TemporaryDirectory() as tmp:
        counts = _download(tmp, photo_ids=None, max_photos=2)
        assert counts["downloaded"] == 2


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} local-request tests passed")
