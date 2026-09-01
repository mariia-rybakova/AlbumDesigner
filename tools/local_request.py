"""Reproduce a production album request locally.

Three steps, each usable on its own:

1. **Find** the newest request the service processed successfully, by pairing
   two log lines in Datadog (:func:`find_latest_successful_request`).
2. **Save** it as a request JSON under ``files/test_requests/`` so the run is
   repeatable without Datadog (:func:`save_request`).
3. **Download** that gallery's photos into the local input folder the PDF
   visualiser reads from (:func:`download_gallery_photos`).

`process_gallery.py --from-datadog` wires the three together.

Datadog access needs ``DD_API_KEY`` and ``DD_APP_KEY`` in the environment
(``DD_SITE`` defaults to the org's ``us3.datadoghq.com``); the photo download
needs the Azure network, i.e. VPN plus a ``ptinfra.intialize`` call.
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests

DEFAULT_SITE = "us3.datadoghq.com"
SEARCH_PATH = "/api/v2/logs/events/search"

#: Emitted by ReportStage.report_one_message only on the success path, so it is
#: the marker for "this request produced an album".
SUCCESS_MARKER = "Message was reported to the queue"
#: Emitted by read_messages for every incoming request, carrying the full
#: payload as a Python repr.
REQUEST_MARKER = "Received message"

#: The service tag is shared by every pic-time AI service, so both markers are
#: needed to isolate AlbumDesigner.
SERVICE = "ai"

_SUCCESS_RE = re.compile(
    r"Message was reported to the queue:\s*(?P<project_id>\d+)\s*/\s*(?P<condition_id>\S+?)\.?\s*$"
)
_REQUEST_PREFIX = "Received message: "
#: read_messages logs '{content}/{message}'; strip the message repr back off.
_REQUEST_SUFFIX_RE = re.compile(r"/<[\w.]+ object at 0x[0-9a-fA-F]+>\s*$")

#: Photos live under this size folder next to the gallery manifest.
PHOTO_SIZE_FOLDER = "smallres"


class DatadogError(RuntimeError):
    pass


class TruncatedLogError(RuntimeError):
    """The logged payload was cut off by Datadog's per-entry size limit.

    Large galleries log thousands of photo ids and ratings and overrun it. The
    request cannot be recovered from that log line — pick a smaller gallery, or
    supply the request JSON by hand.
    """


@dataclass
class SuccessfulRun:
    """One request the service completed, as reconstructed from its logs."""

    project_id: int
    condition_id: str
    reported_at: datetime
    request: Dict[str, Any] = field(default_factory=dict)
    received_at: Optional[datetime] = None

    @property
    def base_url(self) -> Optional[str]:
        return self.request.get("base_url")

    @property
    def photo_ids(self) -> List[int]:
        return list(self.request.get("photos") or [])

    def summary(self) -> str:
        hints = self.request.get("aiMetadata") or {}
        mode = "manual" if hints.get("photoIds") is None else "ai"
        return (
            f"project {self.project_id}  reported {self.reported_at:%Y-%m-%d %H:%M:%S}Z  "
            f"{len(self.photo_ids)} photos  {mode} selection  density={hints.get('density')}"
        )


# --------------------------------------------------------------------------
# Datadog
# --------------------------------------------------------------------------


class DatadogLogs:
    """Thin wrapper over the Logs API v2 search endpoint."""

    def __init__(self, api_key: str = None, app_key: str = None, site: str = None):
        self.api_key = api_key or os.environ.get("DD_API_KEY")
        self.app_key = app_key or os.environ.get("DD_APP_KEY")
        self.site = site or os.environ.get("DD_SITE", DEFAULT_SITE)

        if not self.api_key or not self.app_key:
            raise DatadogError(
                "Datadog needs DD_API_KEY and DD_APP_KEY in the environment "
                "(DD_SITE defaults to {}). Without them, run a saved request "
                "instead: process_gallery.py <in> <out> --request <name>".format(DEFAULT_SITE)
            )

    def search(self, query: str, frm: str, to: str = "now", limit: int = 50) -> List[dict]:
        """Newest-first log entries matching `query`."""
        response = requests.post(
            f"https://api.{self.site}{SEARCH_PATH}",
            headers={
                "DD-API-KEY": self.api_key,
                "DD-APPLICATION-KEY": self.app_key,
                "Content-Type": "application/json",
            },
            json={
                "filter": {"query": query, "from": frm, "to": to},
                "sort": "-timestamp",
                "page": {"limit": limit},
            },
            timeout=60,
        )
        if response.status_code != 200:
            raise DatadogError(
                f"Datadog search failed ({response.status_code}): {response.text[:400]}"
            )
        return response.json().get("data", [])


def _attributes(entry: dict) -> dict:
    return entry.get("attributes", {}) or {}


def _message_of(entry: dict) -> str:
    return _attributes(entry).get("message", "") or ""


def _timestamp_of(entry: dict) -> Optional[datetime]:
    raw = _attributes(entry).get("timestamp")
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(raw / 1000, tz=timezone.utc)
    return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))


def parse_logged_request(message: str) -> Dict[str, Any]:
    """Recover the request dict from a 'Received message: ...' log line.

    The payload is logged as a Python repr (single quotes, ``None``, ``False``),
    not JSON, so it is read with :func:`ast.literal_eval`.
    """
    body = message.split(_REQUEST_PREFIX, 1)[-1].strip()
    body = _REQUEST_SUFFIX_RE.sub("", body).strip()

    try:
        request = ast.literal_eval(body)
    except (SyntaxError, ValueError) as exc:
        raise TruncatedLogError(
            "Could not parse the logged request — it is almost certainly "
            "truncated by Datadog's log size limit ({} chars retrieved). "
            "Try a gallery with fewer photos, or pass the request JSON "
            "directly with --request. Parse error: {}".format(len(body), exc)
        ) from exc

    if not isinstance(request, dict):
        raise TruncatedLogError(f"Logged payload is a {type(request).__name__}, not a request dict")
    return request


def find_latest_successful_request(
    client: "DatadogLogs",
    lookback_hours: int = 48,
    project_id: Optional[int] = None,
    max_candidates: int = 25,
) -> SuccessfulRun:
    """Newest request that produced an album, with its full payload.

    Walks successful runs newest-first and returns the first whose request log
    is intact — a big gallery's payload is often truncated, and skipping it is
    better than failing.
    """
    frm = f"now-{lookback_hours}h"

    success_query = f'service:{SERVICE} "{SUCCESS_MARKER}"'
    if project_id is not None:
        success_query += f' "{project_id}"'

    candidates = client.search(success_query, frm=frm, limit=max_candidates)
    if not candidates:
        raise DatadogError(
            f"No successful album runs in the last {lookback_hours}h"
            + (f" for project {project_id}" if project_id else "")
        )

    problems = []
    for entry in candidates:
        match = _SUCCESS_RE.search(_message_of(entry))
        if not match:
            continue

        found_id = int(match.group("project_id"))
        if project_id is not None and found_id != project_id:
            continue

        reported_at = _timestamp_of(entry) or datetime.now(timezone.utc)
        run = SuccessfulRun(
            project_id=found_id,
            condition_id=match.group("condition_id"),
            reported_at=reported_at,
        )

        try:
            _attach_request(client, run)
        except (TruncatedLogError, DatadogError) as exc:
            problems.append(f"  project {found_id}: {exc}")
            continue
        return run

    raise DatadogError(
        "Found successful runs but could not recover any request payload:\n"
        + "\n".join(problems[:5])
    )


def _attach_request(client: DatadogLogs, run: SuccessfulRun, window_minutes: int = 90) -> None:
    """Find the 'Received message' log that belongs to this run."""
    start = run.reported_at - timedelta(minutes=window_minutes)
    entries = client.search(
        f'service:{SERVICE} "{REQUEST_MARKER}" "{run.project_id}"',
        frm=start.isoformat().replace("+00:00", "Z"),
        to=run.reported_at.isoformat().replace("+00:00", "Z"),
        limit=10,
    )
    if not entries:
        raise DatadogError(
            f"no '{REQUEST_MARKER}' log within {window_minutes} min before the report"
        )

    # Newest first; prefer the one whose conditionId matches this exact run,
    # since a project is often re-run several times.
    for entry in entries:
        message = _message_of(entry)
        if run.condition_id and run.condition_id not in message:
            continue
        run.request = parse_logged_request(message)
        run.received_at = _timestamp_of(entry)
        return

    run.request = parse_logged_request(_message_of(entries[0]))
    run.received_at = _timestamp_of(entries[0])


# --------------------------------------------------------------------------
# Saving / loading requests
# --------------------------------------------------------------------------


REQUESTS_DIR = os.path.join("files", "test_requests")


def request_path(name: str) -> str:
    """Resolve a request name or path to a file path."""
    if os.path.sep in name or name.endswith(".json"):
        return name
    return os.path.join(REQUESTS_DIR, f"{name}.json")


def save_request(request: Dict[str, Any], name: str, requests_dir: str = REQUESTS_DIR) -> str:
    os.makedirs(requests_dir, exist_ok=True)
    path = os.path.join(requests_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(request, handle, indent=2)
    return path


def load_request(name: str) -> Dict[str, Any]:
    with open(request_path(name), "r", encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------
# Photos
# --------------------------------------------------------------------------


def download_gallery_photos(
    base_url: str,
    dest_dir: str,
    photo_ids=None,
    size_folder: str = PHOTO_SIZE_FOLDER,
    max_photos: Optional[int] = None,
    log=print,
) -> Dict[str, int]:
    """Download a gallery's photos into ``dest_dir``.

    Files are named exactly as the gallery manifest names them
    (``<photoId>.jpg``, or ``<photoId>_v<n>.jpg`` for revisions), which is what
    the PDF visualiser matches on.

    ``photo_ids`` restricts the download to the request's own photos; pass None
    to take the whole gallery. Already-present files are skipped, so re-runs are
    cheap.
    """
    from ptinfra.azure.pt_file import PTFile
    from ptinfra.utils.gallery import Gallery

    os.makedirs(dest_dir, exist_ok=True)

    gallery = Gallery(base_url)
    wanted = {int(p) for p in photo_ids} if photo_ids else None

    photos = [
        photo
        for scene in gallery.scenes
        for photo in scene.photos
        if wanted is None or int(photo.photoId) in wanted
    ]
    if max_photos is not None:
        photos = photos[:max_photos]

    counts = {"downloaded": 0, "skipped": 0, "missing": 0, "requested": len(photos)}
    if wanted:
        found = {int(p.photoId) for p in photos}
        counts["not_in_gallery"] = len(wanted - found)

    for photo in photos:
        dest = os.path.join(dest_dir, photo.filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            counts["skipped"] += 1
            continue
        try:
            data = PTFile(f"{base_url}/{size_folder}/{photo.filename}").read_blob()
        except Exception as exc:  # noqa: BLE001 - a purged photo must not stop the run
            log(f"  ! {photo.filename}: {type(exc).__name__}: {exc}")
            counts["missing"] += 1
            continue
        tmp = dest + ".part"
        with open(tmp, "wb") as handle:
            handle.write(data)
        os.replace(tmp, dest)
        counts["downloaded"] += 1

    return counts
