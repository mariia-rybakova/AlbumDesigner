"""Unified data-transfer contracts for the AlbumDesigner substage pipeline.

Everything that moves between substages moves inside :class:`AlbumContext`.
A substage reads the fields it declared in ``requires``, writes the fields it
declared in ``provides``, and returns the same context object. Nothing else is
shared, so any single substage can be swapped for a different implementation as
long as it honours the same declarations.

Three kinds of state live on the context:

``photos``
    The canonical photo table (one row per image). Column names are constants on
    :class:`Col` so substages never spell them inline and a rename is a single
    edit here.

Typed sidecars (``hints``, ``facts``, ``designs``, ``selection``, ...)
    Small dataclasses for everything that is not per-photo.

``message``
    The transport object from ``ptinfra``. It is the *boundary* with the stages
    that have not been decomposed yet (ProcessStage / ReportStage), which still
    read ``message.content[...]``. :meth:`AlbumContext.sync_to_message` pushes
    context state back onto it. New substages should read and write the typed
    fields, never ``message`` directly — when Process/Report are migrated this
    field disappears and nothing else has to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


# --------------------------------------------------------------------------
# Photo-table column registry
# --------------------------------------------------------------------------


class Col:
    """Canonical column names of the photo table.

    Grouped by the substage family that first produces them, which is also the
    order in which they become available.
    """

    # -- ingest.gallery_assets (protobuf reads) ----------------------------
    IMAGE_ID = "image_id"
    EMBEDDING = "embedding"
    MODEL_VERSION = "model_version"
    IMAGE_CLASS = "image_class"
    CLUSTER_LABEL = "cluster_label"
    CLUSTER_CLASS = "cluster_class"
    RANKING = "ranking"
    IMAGE_ORDER = "image_order"
    PERSONS_IDS = "persons_ids"
    MAIN_PERSONS = "main_persons"
    N_FACES = "n_faces"
    FACES_INFO = "faces_info"
    NUMBER_BODIES = "number_bodies"
    BODIES_INFO = "bodies_info"
    IMAGE_TIME = "image_time"
    IMAGE_AS = "image_as"
    IMAGE_COLOR = "image_color"
    IMAGE_ORIENTATION = "image_orientation"
    IMAGE_ORDER_IN_SCENE = "image_orderInScene"
    SCENE_ORDER = "scene_order"
    BACKGROUND_CENTROID = "background_centroid"
    DIAMETER = "diameter"

    # -- ingest.rating -----------------------------------------------------
    USER_RATING = "user_rating"

    # -- enrich.content_class ---------------------------------------------
    CLUSTER_CONTEXT = "cluster_context"

    # -- enrich.identities -------------------------------------------------
    BRIDE_ID = "bride_id"
    GROOM_ID = "groom_id"
    PEOPLE_CLUSTER = "people_cluster"

    # -- enrich.semantic_tags (CLIP projection) ---------------------------
    IMAGE_QUERY_CONTENT = "image_query_content"
    IMAGE_SUBQUERY_CONTENT = "image_subquery_content"

    # -- enrich.temporal ---------------------------------------------------
    GENERAL_TIME = "general_time"
    IMAGE_TIME_DATE = "image_time_date"
    SCENE_NAME = "scene_name"

    # -- enrich.parents ----------------------------------------------------
    PARENT_CATEGORY = "parent_category"

    # -- enrich.ceremony_anchor --------------------------------------------
    SEND_OFF_SCORE = "send_off_score"
    AISLE_SCORE = "aisle_score"

    # -- enrich.key_pages --------------------------------------------------
    KEY_PAGE = "key_page"

    # -- select.* (scoring columns, per category) --------------------------
    TOTAL_SCORE = "total_score"
    CLASS_SCORE = "class_score"
    SIMILARITY_SCORE = "similarity_score"
    PERSON_SCORE = "person_score"
    TAGS_SCORE = "tags_score"
    RATING_SCORE = "rating_score"
    SUB_GROUP_TIME_CLUSTER = "sub_group_time_cluster"
    TEMPORAL_GROUP_ID = "temporal_group_id"


# --------------------------------------------------------------------------
# Requirement tokens
# --------------------------------------------------------------------------
#
# A substage declares its contract as a set of strings. Two namespaces:
#   photo:<column>  -> a column that must exist on ctx.photos
#   ctx:<field>     -> a context attribute that must be set (not None)
#
# The runner checks ``requires`` before a substage runs and ``provides`` after,
# so a replacement substage that forgets to produce something fails loudly at
# the boundary instead of silently downstream.


def photo(column: str) -> str:
    """Requirement token for a photo-table column."""
    return f"photo:{column}"


def ctx(field_name: str) -> str:
    """Requirement token for a context field."""
    return f"ctx:{field_name}"


# --------------------------------------------------------------------------
# Typed sidecars
# --------------------------------------------------------------------------


#: Tag cloud used when a request names no subjects of its own. Single source of
#: truth — this list was previously duplicated inline in main.py and
#: process_gallery.py.
DEFAULT_SUBJECTS = [
    'Wedding dress', 'ceremony', 'bride', 'dancing', 'bride getting ready',
    'groom getting ready', 'table setting', 'flowers', 'decorations', 'family',
    'baby', 'kids', 'mother', 'father', 'Romance', 'affection', 'Intimacy',
    'Happiness', 'Holding hands', 'smiling', 'Hugging', 'Kissing', 'ring',
    'veil', 'soft light', 'portrait',
]


@dataclass
class AiHints:
    """The user-supplied steering signal (``aiMetadata`` on the request).

    ``present`` is False when the request carried no ``aiMetadata`` (or a null
    ``photoIds``), which is what routes the request down the manual path.
    """

    photo_ids: List[int] = field(default_factory=list)
    person_ids: List[int] = field(default_factory=list)
    focus: List[str] = field(default_factory=lambda: ["everyoneElse"])
    subjects: List[str] = field(default_factory=list)
    density: int = 3
    present: bool = False

    @classmethod
    def from_request(cls, content: Dict[str, Any]) -> "AiHints":
        """Read ``aiMetadata`` off a request.

        A missing block, or a null ``photoIds``, means the user drove the
        selection by hand — that is what ``present=False`` signals.
        """
        metadata = content.get('aiMetadata')
        if metadata is None or metadata.get('photoIds') is None:
            return cls(present=False)

        return cls(
            photo_ids=metadata.get('photoIds', []),
            person_ids=metadata.get('personIds', []),
            focus=metadata.get('focus', ['everyoneElse']),
            subjects=metadata.get('subjects', DEFAULT_SUBJECTS),
            density=metadata.get('density', 3),
            present=True,
        )


@dataclass
class GalleryFacts:
    """Derived, gallery-wide facts. Produced by the enrich substages."""

    is_wedding: Optional[bool] = None
    #: True when the content model called the couple `two brides` / `two grooms`.
    #: Set by `enrich.same_sex_couple`, which also splits the solo-portrait class
    #: so both partners get their own allowance.
    same_sex_couple: bool = False
    is_artificial_time: bool = False
    model_version: Optional[int] = None
    bride_id: Optional[int] = None
    groom_id: Optional[int] = None
    #: Identities `enrich.parents` named, empty when that side was
    #: inconclusive. Empty is the normal outcome on a gallery whose candidates
    #: cannot be separated -- not an error.
    bride_parents: tuple = ()
    groom_parents: tuple = ()


@dataclass
class DesignSpec:
    """Layout/design data read from the request or its blob."""

    album_ar: Dict[str, float] = field(default_factory=lambda: {"anyPage": 2})
    pages: Dict[str, bool] = field(default_factory=dict)
    designs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectionOutcome:
    """What the selection substages hand to album processing."""

    photo_ids: List[int] = field(default_factory=list)
    spreads: Dict[str, float] = field(default_factory=dict)
    min_total_spreads: Optional[int] = None
    max_total_spreads: Optional[int] = None
    manual: bool = False
    lookup_table: Optional[Dict[str, tuple]] = None
    #: {category: {'actual': int, 'selected': int}} — selection diagnostics.
    per_category: Dict[str, Dict[str, int]] = field(default_factory=dict)


@dataclass
class KeyPages:
    """The photos that open and close the album.

    Ranked best-first, though today each list holds at most one id -- the rule
    that fills them picks a single cover per end. Lists rather than scalars
    because the consumer has to be able to fall through: the photo that best
    opens the gallery is not guaranteed to be one selection kept.
    """

    opening: List[int] = field(default_factory=list)
    closing: List[int] = field(default_factory=list)

    @property
    def photo_ids(self) -> List[int]:
        """Every id spoken for, in either role."""
        return list(self.opening) + list(self.closing)

    def as_content(self) -> Dict[str, List[int]]:
        return {"opening": list(self.opening), "closing": list(self.closing)}

    @classmethod
    def from_content(cls, payload: Any) -> Optional["KeyPages"]:
        if not isinstance(payload, dict):
            return None
        return cls(opening=list(payload.get("opening") or []),
                   closing=list(payload.get("closing") or []))


@dataclass
class Services:
    """External clients, injected rather than constructed inside substages so a
    substage can be exercised against fakes."""

    project_status_collection: Any = None
    qdrant_client: Any = None


# --------------------------------------------------------------------------
# The context
# --------------------------------------------------------------------------


@dataclass
class AlbumContext:
    """The single object that flows through every substage."""

    # -- transport / plumbing ---------------------------------------------
    message: Any = None
    logger: Any = None
    services: Services = field(default_factory=Services)

    # -- request ------------------------------------------------------------
    #: The raw request dict (``message.content``).
    request: Dict[str, Any] = field(default_factory=dict)
    project_url: Optional[str] = None
    project_id: Any = None
    #: Photo ids the user marked available. Empty means "the whole gallery".
    available_photo_ids: List[int] = field(default_factory=list)
    hints: AiHints = field(default_factory=AiHints)

    # -- data ---------------------------------------------------------------
    photos: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Pre-selection snapshot, kept for first/last page generation.
    all_photos: Optional[pd.DataFrame] = None
    designs: DesignSpec = field(default_factory=DesignSpec)

    # -- ingest sidecars ----------------------------------------------------
    clip_embeddings: Optional[pd.DataFrame] = None
    ratings: Optional[pd.DataFrame] = None
    social_circles: Optional[pd.DataFrame] = None
    person_details: Optional[pd.DataFrame] = None
    is_in_vector_db: Optional[bool] = None

    # -- derived ------------------------------------------------------------
    facts: GalleryFacts = field(default_factory=GalleryFacts)
    key_pages: Optional[KeyPages] = None
    selection: Optional[SelectionOutcome] = None

    # -- selection working state -------------------------------------------
    # Typed as Any to keep this module free of a dependency on the selection
    # package; the real types are
    # ``src.pipeline.select.contracts.SelectionInputs`` and ``SelectionPlan``.
    selection_inputs: Optional[Any] = None
    selection_plan: Optional[Any] = None

    # -- outcome ------------------------------------------------------------
    error: Optional[str] = None
    diagnostics: List["StageRecord"] = field(default_factory=list)

    # -- helpers ------------------------------------------------------------

    @property
    def failed(self) -> bool:
        return self.error is not None

    def fail(self, reason: str) -> "AlbumContext":
        """Record a terminal error. The runner stops the pipeline on the next
        boundary check."""
        self.error = reason
        if self.logger:
            self.logger.error(reason)
        return self

    def has(self, token: str) -> bool:
        """Whether a ``photo:``/``ctx:`` requirement token is satisfied."""
        kind, _, name = token.partition(":")
        if kind == "photo":
            return self.photos is not None and name in self.photos.columns
        if kind == "ctx":
            value = getattr(self, name, None)
            if isinstance(value, pd.DataFrame):
                return not value.empty
            return value is not None
        raise ValueError(f"Unknown requirement namespace in token {token!r}")

    def missing(self, tokens) -> List[str]:
        return sorted(t for t in tokens if not self.has(t))

    # -- boundary with the not-yet-decomposed stages ------------------------

    #: Attribute the context is parked under while the message hops between
    #: stages. Private to this module; use :meth:`for_message`.
    _MESSAGE_SLOT = "_album_context"

    @classmethod
    def from_message(cls, message, logger=None, services=None) -> "AlbumContext":
        """Build a fresh context from an incoming ``ptinfra`` message."""
        content = message.content if isinstance(message.content, dict) else {}
        context = cls(
            message=message,
            logger=logger,
            services=services or Services(),
            request=content,
            project_url=content.get("base_url"),
            project_id=content.get("projectId"),
        )
        context.attach_to_message()
        return context

    @classmethod
    def for_message(cls, message, logger=None, services=None) -> "AlbumContext":
        """The context for a message that an earlier stage already processed.

        Reuses the attached context when the message came straight from a
        previous stage in this process. Otherwise rebuilds one from
        ``message.content`` — which is what lets a later stage run on its own,
        against a message that was assembled by hand or by older code.
        """
        existing = getattr(message, cls._MESSAGE_SLOT, None)
        if isinstance(existing, cls):
            if logger is not None:
                existing.logger = logger
            if services is not None:
                existing.services = services
            existing.diagnostics = []
            return existing

        return cls._rehydrate(message, logger=logger, services=services)

    @classmethod
    def _rehydrate(cls, message, logger=None, services=None) -> "AlbumContext":
        content = message.content if isinstance(message.content, dict) else {}
        context = cls.from_message(message, logger=logger, services=services)

        context.photos = content.get("gallery_photos_info", pd.DataFrame())
        context.all_photos = content.get("gallery_all_photos_info")
        context.available_photo_ids = content.get("photos", []) or []
        context.hints = AiHints.from_request(content)
        context.key_pages = KeyPages.from_content(content.get("key_pages"))
        context.facts = GalleryFacts(
            is_wedding=content.get("is_wedding"),
            is_artificial_time=content.get("is_artificial_time", False),
        )
        return context

    def attach_to_message(self) -> "AlbumContext":
        """Park this context on the message so the next stage can pick it up."""
        if self.message is not None:
            setattr(self.message, self._MESSAGE_SLOT, self)
        return self

    def sync_to_message(self) -> Any:
        """Push context state back onto the message.

        This is what keeps ProcessStage and ReportStage working unchanged. It
        writes exactly the keys those stages read today.
        """
        message = self.message
        if message is None:
            return None

        content = message.content

        if self.photos is not None:
            content["gallery_photos_info"] = self.photos
        if self.all_photos is not None:
            content["gallery_all_photos_info"] = self.all_photos
        if self.facts.is_wedding is not None:
            content["is_wedding"] = self.facts.is_wedding
        content["is_artificial_time"] = self.facts.is_artificial_time

        if self.key_pages is not None:
            content["key_pages"] = self.key_pages.as_content()

        if self.selection is not None:
            content["photos"] = self.selection.photo_ids
            content["spreads_dict"] = self.selection.spreads
            content["min_total_spreads"] = self.selection.min_total_spreads
            content["max_total_spreads"] = self.selection.max_total_spreads
            content["modified_lut"] = self.selection.lookup_table
            if self.selection.manual:
                content["manual_selection"] = True

        if self.error is not None:
            content["error"] = self.error

        return message


@dataclass
class StageRecord:
    """One line of the pipeline's execution trace."""

    name: str
    ok: bool
    seconds: float
    photos_in: int
    photos_out: int
    note: str = ""

    def __str__(self) -> str:
        status = "ok " if self.ok else "ERR"
        delta = ""
        if self.photos_in != self.photos_out:
            delta = f" {self.photos_in}->{self.photos_out}"
        note = f" | {self.note}" if self.note else ""
        return f"[{status}] {self.name:<28} {self.seconds:7.3f}s{delta}{note}"
