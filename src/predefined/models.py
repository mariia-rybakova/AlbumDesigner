"""Input contract for predefined-spreads mode.

This is the boundary type we own: an external service hands us a fixed set of
photos per spread (and, when the design has cover pages, the cover photos). The
JSON shape is serialized from these dataclasses, so if the external format
changes only ``from_request`` has to change.

Granularity today is "photos only" -- ``layout_id`` / ``left_photo_ids`` /
``right_photo_ids`` are reserved for future modes where the service also fixes
the template or the page split, and are ``None`` for now (full stage-3 search).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

# Photo ids match the gallery's image_id dtype -- int in this codebase, but the
# external service may serialize them as strings, so accept either.
PhotoId = Union[int, str]


@dataclass
class PredefinedSpread:
    """One spread with its fixed photo set.

    Attributes:
        photo_ids: Image ids assigned to this spread (the fixed content).
        layout_id: Reserved -- when set, the layout template is fixed and the
            stage-3 layout search is skipped for this spread.
        left_photo_ids / right_photo_ids: Reserved -- when set, the left/right
            page split is fixed and the page-split search is skipped.
    """
    photo_ids: List[PhotoId]
    layout_id: Optional[int] = None
    left_photo_ids: Optional[List[PhotoId]] = None
    right_photo_ids: Optional[List[PhotoId]] = None


@dataclass
class PredefinedLayoutInput:
    """The whole predefined-layout request.

    Attributes:
        spreads: Ordered list of spreads (order is only a hint; the downstream
            project re-orders final spreads by time).
        first_page_photo_ids / last_page_photo_ids: Cover photos, consulted only
            when the album design actually has first/last cover slots.
    """
    spreads: List[PredefinedSpread]
    first_page_photo_ids: Optional[List[PhotoId]] = None
    last_page_photo_ids: Optional[List[PhotoId]] = None

    @classmethod
    def from_request(cls, content: Dict[str, Any]) -> Optional["PredefinedLayoutInput"]:
        """Parse the ``predefinedLayout`` block from a request's content.

        Returns ``None`` when the block is absent -- the single signal that the
        request should take the normal selection pipeline instead of the bypass.
        """
        block = content.get("predefinedLayout")
        if not block:
            return None

        spreads = [
            PredefinedSpread(
                photo_ids=list(s["photoIds"]),
                layout_id=s.get("layoutId"),
                left_photo_ids=s.get("leftPhotoIds"),
                right_photo_ids=s.get("rightPhotoIds"),
            )
            for s in block.get("spreads", [])
        ]
        return cls(
            spreads=spreads,
            first_page_photo_ids=block.get("firstPagePhotoIds"),
            last_page_photo_ids=block.get("lastPagePhotoIds"),
        )

    def all_photo_ids(self) -> List[PhotoId]:
        """Every image id referenced by the request (spreads + covers), deduped.

        Order-preserving dedup -- this is the photo universe to filter the
        gallery to and to run cropping over (covers need crop data too).
        """
        ids: List[str] = []
        for spread in self.spreads:
            ids.extend(spread.photo_ids)
        if self.first_page_photo_ids:
            ids.extend(self.first_page_photo_ids)
        if self.last_page_photo_ids:
            ids.extend(self.last_page_photo_ids)
        return list(dict.fromkeys(ids))