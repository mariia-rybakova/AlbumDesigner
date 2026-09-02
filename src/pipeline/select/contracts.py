"""Data passed between the selection substages and the per-category strategies.

The selection stage has two levels of replaceability:

* **Substages** (``select.route`` -> ``select.budget`` -> ``select.pick`` ->
  ``select.publish``) — the coarse phases, swapped through the pipeline registry.
* **Category strategies** — how one content category picks its photos. The
  wedding picker is a per-category dispatch, so each branch is its own
  :class:`~src.pipeline.select.strategies.base.CategoryStrategy` and can be
  replaced without touching the driver.

:class:`CategoryRequest` is the whole input a strategy gets, and
:class:`CategoryPicks` is the whole output it may produce. A strategy that
honours those two types is interchangeable with any other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class SelectionInputs:
    """The request-level constants every category is judged against.

    Resolved once by ``select.route`` so the per-category loop does no parsing.
    """

    #: Photo ids the user hand-picked (``aiMetadata.photoIds``).
    user_selected_ids: List[Any] = field(default_factory=list)
    person_ids: List[Any] = field(default_factory=list)
    #: {tag: embedding matrix}, or ``[]`` when the request named no subjects.
    tags_features: Any = field(default_factory=list)
    #: {photo_id: rating}
    ratings: Dict[Any, Any] = field(default_factory=dict)
    density: int = 3
    focus: List[str] = field(default_factory=lambda: ["everyoneElse"])

    @property
    def unscored(self) -> bool:
        """Nothing to score against, so ``image_order`` decides the ranking."""
        return (
            len(self.person_ids) == 0
            and len(self.user_selected_ids) == 0
            and len(self.tags_features) == 0
        )


@dataclass
class SelectionPlan:
    """The per-category budget, produced by ``select.budget``."""

    #: {category: how many photos to pick}
    images: Dict[str, int] = field(default_factory=dict)
    #: {category: how many spreads it should occupy}
    spreads: Dict[str, float] = field(default_factory=dict)
    min_total_spreads: Optional[int] = None
    max_total_spreads: Optional[int] = None
    #: Density-scaled photos-per-spread table.
    lookup_table: Optional[Dict[str, tuple]] = None

    #: Categories the focus profile budgets as `yes` rather than a percentage,
    #: and that this gallery has. Resolved once by `select.budget` so nothing
    #: downstream re-reads the profile or re-decides what `yes` means.
    yes_categories: tuple = ()

    #: {image_id: why} -- photos `select.preselect` committed before any
    #: ranking. `select.pick` starts from these and does not re-pick them.
    committed: Dict[Any, str] = field(default_factory=dict)


@dataclass
class CategoryRequest:
    """Everything a strategy needs to pick photos for one content category.

    The driver has already done the shared work: scoring, threshold gating,
    removal of user-picked photos, temporal narrowing, the fixed-span time
    clustering and the colour/greyscale split.
    """

    category: str
    #: How many photos this category still needs.
    need: int

    #: Colour candidates (``image_color != 0``) — what strategies pick from.
    color: pd.DataFrame
    #: Greyscale candidates, used by the driver to top up a shortfall.
    grayscale: pd.DataFrame
    #: Colour + greyscale, carrying ``sub_group_time_cluster``.
    pool: pd.DataFrame

    #: False when the request carried no hints, so ordering falls back to
    #: ``image_order`` ascending instead of ``total_score`` descending.
    scored: bool = True
    #: {image_id: ordering key} in the direction implied by ``scored``.
    order_index: Dict[Any, Any] = field(default_factory=dict)

    #: Photos the user hand-picked, for the categories that prefer them.
    user_selected: pd.DataFrame = field(default_factory=pd.DataFrame)

    is_artificial_time: bool = False
    logger: Any = None

    # -- convenience -------------------------------------------------------

    @property
    def bride_id(self) -> int:
        return int(self.color['bride_id'].iloc[0])

    @property
    def groom_id(self) -> int:
        return int(self.color['groom_id'].iloc[0])

    def ordered(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Sort a frame by this request's ordering convention."""
        if self.scored:
            return frame.sort_values('total_score', ascending=False)
        return frame.sort_values('image_order', ascending=True)


@dataclass
class CategoryPicks:
    """What a strategy returns.

    ``preferred`` is the ranked candidate list; the driver truncates it to
    ``need`` and tops up from greyscale if it falls short.

    ``forced`` are photos the strategy committed to outright, outside the
    ranked list (only ``walking the aisle`` does this today: the first
    bride-and-groom frame and the bride's arrival at the altar). They are
    already deducted from the ``need`` the strategy reports back.
    """

    preferred: Optional[List[Any]] = None
    forced: List[Any] = field(default_factory=list)
    #: Remaining need after ``forced``. ``None`` means "unchanged".
    remaining_need: Optional[int] = None

    #: Abandon this category entirely — no ranked list, no carry-over.
    skip: bool = False
