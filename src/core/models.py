from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Set, Optional
import pandas as pd
from src.core.photos import Photo
from utils.lookup_table_tools import LookUpTable


@dataclass
class SpreadSearchParams:
    """Parameters controlling the spread search and sampling process.

    Attributes:
        score_threshold: Minimum score for a multi-spread layout to be accepted.
        weight_threshold_divisor: Divisor applied to the max partition weight
            to compute the early-stopping threshold during partition filtering.
        max_spreads_sample: Maximum number of spreads/combinations to sample
            for large groups (> small_group_threshold photos).
        max_combs_small_group: Maximum number of combinations to sample
            for small groups (<= small_group_threshold photos).
        max_oriented_combs: Maximum number of oriented (portrait/landscape)
            combinations to sample per spread.
        small_group_threshold: Photo count threshold distinguishing small
            groups from large groups for sampling limits.
    """
    score_threshold: float = 0.01
    weight_threshold_divisor: float = 100.0
    max_spreads_sample: int = 1000
    max_combs_small_group: int = 100
    max_oriented_combs: int = 300
    small_group_threshold: int = 12


@dataclass
class AlbumDesignResources:
    layouts_df: pd.DataFrame
    layout_id2data: Dict[int, Any]
    box_id2data: Dict[Tuple[int, int], Any]
    max_pages: int
    look_up_table: LookUpTable = field(default_factory=LookUpTable)

    @classmethod
    def from_dict(cls, designs_info: Dict[str, Any], look_up_table: LookUpTable = None):
        return cls(
            layouts_df=designs_info['anyPagelayouts_df'],
            layout_id2data=designs_info['anyPagelayout_id2data'],
            box_id2data=designs_info['anyPagebox_id2data'],
            max_pages=designs_info['maxPages'],
            look_up_table=look_up_table or LookUpTable()
        )

@dataclass
class Spread:
    layout_id: int
    left_photos: List[Photo]
    right_photos: List[Photo]

@dataclass
class GroupProcessingResult:
    group_name: str
    spreads: List[Spread]
    score: float = 0.0

    def to_legacy_format(self):
        """
        Converts to the legacy [spreads_list, score] format for backward compatibility
        if needed during transition.
        """
        legacy_spreads = []
        for s in self.spreads:
            legacy_spreads.append([s.layout_id, s.left_photos, s.right_photos])
        return [legacy_spreads, self.score]
