from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Set, Optional
import pandas as pd
from src.core.photos import Photo

@dataclass
class AlbumDesignResources:
    layouts_df: pd.DataFrame
    layout_id2data: Dict[int, Any]
    box_id2data: Dict[Tuple[int, int], Any]
    max_pages: int
    look_up_table: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, designs_info: Dict[str, Any], look_up_table: Dict[str, Tuple[float, float]] = None):
        return cls(
            layouts_df=designs_info['anyPagelayouts_df'],
            layout_id2data=designs_info['anyPagelayout_id2data'],
            box_id2data=designs_info['anyPagebox_id2data'],
            max_pages=designs_info['maxPages'],
            look_up_table=look_up_table or {}
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
