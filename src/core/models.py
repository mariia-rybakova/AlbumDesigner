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
class LayoutsResources:
    """Static print-lab reference data shared across the whole album request.

    Built once per request from designs_info; fields are immutable through
    grouping and layouting. Carries everything needed to reconstruct or
    render a layout: the DataFrame of layouts, idx → layout-data lookup,
    (layout_id, box_id) → orientation/area lookup, and the album page cap.
    """
    layouts_df: pd.DataFrame
    layout_id2data: Dict[int, Any]
    box_id2data: Dict[Tuple[int, int], Any]
    max_pages: int

    @classmethod
    def from_dict(cls, designs_info: Dict[str, Any]) -> 'LayoutsResources':
        return cls(
            layouts_df=designs_info['anyPagelayouts_df'],
            layout_id2data=designs_info['anyPagelayout_id2data'],
            box_id2data=designs_info['anyPagebox_id2data'],
            max_pages=designs_info['maxPages'],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize layouts reference data to a JSON-ready structure.

        layouts_df rows carry their iterrows-index as 'index' — that's the
        same key recorded as 'layout_idx' in per-group spread files, so an
        analysis script can join top-k candidates / chosen layouts back to
        their full geometry. layout_id2data and box_id2data are emitted as
        list-of-records (their original keys are int / tuple, neither
        JSON-friendly).
        """
        return {
            'max_pages': self.max_pages,
            'layouts_df': self.layouts_df.reset_index().to_dict('records'),
            'layout_id2data': [
                {'idx': idx,
                 'layout_id': d.get('layout_id'),
                 'boxes_areas': d.get('boxes_areas'),
                 'left_box_ids': d.get('left_box_ids'),
                 'right_box_ids': d.get('right_box_ids')}
                for idx, d in self.layout_id2data.items()
            ],
            'box_id2data': [
                {'layout_id': k[0], 'box_id': k[1],
                 'orientation': v.get('orientation'),
                 'area': v.get('area')}
                for k, v in self.box_id2data.items()
            ],
        }


@dataclass
class AlbumDesignResources:
    printlab_data: LayoutsResources
    look_up_table: LookUpTable = field(default_factory=LookUpTable)

    @classmethod
    def from_dict(cls, designs_info: Dict[str, Any],
                  look_up_table: LookUpTable = None) -> 'AlbumDesignResources':
        return cls(
            printlab_data=LayoutsResources.from_dict(designs_info),
            look_up_table=look_up_table or LookUpTable(),
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
