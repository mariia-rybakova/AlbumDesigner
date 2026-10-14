"""Fold records — used for files/stages_info/fold/fold.json.

Answers one question per one-photo spread: *where does the fold fall inside
this photo, and how close is it to anyone in it?*

Only one-photo spreads are recorded, because they are the only ones a box can
cross the fold on: a design holding a cross-fold box holds nothing else beside
it. The record is taken in `assembly_output`, which is the first place the
printed geometry is settled -- the box as it is actually laid down (mirrored
where the layout is), and the crop that `customize_box` chose for it.

Everything is stored in *image* coordinates, 0 to 1 across the photo, so the
renderer can map them onto the page through the same crop the album uses:

  * `fold_u`    -- the image column the fold lands on, or null when the box
                   does not cross it.
  * `spans`     -- each face found and each head estimated from a body with no
                   face in it (`spreads_layout.fold.subject_spans`), with the
                   margin the cost actually applies around it: a face widened
                   by `face_margin` of its own half-width, plus half the gutter.
                   That margin is the thing worth seeing on the page -- the
                   fold being outside the face but inside the margin is exactly
                   the case that reads as "too close".

Gated on `CONFIGS['save_files']['fold']`; every entry point is a cheap no-op
when the flag is off.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List, Optional

from utils.configs import CONFIGS

_FOLD_OUT_DIR = os.path.join('files', 'stages_info', 'fold')
_FOLD_FILE = 'fold.json'

#: Spreads recorded so far this album, in page order.
_pages: List[Dict[str, Any]] = []


def is_enabled() -> bool:
    """Whether fold recording is switched on for this run."""
    return bool(CONFIGS.get('save_files', {}).get('fold', False))


def reset() -> None:
    """Drop anything a previous run left, so one album's pages never mix."""
    _pages.clear()
    if not is_enabled():
        return
    try:
        shutil.rmtree(_FOLD_OUT_DIR, ignore_errors=True)
        os.makedirs(_FOLD_OUT_DIR, exist_ok=True)
    except OSError:
        pass


def record_spread(composition_id: int, design_id: Any, page: int,
                  box: Dict[str, Any], placement: Dict[str, Any],
                  image_info: Any, album_ar: float = 2.0) -> None:
    """Record one one-photo spread. Never raises: a debug dump is not worth a run.

    `box` is the printed box -- mirrored geometry where the layout is mirrored
    -- and `placement` the entry already appended to `placementsImg`, so the
    crop recorded is the one the album prints.
    """
    if not is_enabled():
        return
    try:
        _pages.append(_build(composition_id, design_id, page, box, placement,
                             image_info, album_ar))
    except Exception:  # noqa: BLE001 - see docstring
        pass


def _build(composition_id, design_id, page, box, placement, image_info, album_ar):
    # Imported here so the recorder costs nothing to import when the flag is off.
    from src.spreads_layout import fold as fold_rules

    faces = image_info.get('faces_info') if hasattr(image_info, 'get') else None
    bodies = image_info.get('bodies_info') if hasattr(image_info, 'get') else None
    face_only = fold_rules.face_spans(faces)
    spans = fold_rules.subject_spans(faces, bodies)

    crop = {'x': float(placement['cropX']), 'y': float(placement['cropY']),
            'width': float(placement['cropWidth']), 'height': float(placement['cropHeight'])}
    window = (crop['x'], crop['y'], crop['width'], crop['height'])

    fraction = fold_rules.fold_fraction(box)
    gutter = fold_rules.gutter_in_image(box, window) if fraction is not None else 0.0
    margin = float(fold_rules.settings().get('face_margin', 0.15))
    fold_u = fold_rules.fold_image_x(window, fraction) if fraction is not None else None

    recorded = []
    for index, (x1, x2, weight) in enumerate(spans):
        centre = (x1 + x2) / 2
        half = (x2 - x1) / 2 * (1.0 + 2 * margin) + max(0.0, gutter) / 2
        recorded.append({
            'kind': 'face' if index < len(face_only) else 'body',
            'x1': float(x1), 'x2': float(x2), 'weight': float(weight),
            'limit_x1': float(centre - half), 'limit_x2': float(centre + half),
        })

    return {
        'composition_id': int(composition_id),
        'page': int(page),
        'design_id': int(design_id) if design_id is not None else None,
        'photo_id': placement.get('photoId'),
        'box': {'x': float(box['x']), 'y': float(box['y']),
                'width': float(box['width']), 'height': float(box['height'])},
        'crop': crop,
        'album_ar': float(album_ar),
        'crosses_fold': fraction is not None,
        'fold_fraction': None if fraction is None else float(fraction),
        'fold_u': None if fold_u is None else float(fold_u),
        'gutter': float(gutter),
        'face_margin': margin,
        'spans': recorded,
        'clearance': (float(fold_rules.fold_clearance(spans, fold_u))
                      if (fold_u is not None and spans) else None),
        'cost': (float(fold_rules.fold_cost(spans, fold_u, gutter=gutter))
                 if (fold_u is not None and spans) else 0.0),
    }


def flush(project_id: Any = None) -> Optional[str]:
    """Write the album's pages, or nothing at all when it has none.

    An album whose every spread holds more than one photo has nothing the fold
    can cross, and a PDF of no pages is worse than no PDF: `visualization.py`
    skips the stage when this file is absent.
    """
    if not is_enabled() or not _pages:
        return None
    try:
        os.makedirs(_FOLD_OUT_DIR, exist_ok=True)
        path = os.path.join(_FOLD_OUT_DIR, _FOLD_FILE)
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump({'project_id': project_id,
                       'pages': sorted(_pages, key=lambda p: p['page'])},
                      handle, indent=2)
        return path
    except Exception:  # noqa: BLE001 - a debug dump must never kill a run
        return None
