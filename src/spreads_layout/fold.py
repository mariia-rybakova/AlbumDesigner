"""Keeping the album's fold off a person's face.

A design box is positioned in spread coordinates: ``x`` runs 0 to 1 across both
pages and the fold sits at 0.5. Most boxes live entirely on one page, but a box
whose range straddles 0.5 is printed across the fold, and the gutter runs down
the middle of whatever is at that column of the photo. On a face that is the
one crop mistake a viewer always notices.

Of the 13826 fold-crossing boxes in `files/designs.json`, 12837 are landscape
and 8986 of those put the fold within 5% of the box's centre -- the full-spread
photograph, and the centre of a full-spread photograph is where two people
standing together are. The rest divide into families that put the fold near one
edge of the box: a photo covering a page and a lip of the other.

Nothing here moves a photo. The crop for a full-spread box is the whole width
of the photo -- nearly every photo is narrower than the 2:1 that box asks for,
so it is cropped top and bottom and the fold lands on image column 0.5 with no
room to slide it. The only thing left to change is which layout holds the
photo.

This applies to the opening and closing pages only. Those are placed by
`cover_box` from the candidates `_find_single_box_layout` returns, and the
layout search never sees them, so the fold is chosen here or not at all. The
content spreads need nothing: a box only crosses the fold in a design holding
one photo, and a content spread with one photo is a bug of its own.

Knowing where the fold falls *inside the photo* means knowing the crop.
`estimate_window` answers with `smart_cropping.placement_window` -- the same
function the placement itself uses.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.smart_cropping import _as_list, placement_window
from utils.configs import CONFIGS

#: Where the fold is, in spread coordinates.
FOLD_X = 0.5

#: Below this a cost is floating-point noise, not a face on the fold.
_EPSILON = 1e-9

#: One face as the fold sees it: its left and right edge in image coordinates,
#: and how much it counts -- 1.0 for the widest face in the photo, less for the
#: smaller ones, so a guest in the background does not outvote the subject.
FaceSpan = Tuple[float, float, float]

#: A crop window, normalised to the image: (x, y, width, height).
Window = Tuple[float, float, float, float]


def settings() -> Dict[str, Any]:
    return CONFIGS.get('fold_awareness', {}) or {}


def enabled() -> bool:
    return bool(settings().get('enabled', True))


def fold_fraction(box: Dict[str, Any], edge_tolerance: Optional[float] = None) -> Optional[float]:
    """Where the fold falls inside `box`, as a fraction of its width.

    0 is the box's left edge and 1 its right, so a full-spread box answers 0.5
    and a box covering the left page plus a lip of the right answers ~0.9.
    None when the box does not cross the fold at all, which is the answer for
    most boxes and the reason everything else here is cheap.

    A box that merely *ends* at the fold is flush with it, not cut by it, so it
    has to reach `edge_tolerance` past it to count -- design data carries those
    edges a hair over 0.5.
    """
    try:
        x = float(box['x'])
        width = float(box['width'])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0:
        return None

    tol = edge_tolerance if edge_tolerance is not None else float(settings().get('edge_tolerance', 0.005))
    if not (x < FOLD_X - tol and x + width > FOLD_X + tol):
        return None
    return (FOLD_X - x) / width


def face_spans(faces: Any) -> Tuple[FaceSpan, ...]:
    """The horizontal extent of each real face, weighted by how big it is.

    `blurLevel` of -1 is Face-Recognition's mark for a detection it judged not
    to be a face, the same filter `single_subject_window` applies.
    """
    boxes: List[Tuple[float, float]] = []
    for face in _as_list(faces):
        if getattr(face, 'blurLevel', 0) < 0:
            continue
        bbox = getattr(face, 'bbox', None)
        if bbox is None:
            continue
        try:
            x1, x2 = float(bbox.x1), float(bbox.x2)
        except (AttributeError, TypeError, ValueError):
            continue
        if x2 > x1:
            boxes.append((x1, x2))

    if not boxes:
        return ()

    widest = max(x2 - x1 for x1, x2 in boxes)
    return tuple((x1, x2, (x2 - x1) / widest) for x1, x2 in boxes)


def gutter_in_image(box: Dict[str, Any], window: Window,
                    gutter_width: Optional[float] = None) -> float:
    """The band the fold swallows, converted from the page to the photo.

    `gutter_width` is a share of the spread's width, because that is what the
    gutter is: a fixed strip of paper either side of the fold, the same size
    whoever is standing in it. The photo only shows `crop_w` of itself across
    `box['width']` of the spread, so a strip of the page is that much wider or
    narrower measured in the image.
    """
    width = gutter_width if gutter_width is not None else float(settings().get('gutter_width', 0.025))
    try:
        box_width = float(box['width'])
    except (KeyError, TypeError, ValueError):
        return 0.0
    if box_width <= 0:
        return 0.0
    return width * float(window[2]) / box_width


def fold_cost(spans: Sequence[FaceSpan], fold_u: float,
              face_margin: Optional[float] = None, gutter: float = 0.0) -> float:
    """How badly a fold at image column `fold_u` cuts these faces.

    Graded rather than yes/no, so the search can prefer the lesser of two
    damaged layouts: 1 per face for a fold straight down the middle of it,
    falling to 0 at the edge of its margin. Summed over the faces and weighted
    by their size, so a fold through two faces at once is worse than through
    one, and a face in the background costs less than the subject's.

    A face is widened on both sides before it is measured, by two separate
    things. `face_margin` is a share of the face's own half-width -- a fold
    against someone's ear is barely better than through their nose, and how
    close that is depends on the size of the head. `gutter` is the strip of
    paper the fold itself eats, already converted to image units by
    `gutter_in_image`, and it does *not* scale with the face: the gutter is the
    same width of paper whoever is standing in it. Without that term the margin
    shrank with the head, and on 53316807 a face ending 0.009 of the image from
    the fold scored exactly zero while the gutter ran through its edge.
    """
    margin = face_margin if face_margin is not None else float(settings().get('face_margin', 0.15))

    total = 0.0
    for x1, x2, weight in spans:
        centre = (x1 + x2) / 2
        half = (x2 - x1) / 2 * (1.0 + 2 * margin) + max(0.0, gutter) / 2
        if half <= 0:
            continue
        overlap = 1.0 - abs(fold_u - centre) / half
        # A fold exactly on the margin touches nothing, but saying so in
        # floating point leaves a few 1e-16 behind, and every caller tests
        # `cost <= 0` to mean "clean, stop looking".
        if overlap > _EPSILON:
            total += weight * overlap
    return total if total > _EPSILON else 0.0


def box_target_ar(box: Dict[str, Any], album_ar: float = 2.0) -> float:
    """The box's width:height in image terms -- what a crop is fitted to.

    The same arithmetic as `request_processing.box_target_ar`, kept here so
    this module does not have to import that one (it opens Qdrant and Mongo).
    """
    return float(box['width']) / float(box['height']) * float(album_ar)


def estimate_window(image_ar: float, box: Dict[str, Any], album_ar: float,
                    faces: Any = None, bodies: Any = None) -> Window:
    """Where this photo's crop will sit inside this box."""
    return placement_window(faces, bodies, image_ar, box_target_ar(box, album_ar))


def fold_image_x(window: Window, fraction: float) -> float:
    """The image column the fold lands on, given the crop and where it cuts the box."""
    crop_x, _, crop_w, _ = window
    return crop_x + fraction * crop_w


# --------------------------------------------------------------------------
# Layout-time scoring
# --------------------------------------------------------------------------

def pick_cover_layout(candidates: List[int], layouts_df, box_id2data: Dict[Tuple[int, int], Any],
                      image_row, album_ar: float = 2.0, logger=None) -> Optional[int]:
    """Which of the opening/closing candidates keeps the fold off the faces.

    `_find_single_box_layout` returns every single-box design that suits the
    photo's orientation, in its own order of preference -- large boxes first --
    and the caller has always taken the first. On a pack whose single-box
    designs all cross the fold, as 53316807's does, that put the gutter through
    the opening photo with nothing to notice it: the covers are placed by
    `cover_box`, not by the layout search, so none of the rest of this module
    ever sees them.

    The order is kept where it does not matter. A candidate only displaces the
    one in front of it by being strictly cleaner, so among equals -- and when
    no candidate crosses the fold at all -- this returns exactly what it used
    to.
    """
    if not candidates:
        return None
    if not enabled():
        return candidates[0]

    faces = image_row.get('faces_info') if hasattr(image_row, 'get') else None
    spans = face_spans(faces)
    if not spans:
        return candidates[0]
    try:
        image_ar = float(image_row['image_as'])
    except (KeyError, TypeError, ValueError):
        return candidates[0]

    best_idx, best_cost = candidates[0], None
    for layout_idx in candidates:
        try:
            row = layouts_df.loc[layout_idx]
        except KeyError:
            continue
        box_ids = list(row['left_box_ids']) + list(row['right_box_ids'])
        if len(box_ids) != 1:
            continue
        box = box_id2data.get((row['id'], box_ids[0]))
        if not box:
            continue

        fraction = fold_fraction(box)
        if fraction is None:
            cost = 0.0            # does not cross the fold: nothing to damage
        else:
            window = estimate_window(image_ar, box, album_ar, faces,
                                     image_row.get('bodies_info'))
            cost = fold_cost(spans, fold_image_x(window, fraction),
                             gutter=gutter_in_image(box, window))

        if best_cost is None or cost < best_cost - _EPSILON:
            best_idx, best_cost = layout_idx, cost
            if cost <= 0:
                break

    if logger and best_idx != candidates[0]:
        logger.info(f"fold: cover layout {layouts_df.loc[best_idx]['id']} taken over "
                    f"{layouts_df.loc[candidates[0]]['id']}, faces on the fold {best_cost:.2f}")
    return best_idx


