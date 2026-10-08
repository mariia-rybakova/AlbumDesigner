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

Three levers, because no one of them reaches every case:

**Choosing the layout.** The crop for a full-spread box is the whole width of
the photo -- nearly every photo is narrower than the 2:1 that box asks for, so
it is cropped top and bottom and the fold lands on image column 0.5 with no
room to move. Nothing about the placement can save that; only a different
layout, or a different split of the photos across spreads, can. So the layout
search re-ranks its own candidates: it keeps the best-scoring one unless a
close-enough rival puts the fold somewhere harmless. See `choose_cleaner_fold`.

**Re-housing the spread.** The candidate list reaches less far than it looks:
`process_group_lists` builds it with `itertools.product`, which varies the last
spread fastest, so the leading candidates all repeat one layout for the first
spread. A damaged photo there is unreachable however many candidates are
walked. So each spread still carrying a face on the fold is moved into another
layout of the same shape -- the same orientations in the same places, which
scores identically and differs only in where the boxes sit on the paper. See
`repair_spreads`.

**Sliding the crop.** Where the photo *is* wider than the box, the crop window
has slack to either side, and the fold can be walked out of a face without
changing anything else about the placement. See `clear_fold`.

Both need to know where the fold falls *inside the photo*, which means knowing
the crop. For every orientation but square that is `smart_cropping`'s
`placement_window`, the same function the placement itself uses, so the layout's
answer is exact. A square box takes the pre-computed 1:1 crop, which is made in
a subprocess that runs alongside the layout and is not available to it;
`estimate_window` approximates it with the two cheap branches of
`process_cropping` it can reproduce. Squares are 549 of the 13826 crossing
boxes, so the approximation covers 4% of the cases and never the common one.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.smart_cropping import (_as_list, centered_window, placement_window,
                                single_subject_window)
from src.spreads_layout.spreads.spread import SingleSpreadLayout
from utils.configs import CONFIGS

#: Where the fold is, in spread coordinates.
FOLD_X = 0.5

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


def fold_cost(spans: Sequence[FaceSpan], fold_u: float,
              face_margin: Optional[float] = None) -> float:
    """How badly a fold at image column `fold_u` cuts these faces.

    Graded rather than yes/no, so the search can prefer the lesser of two
    damaged layouts: 1 per face for a fold straight down the middle of it,
    falling to 0 at the edge of its margin. Summed over the faces and weighted
    by their size, so a fold through two faces at once is worse than through
    one, and a face in the background costs less than the subject's.

    The margin widens each face by `face_margin` of its half-width on both
    sides, because the gutter takes a band of the print, not a line, and a fold
    against someone's ear is barely better than through their nose.
    """
    margin = face_margin if face_margin is not None else float(settings().get('face_margin', 0.15))

    total = 0.0
    for x1, x2, weight in spans:
        centre = (x1 + x2) / 2
        half = (x2 - x1) / 2 * (1.0 + 2 * margin)
        if half <= 0:
            continue
        overlap = 1.0 - abs(fold_u - centre) / half
        if overlap > 0:
            total += weight * overlap
    return total


def box_target_ar(box: Dict[str, Any], album_ar: float = 2.0) -> float:
    """The box's width:height in image terms -- what a crop is fitted to.

    The same arithmetic as `request_processing.box_target_ar`, kept here so
    this module does not have to import that one (it opens Qdrant and Mongo).
    """
    return float(box['width']) / float(box['height']) * float(album_ar)


def _square_window(image_ar: float, faces: Any, centroid: Any) -> Optional[Window]:
    """`process_cropping`'s 1:1 crop, for the branches reproducible from here.

    A square box is placed with the crop that subprocess computes, not with a
    window of the box's own ratio, so this is what the layout has to predict.
    Two of its three branches are arithmetic -- centre the 1:1 window on the
    saliency blob when there are no faces, on the faces' own range when there
    are and they fit -- and the third is a search over a mask that would cost
    more here than it is worth. None hands the caller back to `centered_window`.
    """
    width, height = (1.0 / image_ar, 1.0) if image_ar > 1 else (1.0, image_ar)

    spans = [(x1, x2) for x1, x2, _ in face_spans(faces)]
    if not spans:
        if centroid is None:
            return None
        try:
            cx, cy = float(centroid.x), float(centroid.y)
        except (AttributeError, TypeError, ValueError):
            return None
        if image_ar > 1:
            return min(max(cx - width / 2, 0.0), 1.0 - width), 0.0, width, height
        return 0.0, min(max(cy - height / 2, 0.0), 1.0 - height), width, height

    if image_ar > 1:
        low, high = min(x1 for x1, _ in spans), max(x2 for _, x2 in spans)
        if high - low < width:
            x = min(max((low + high) / 2 - width / 2, 0.0), 1.0 - width)
            return x, 0.0, width, height

    # The portrait branch moves the window vertically only, so it leaves the
    # fold exactly where a centred window would. Not worth reproducing.
    return None


def estimate_window(image_ar: float, box: Dict[str, Any], album_ar: float,
                    faces: Any = None, bodies: Any = None, centroid: Any = None) -> Window:
    """Where this photo's crop will sit inside this box.

    Exact for every orientation but square, where it approximates -- see the
    module docstring and `_square_window`.
    """
    if box.get('orientation') == 'square':
        window = single_subject_window(faces, bodies, image_ar, 1.0)
        if window is not None:
            return window
        window = _square_window(image_ar, faces, centroid)
        if window is not None:
            return window
        return centered_window(image_ar, 1.0)

    return placement_window(faces, bodies, image_ar, box_target_ar(box, album_ar))


def fold_image_x(window: Window, fraction: float) -> float:
    """The image column the fold lands on, given the crop and where it cuts the box."""
    crop_x, _, crop_w, _ = window
    return crop_x + fraction * crop_w


def clear_fold(window: Window, fraction: float, spans: Sequence[FaceSpan],
               face_margin: Optional[float] = None) -> Window:
    """`window` slid sideways until the fold is off the faces, as far as it can.

    Only the x of the window moves, so the photo keeps the framing the crop
    rules chose for it; it is the same pixels, shifted. Slack exists only where
    the photo is wider than the box -- a full-spread box shows the whole width
    and this returns the window untouched.

    The candidates are the positions that put the fold just clear of one face
    or another, clamped to the image. The one that costs least wins, and among
    equals the one that moves least, so a window that is already clean stays
    exactly where it was.
    """
    x, y, w, h = (float(value) for value in window)
    if not spans or w >= 1.0:
        return window

    margin = face_margin if face_margin is not None else float(settings().get('face_margin', 0.15))
    if fold_cost(spans, x + fraction * w, margin) <= 0:
        return window

    limit = 1.0 - w
    candidates = {x}
    for x1, x2, _ in spans:
        centre = (x1 + x2) / 2
        half = (x2 - x1) / 2 * (1.0 + 2 * margin)
        for edge in (centre - half, centre + half):
            candidates.add(edge - fraction * w)

    best = None
    for candidate in candidates:
        candidate = min(max(candidate, 0.0), limit)
        cost = fold_cost(spans, candidate + fraction * w, margin)
        key = (cost, abs(candidate - x))
        if best is None or key < best[0]:
            best = (key, candidate)

    return float(best[1]), y, w, h


# --------------------------------------------------------------------------
# Layout-time scoring
# --------------------------------------------------------------------------

def photo_fold_info(images_df) -> Dict[Any, Dict[str, Any]]:
    """What the fold needs to know about each photo of a group, by image id.

    Built once per group from the frame the layout is handed. Photos with no
    face in them are kept out: nothing about them can be damaged by a fold, and
    their absence is what lets `group_layout_fold_cost` leave early.
    """
    info: Dict[Any, Dict[str, Any]] = {}
    if images_df is None or getattr(images_df, 'empty', True):
        return info

    # Narrowed to the columns wanted rather than read off the whole row, both
    # because the gallery frame is wide and because a missing one here would
    # raise inside `process_group`'s try/except and cost the album the group.
    columns = set(images_df.columns)
    if not {'image_id', 'image_as', 'faces_info'} <= columns:
        return info
    optional = [name for name in ('bodies_info', 'background_centroid') if name in columns]

    frame = images_df[['image_id', 'image_as', 'faces_info'] + optional]
    for row in frame.itertuples(index=False, name=None):
        image_id, image_as, faces = row[0], row[1], row[2]
        extra = dict(zip(optional, row[3:]))

        spans = face_spans(faces)
        if not spans:
            continue
        try:
            image_ar = float(image_as)
        except (TypeError, ValueError):
            continue
        if image_ar <= 0:
            continue

        info[image_id] = {
            'ar': image_ar,
            'spans': spans,
            'faces': faces,
            'bodies': extra.get('bodies_info'),
            'centroid': extra.get('background_centroid'),
        }
    return info


def _spread_pairs(spread, layout_id2data: Dict[int, Any]):
    """(box data key, photo) for every filled box of a spread, or nothing.

    The ordered photo lists a resolved layout carries line up with the layout's
    box ids position by position -- the same correspondence `assembly_output`
    relies on to place them.
    """
    layout = layout_id2data.get(spread.layout_idx)
    if layout is None:
        return
    layout_id = layout['layout_id']
    for box_ids, photos in ((layout['left_box_ids'], spread.left_page_photos),
                            (layout['right_box_ids'], spread.right_page_photos)):
        for box_id, photo in zip(box_ids or (), photos or ()):
            if photo is not None:
                yield (layout_id, box_id), photo


def spread_fold_cost(spread, layout_id2data: Dict[int, Any],
                     box_id2data: Dict[Tuple[int, int], Any],
                     photo_info: Dict[Any, Dict[str, Any]],
                     album_ar: float = 2.0,
                     face_margin: Optional[float] = None) -> float:
    """How much face one spread puts on the fold; its photos must be resolved."""
    tolerance = float(settings().get('edge_tolerance', 0.005))
    total = 0.0
    for key, photo in _spread_pairs(spread, layout_id2data):
        box = box_id2data.get(key)
        if not box:
            continue
        fraction = fold_fraction(box, tolerance)
        if fraction is None:
            continue
        info = photo_info.get(photo.id)
        if info is None:
            continue
        window = estimate_window(info['ar'], box, album_ar,
                                 info['faces'], info['bodies'], info['centroid'])
        total += fold_cost(info['spans'], fold_image_x(window, fraction), face_margin)
    return total


def group_layout_fold_cost(group_layout, layout_id2data: Dict[int, Any],
                           box_id2data: Dict[Tuple[int, int], Any],
                           photo_info: Dict[Any, Dict[str, Any]],
                           album_ar: float = 2.0,
                           face_margin: Optional[float] = None) -> float:
    """How much face this candidate puts on the fold, over all its spreads.

    Zero for a layout with no cross-fold box, which is most of them, and the
    photos must already be resolved onto their boxes.
    """
    return sum(spread_fold_cost(spread, layout_id2data, box_id2data, photo_info,
                                album_ar, face_margin)
               for spread in group_layout.spreads_layouts)


def _page_shape(ids, layout_id, box_id2data) -> Tuple[int, int, int]:
    """How many portrait / landscape / square boxes a page holds."""
    counts = {'portrait': 0, 'landscape': 0, 'square': 0}
    for box_id in ids or ():
        orientation = (box_id2data.get((layout_id, box_id)) or {}).get('orientation')
        if orientation in counts:
            counts[orientation] += 1
    return counts['portrait'], counts['landscape'], counts['square']


def layout_shape(layout_idx: int, layout_id2data: Dict[int, Any],
                 box_id2data: Dict[Tuple[int, int], Any]) -> Optional[Tuple[int, ...]]:
    """The per-page box orientations of a layout, as one comparable key.

    Two layouts with the same shape hold the same photos the same way round:
    the same count of each orientation on each page. Everything
    `SingleSpreadLayout.get_score` measures is then identical between them --
    the colour, class, context and time terms read the photos, which do not
    move, and `left_mixed` / `right_mixed` and the square count follow from
    these counts. So one can be swapped for the other at no cost to the score,
    and the only thing that differs is where the boxes sit on the paper.
    """
    layout = layout_id2data.get(layout_idx)
    if layout is None:
        return None
    layout_id = layout['layout_id']
    return (_page_shape(layout['left_box_ids'], layout_id, box_id2data)
            + _page_shape(layout['right_box_ids'], layout_id, box_id2data))


def _shape_index(layout_id2data, box_id2data) -> Dict[Tuple[int, ...], List[int]]:
    index: Dict[Tuple[int, ...], List[int]] = {}
    for layout_idx in layout_id2data:
        shape = layout_shape(layout_idx, layout_id2data, box_id2data)
        if shape is not None:
            index.setdefault(shape, []).append(layout_idx)
    return index


def repair_spreads(group_layout, photos: List[Any], layout_id2data: Dict[int, Any],
                   box_id2data: Dict[Tuple[int, int], Any],
                   photo_info: Dict[Any, Dict[str, Any]],
                   album_ar: float = 2.0, logger=None) -> Tuple[float, float]:
    """Re-house each damaged spread in an interchangeable layout that is clean.

    `choose_cleaner_fold` can only offer what the candidate list holds, and for
    a group of several spreads that is less than it looks: `process_group_lists`
    builds candidates with `itertools.product`, which varies the *last* spread
    fastest, so the leading candidates all share one layout for the first
    spread. On 53316807 the two one-photo spreads of group `1_other` had the
    damaged photo in the first, and every candidate within reach repeated the
    same design for it.

    This does not search candidates at all. It takes the spread as the search
    settled it -- the same photos, the same page split -- and looks for another
    layout of the same shape (`layout_shape`), which by construction scores the
    same. Among those it takes the one whose fold falls clearest of the faces.

    Returns (cost before, cost after) over the whole group layout.
    """
    shapes = None
    before_total = after_total = 0.0

    for spread in group_layout.spreads_layouts:
        before = spread_fold_cost(spread, layout_id2data, box_id2data, photo_info, album_ar)
        before_total += before
        if before <= 0:
            after_total += before
            continue

        if shapes is None:
            shapes = _shape_index(layout_id2data, box_id2data)
        shape = layout_shape(spread.layout_idx, layout_id2data, box_id2data)

        best_idx, best_cost = spread.layout_idx, before
        for layout_idx in shapes.get(shape, ()):
            if layout_idx == spread.layout_idx:
                continue
            trial = SingleSpreadLayout(layout_idx=layout_idx,
                                       left_page_photo_idxs=set(spread.left_page_photo_idxs),
                                       right_page_photo_idxs=set(spread.right_page_photo_idxs),
                                       number_of_squares=spread.number_of_squares)
            trial.resolve_photos(photos)
            trial.assign_photos_order(layout_id2data, box_id2data, merge_pages=False)
            cost = spread_fold_cost(trial, layout_id2data, box_id2data, photo_info, album_ar)
            if cost < best_cost - 1e-9:
                best_idx, best_cost = layout_idx, cost
                if cost <= 0:
                    break

        if best_idx != spread.layout_idx:
            if logger:
                logger.info(
                    f"fold: spread moved from layout {layout_id2data[spread.layout_idx]['layout_id']} "
                    f"to {layout_id2data[best_idx]['layout_id']} (same shape), "
                    f"faces on the fold {before:.2f} -> {best_cost:.2f}")
            spread.layout_idx = best_idx
            spread.resolve_photos(photos)
            spread.assign_photos_order(layout_id2data, box_id2data, merge_pages=False)
        after_total += best_cost

    return before_total, after_total


def choose_cleaner_fold(candidates: List[Any], best: Any, photos: List[Any],
                        layout_id2data: Dict[int, Any],
                        box_id2data: Dict[Tuple[int, int], Any],
                        photo_info: Dict[Any, Dict[str, Any]],
                        album_ar: float = 2.0, logger=None) -> Optional[Any]:
    """A rival candidate that keeps the fold off the faces, or None to keep `best`.

    `best` -- the layout the search scored highest -- is already resolved onto
    its boxes; the rivals are resolved here, in weight order, and only while
    they are worth considering. A rival has to be within `min_weight_ratio` of
    the best weight to be eligible at all, so this never trades the layout the
    search actually wanted for a fold improvement: it picks between layouts the
    search already called comparable. The walk stops at the first candidate
    with a clean fold, and at the first one too light to qualify -- the list is
    sorted by weight, so everything after it is lighter still.

    None when the feature is off, when there is nothing to fix, or when no
    rival does better, and the caller keeps what it had.
    """
    if not enabled() or not photo_info or not candidates:
        return None

    base_cost = group_layout_fold_cost(best, layout_id2data, box_id2data, photo_info, album_ar)
    if base_cost <= 0:
        return None

    config = settings()
    limit = max(1, int(config.get('candidates', 30)))
    ratio = float(config.get('min_weight_ratio', 0.5))
    best_weight = getattr(best, 'weight', None)
    floor = best_weight * ratio if best_weight and best_weight > 0 else None

    chosen, chosen_cost = None, base_cost
    for candidate in candidates[1:limit]:
        weight = getattr(candidate, 'weight', None)
        if floor is not None and (weight is None or weight < floor):
            break
        candidate.resolve_and_order(photos, layout_id2data, box_id2data, merge_pages=False)
        cost = group_layout_fold_cost(candidate, layout_id2data, box_id2data, photo_info, album_ar)
        if cost < chosen_cost - 1e-9:
            chosen, chosen_cost = candidate, cost
            if cost <= 0:
                break

    if chosen is None:
        if logger:
            logger.debug(f"fold: no candidate within {ratio:.0%} of the best weight improves on "
                         f"{base_cost:.2f} faces on the fold; keeping the best-scoring layout")
        return None

    if logger:
        logger.info(f"fold: took a layout costing {chosen.weight:.3g} against {best_weight:.3g} "
                    f"to bring the faces on the fold from {base_cost:.2f} to {chosen_cost:.2f}")
    return chosen
