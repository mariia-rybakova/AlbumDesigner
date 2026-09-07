"""Keeping the same shot out of the album twice.

A photographer often uploads a second copy of their best frames with a
different treatment -- classically black and white, sometimes a blue or brown
tone. On production gallery 53273032 both batches were bulk-uploaded in order,
so the copies sit at a constant photo-id offset (`+117522`, `+118210`): **512
pairs**, and fifteen of them reached a 100-photo album, one spread showing the
same dance frame in colour and in black and white.

Selection could not see it. The picker splits each category into colour and
greyscale candidates and fills from the two independently, so a frame and its
own re-upload are never compared.

### Nothing tells the two copies apart photo by photo

Two obvious approaches both fail, and it is worth recording why so nobody
retries them. Measured on that gallery's 512 known pairs:

*CLIP similarity.* The same shot in colour vs greyscale scores **0.777-0.930**;
*different* frames of the same moment score **0.870-0.954**. Desaturating an
image moves it as far in CLIP space as photographing a different instant does,
so the ranges overlap almost entirely and no threshold separates them.

*Composition.* Background segmentation is not pixel-deterministic, so a known
re-upload pair differs by as much as a burst pair does -- centroids 0.445 vs
0.452 on one confirmed twin -- and `n_faces` came out 37 against 67 on one
same-second pair. Neither is a shot fingerprint.

An identical capture second plus an identical aspect ratio does hold for all
512 pairs, because a re-export preserves the EXIF. But on its own it is far too
eager: a camera at three frames a second produces several genuinely different
photos in one second at the same aspect ratio, and on the other validation
galleries that rule wanted to drop 46, 32 and 11 real frames.

### One pair-level signal does work: the colour flag

A burst does not change treatment between frames taken in the same second. So a
same-second, same-aspect group holding **both a colour and a greyscale frame**
is a re-export, and that needs no gallery-level judgement at all. Measured
across seven galleries: 505 of 505 such groups on the wholesale-duplicated one,
61 on a second (51 at adjacent photo ids), 35 and 20 on two more, and **zero**
on the three with no re-uploads.

What the flag cannot catch is a *toned* copy -- blue or brown rather than grey
-- because that is still classified as colour and its flag matches the
original's. For those, only the bulk signal below works.

### And a second judgement about the gallery

What actually marks 53273032 is the *regularity*: 512 groups of exactly two,
covering essentially the whole gallery. That is a photographer uploading their
set twice, and it is a property of the gallery. Ordinary galleries look nothing
like it:

| gallery | photos sitting in duplicate `(time, aspect)` groups |
|---|---|
| 53273032 | **~100%** |
| 49994361 | 3.8% |
| 49995684 | 3.1% |
| 53496523 | 2.1% |
| 47981912 | 0% |

Two orders of magnitude of daylight, so ``min_gallery_share`` sits at 0.5 and
the rule fires only on a gallery that was systematically duplicated. Everywhere
else it does nothing at all, and a same-second burst keeps every frame.

The cost of drawing the line there is that a photographer who re-uploads only a
handful of favourites in black and white is not caught. That is the direction
to err in: doing nothing leaves one redundant spread, while guessing wrong
deletes photos the album should have had.

A gallery whose EXIF is unusable needs no special case either. Two validation
galleries carry **2 distinct `image_time` values across 528 and 582 photos**;
that puts nearly everything in one enormous group, and only groups small enough
to be a re-upload set are counted at all.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.pipeline.contracts import Col
from utils.configs import CONFIGS

#: Aspect ratios are floats off a protobuf; compare them at a sane precision.
_AR_PRECISION = 4


def shot_groups(photos: pd.DataFrame, logger=None) -> Dict[Tuple, List[Any]]:
    """``{shot key: [image_id, ...]}`` for shots that are in twice.

    Two routes in, because two different things produce a second copy:

    **A differing colour flag inside one same-second group.** A burst does not
    change treatment between frames taken in the same second, so a group at one
    capture second and one aspect ratio holding both a colour and a greyscale
    frame is a re-export. This needs no gallery-level judgement and fires on any
    gallery. Measured: 505 of 505 groups on the wholesale-duplicated gallery,
    61 on another (51 of them at adjacent photo ids), 35 and 20 on two more --
    and **zero** on the three galleries with no re-uploads.

    **A gallery duplicated wholesale.** When most of the gallery sits in such
    groups the photographer uploaded their whole set again, and then the colour
    flag is not needed -- which is what catches a *toned* copy, since a blue- or
    brown-toned export is still classified as colour and its flag matches the
    original's.

    The gap between the two: a photographer who re-exports only a handful of
    favourites with a colour tone rather than to grey. That pair shares a
    second, an aspect ratio and a colour flag, and nothing in the photo table
    separates it from a burst -- CLIP cosine and composition both overlap
    completely, see the module docstring. It is left alone.
    """
    settings = CONFIGS.get('near_duplicates') or {}
    if not settings.get('enabled', True):
        return {}

    needed = (Col.IMAGE_ID, Col.IMAGE_TIME, Col.IMAGE_AS)
    if photos is None or photos.empty or any(c not in photos.columns for c in needed):
        return {}

    max_copies = int(settings.get('max_copies_per_shot', 4))
    min_share = float(settings.get('min_gallery_share', 0.5))

    grouped: Dict[Tuple, List[Any]] = {}
    for image_id, time, aspect in zip(
        photos[Col.IMAGE_ID], photos[Col.IMAGE_TIME], photos[Col.IMAGE_AS]
    ):
        key = _shot_key(time, aspect)
        if key is not None:
            grouped.setdefault(key, []).append(image_id)

    # Only groups small enough to be a re-upload set. A gallery with unusable
    # EXIF puts nearly everything on one timestamp; that group says nothing
    # about which shot is which.
    candidates = {
        key: members for key, members in grouped.items()
        if 1 < len(members) <= max_copies
    }
    if not candidates:
        return {}

    covered = sum(len(members) for members in candidates.values())
    share = covered / len(photos)

    if share >= min_share:
        if logger:
            logger.info(
                "Duplicate shots: {}/{} photos ({:.1%}) sit in {} same-second groups -- "
                "this gallery was uploaded more than once".format(
                    covered, len(photos), share, len(candidates))
            )
        return candidates

    recoloured = {
        key: members for key, members in candidates.items()
        if _mixed_treatment(photos, members)
    }
    if logger:
        if recoloured:
            logger.info(
                "Duplicate shots: {} of {} same-second groups hold both a colour and a "
                "greyscale copy of one shot; the other {} are bursts and are left "
                "alone".format(len(recoloured), len(candidates),
                               len(candidates) - len(recoloured))
            )
        else:
            logger.debug(
                "Duplicate shots: {}/{} photos ({:.1%}) share a capture second, below "
                "{:.0%}, and none mixes colour with greyscale -- all bursts".format(
                    covered, len(photos), share, min_share)
            )
    return recoloured


def _mixed_treatment(photos: pd.DataFrame, members: List[Any]) -> bool:
    """Whether one same-second group holds both a colour and a greyscale frame.

    `image_color` is the segmentation's `colorEnum`: 0 is greyscale, 1 colour.
    """
    if Col.IMAGE_COLOR not in photos.columns:
        return False
    flags = photos.loc[photos[Col.IMAGE_ID].isin(members), Col.IMAGE_COLOR]
    return flags.apply(lambda v: 0 if v == 0 else 1).nunique() > 1


def _shot_key(time, aspect) -> Optional[Tuple]:
    """``(capture second, aspect ratio)``, or None when there is no real time.

    A missing or non-positive `image_time` is the protobuf's way of saying the
    EXIF had nothing; `dateTaken <= 0` is treated as absent everywhere else too.
    """
    if pd.isna(time) or pd.isna(aspect):
        return None
    try:
        seconds = int(time)
        ratio = round(float(aspect), _AR_PRECISION)
    except (TypeError, ValueError):
        return None
    return None if seconds <= 0 else (seconds, ratio)
