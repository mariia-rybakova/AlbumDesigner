"""A small wedding gallery goes into the album nearly whole -- gradually.

Selection is built to choose: a per-category budget from the focus profile,
then CP-SAT under a candidate gate, admission costs, near-duplicate and repeat
charges, coverage windows and identity penalties. On a gallery of a few
thousand frames that is the job. On one of a hundred it throws away photos the
album needs -- 53812098 (109 photos) came back with 36, while the photographer
had delivered every one of them for a reason.

**Nothing switches.** `weight(n)` is 1 up to `full_up_to` photos and falls
linearly to 0 at `none_from`, and everything it touches moves continuously
with it, so a gallery one photo over a threshold is composed by the same rules
as one under it:

* the **budget** -- each class's allowance moves ``weight`` of the way from
  what the profile gave it to everything it holds (`blended_budget`), so the
  picker is asked for more photos and layout plans for the album that will
  arrive. Held under the design's spread maximum times
  `max_photos_per_spread`: without that, 105 photos went into the 16 spreads
  53812098's design allows and layout put 24 of them on one.
* the **penalties** -- `CpSatPicker.penalty_scale` is ``1 - weight`` and
  multiplies every charge that holds a photo out (admission bar, greyscale,
  near-duplicate, repeat, coverage window, identity contradiction). The
  charges that push toward the allowance, and the hard rules (an exact
  repeat, a contradicted identity), are left as they are.
"""

from __future__ import annotations

from typing import Dict, Tuple

from src.pipeline.contracts import AlbumContext
from utils.configs import CONFIGS


def settings() -> dict:
    return CONFIGS.get('small_gallery_album', {}) or {}


def weight(n: int) -> float:
    """1 up to `full_up_to` photos, 0 from `none_from`, linear between."""
    config = settings()
    if not config.get('enabled', False):
        return 0.0
    full = float(config.get('full_up_to', 100))
    none = float(config.get('none_from', 200))
    if n <= full:
        return 1.0
    if n >= none or none <= full:
        return 0.0
    return (none - n) / (none - full)


def weight_for(context: AlbumContext) -> float:
    """The weight for this request; 0 where selection is not the AI's."""
    facts = getattr(context, 'facts', None)
    selection = getattr(context, 'selection', None)
    if facts is None or not facts.is_wedding or selection is None or selection.manual:
        return 0.0
    if context.photos is None:
        return 0.0
    return weight(len(context.photos))


def blended_budget(images: Dict[str, int], spreads: Dict[str, float],
                   available: Dict[str, int], lookup_table: Dict[str, tuple], w: float,
                   ceiling: int = None) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Each class's allowance moved ``w`` of the way to everything it holds.

    The growth is scaled down, never the base, when the total would pass
    ``ceiling`` -- continuous in ``w`` either way.
    """
    base_total = sum(int(images.get(e, 0)) for e in available)
    growth = {e: w * max(0, count - int(images.get(e, 0))) for e, count in available.items()}
    room = sum(growth.values())
    if ceiling is not None and room > 0 and base_total + room > ceiling:
        factor = max(0.0, ceiling - base_total) / room
        growth = {e: g * factor for e, g in growth.items()}

    out_images, out_spreads = dict(images), dict(spreads)
    for event, extra in growth.items():
        grown = int(round(int(images.get(event, 0)) + extra))
        added = grown - int(images.get(event, 0))
        if added <= 0:
            continue
        entry = lookup_table.get(event)
        per_spread = entry[0] if isinstance(entry, (tuple, list)) else entry
        try:
            per_spread = max(float(per_spread), 1.0)
        except (TypeError, ValueError):
            per_spread = 4.0
        out_images[event] = grown
        out_spreads[event] = float(spreads.get(event, 0.0)) + added / per_spread
    return out_images, out_spreads
