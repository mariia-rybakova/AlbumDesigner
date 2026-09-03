"""Turning the focus profile into a per-category photo and page budget.

Ported step for step from `calculate_optimal_selection`, which stays in place
untouched as the reference `tests/test_selection_equivalence.py` measures
against. What is new here is the treatment of the ceremony's `yes` classes, and
the split into named steps so that treatment has somewhere to sit.

Two numbers come out per category:

``images``
    How many photos to pick. This is the one that decides the album -- it is
    what `select.pick` spends -- so every rule here ultimately moves photos.
``spreads``
    How much of the album the category should occupy. Diagnostic today, since
    nothing downstream reads `spreads_dict`, but it is the currency the fill-up
    arithmetic is done in, so it still has to be right.

### The ceremony's `yes` classes

`enrich.ceremony_anchor` finds the moments a wedding is actually remembered by:
the kiss, the processionals, the send-off. Three of the four carry `yes` in
`files/focus_csv.csv` rather than a percentage, which in the original
arithmetic means: one photo if it happened, no page of its own, and a surplus
free for the fill-up loop to draw on. That last part is the problem. A send-off
is a burst -- fifteen frames against a LUT base of two -- so its surplus reads
as seven spare pages, and an album short of material would keep taking from it
until the send-off was most of what was left.

So the group is settled before redistribution and then kept out of it:

* **The album is short of pages.** Two or more of these moments together are
  worth a page of their own, and they supply exactly one of the missing pages.
  Not more, however large the burst.
* **The album is full.** They add nothing to its length. Their photos come out
  of the ceremony's own allowance, so a gallery with a send-off and two
  processionals does not quietly run three photos longer than one without.

Which classes are in the group is read from the focus profile, not fixed here.
Give `send off` a percentage in `focus_csv.csv` and it becomes an ordinary
category again, budgeted like any other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from src.pipeline.enrich.ceremony_anchor import (
    BRIDE_AISLE,
    GROOM_AISLE,
    MAY_KISS_BRIDE,
    SEND_OFF,
)
from src.selection.ai_wedding_selection import define_min_max_spreads
from utils.configs import CONFIGS

#: The ceremony's own classes, in the order a wedding produces them. Membership
#: of the `yes` group is decided per request from the focus profile.
CEREMONY_EVENT_CLASSES = (BRIDE_AISLE, GROOM_AISLE, MAY_KISS_BRIDE, SEND_OFF)

#: The category the `yes` photos are charged to when the album needs no filling.
CEREMONY = "ceremony"

#: Ceiling and floor the original applies to every density-scaled LUT base.
_MAX_PHOTOS_PER_SPREAD = 24
_MIN_PHOTOS_PER_SPREAD = 1


@dataclass
class Allocation:
    """The budget, plus enough of the reasoning to explain it in a log line."""

    #: {category: how many photos to pick}
    images: Dict[str, int] = field(default_factory=dict)
    #: {category: how much of the album it should occupy}
    spreads: Dict[str, float] = field(default_factory=dict)
    min_total_spreads: Optional[int] = None
    max_total_spreads: Optional[int] = None
    #: Density-scaled photos-per-spread table.
    lookup_table: Dict[str, tuple] = field(default_factory=dict)

    #: Pages the album was short of before anything was redistributed.
    shortfall: int = 0
    #: ...and after.
    unfilled: int = 0
    #: The ceremony `yes` classes this gallery actually has.
    ceremony_yes: List[str] = field(default_factory=list)
    #: Whether they were granted their page.
    ceremony_page_granted: bool = False
    #: Photos charged back to `ceremony` when they were not.
    charged_to_ceremony: int = 0
    #: Every category budgeted as `yes` on this gallery, ceremony or otherwise.
    yes_categories: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.ceremony_yes:
            return (f"{sum(self.images.values())} photos over "
                    f"{self.min_total_spreads}-{self.max_total_spreads} spreads, "
                    f"{self.unfilled} unfilled")
        verdict = ("granted a page" if self.ceremony_page_granted
                   else f"charged {self.charged_to_ceremony} photos to {CEREMONY}")
        return (f"{sum(self.images.values())} photos over "
                f"{self.min_total_spreads}-{self.max_total_spreads} spreads, "
                f"{self.shortfall} short -> {self.unfilled} unfilled; "
                f"ceremony yes {self.ceremony_yes} {verdict}")


@dataclass
class Settlement:
    """What the ceremony `yes` group ended up costing.

    Exactly one of the two is ever non-zero: the group either supplies a page
    or gives its photos back to the ceremony.
    """

    #: Pages it supplies toward the shortfall. One, or none.
    pages: int = 0
    #: Photos taken out of the ceremony's allowance instead.
    photos_charged: int = 0


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------


def density_scaled(lookup_table: Dict[str, tuple], density: int) -> Dict[str, tuple]:
    """Photos-per-spread, scaled by the requested density."""
    factor = CONFIGS['density_factors'].get(density, 1.0)
    return {
        event: (min(_MAX_PHOTOS_PER_SPREAD,
                    max(_MIN_PHOTOS_PER_SPREAD, base * factor)), std)
        for event, (base, std) in lookup_table.items()
    }


def budget_each(focus_table: dict, available: Dict[str, int], lut: Dict[str, tuple],
                target: int) -> None:
    """First pass: a share of the album each, trimmed to what the gallery has.

    Mutates ``focus_table``, adding ``spreads``, ``photos``, ``miss``,
    ``miss_spreads``, ``over_photos`` and ``over_spreads`` to every entry --
    ``miss`` being what the category was promised and cannot supply, ``over``
    being what it has spare.

    A `yes` (or `no`) category asks for a single photo and no page at all, so it
    never contributes to the shortfall.
    """
    # Normalised over the categories the gallery *has*, not the whole profile.
    #
    # The profile weights sum to 107%, and a quarter to a third of that is
    # routinely spent on categories a given wedding has none of -- measured
    # across the validation galleries: 23%, 25%, 26%, 27%, 31%, 33%. Counting
    # those in meant the present categories asked for only ~70% of the album
    # between them, and the missing third came straight back as `miss_spreads`,
    # i.e. shortfall. That is why every gallery was "5-7 pages short" and why
    # the fill loop then had to reach down the file for `other` and `None`,
    # which carry 0% and were never meant to fill anything.
    #
    # This cannot be done before `enrich.same_sex_couple`. On a same-sex
    # gallery `groom` is exactly such an absent category, and redistributing
    # its 12% is what would leave the second partner unrepresented -- the split
    # has to give her a populated class first.
    present_only = bool(CONFIGS.get('budget_normalise_present_only', True))
    total_value = sum(
        config['value']
        for event, config in focus_table.items()
        if isinstance(config, dict) and isinstance(config.get('value'), (int, float))
        and (available.get(event, 0) > 0 or not present_only)
    )
    if total_value <= 0:
        # Nothing the profile weights is in this gallery at all. Fall back to
        # the whole profile rather than divide by zero; every percentage
        # category will come up short and the `yes` ones will carry the album.
        total_value = sum(
            config['value']
            for config in focus_table.values()
            if isinstance(config, dict) and isinstance(config.get('value'), (int, float))
        ) or 1.0

    for event, config in focus_table.items():
        if not (isinstance(config, dict) and 'value' in config):
            continue
        have = available.get(event, 0)

        if isinstance(config['value'], str):
            config['spreads'] = 0
            config['photos'] = 1
            config['miss'] = max(0, config['photos'] - have)
            config['miss_spreads'] = 0
            if config['miss'] > 0:
                config['photos'] = 0
        else:
            config['spreads'] = config['value'] / total_value * target
            config['photos'] = config['spreads'] * lut[event][0]
            config['miss'] = max(0, config['photos'] - have)
            config['miss_spreads'] = round(config['miss'] / lut[event][0])
            if present_only and have == 0:
                # An event this wedding simply did not have is not a shortfall.
                # Normalising over the present categories is only half the job:
                # an absent one still claims a share of the target and misses
                # all of it, and with a smaller denominator it claims a *bigger*
                # one -- measured, that pushed the shortfall up rather than down
                # (10 -> 13 pages on one gallery) and left `other` and `None`
                # absorbing exactly as much as before.
                config['miss_spreads'] = 0
            if config['miss'] > 0:
                config['photos'] = config['photos'] - config['miss']
                config['spreads'] = round(config['photos'] / lut[event][0])

        config['over_photos'] = max(0, have - config['photos'])
        config['over_spreads'] = config['over_photos'] / lut[event][0]


def yes_categories(focus_table: dict, available: Dict[str, int]) -> List[str]:
    """Categories the profile budgets as `yes`, and that the gallery has.

    The single definition of "yes" in the selection stage. A `yes` category is
    promised one photo if the thing happened and no page of its own, so there is
    nothing for the ranked picker to weigh -- which is why `select.preselect`
    resolves them outright.
    """
    return [
        event for event, config in focus_table.items()
        if isinstance(config, dict)
        and isinstance(config.get('value'), str)
        and config['value'].strip().lower() == 'yes'
        and available.get(event, 0) > 0
    ]


def ceremony_yes_classes(focus_table: dict, available: Dict[str, int]) -> List[str]:
    """The ceremony moments that are budgeted as `yes` and actually turned up."""
    budgeted_as_yes = set(yes_categories(focus_table, available))
    return [event for event in CEREMONY_EVENT_CLASSES if event in budgeted_as_yes]


def settle_ceremony_yes(focus_table: dict, present: List[str], available: Dict[str, int],
                        lut: Dict[str, tuple], shortfall: int,
                        logger=None) -> "Settlement":
    """Decide what the ceremony's `yes` classes cost the album.

    See the module docstring for why the two cases differ. Either way they come
    out of the redistribution that follows, by having their surplus zeroed: it
    is far too large to leave in the fill loop's reach.
    """
    if not present:
        return Settlement()

    for event in present:
        focus_table[event]['over_photos'] = 0
        focus_table[event]['over_spreads'] = 0.0

    enough = len(present) >= CONFIGS['ceremony_yes_min_classes']
    if shortfall >= 1 and enough:
        _fill_one_page(focus_table, present, available, lut)
        if logger:
            logger.info(
                f"Ceremony highlights {present} fill one of {shortfall} missing spreads"
            )
        return Settlement(pages=1)

    if logger and not enough:
        logger.debug(
            f"Ceremony highlights {present}: only {len(present)} class(es), "
            f"below the {CONFIGS['ceremony_yes_min_classes']} worth a page"
        )
    return Settlement(photos_charged=_charge_to_ceremony(focus_table, present, logger) or 0)


def _fill_one_page(focus_table: dict, present: List[str], available: Dict[str, int],
                   lut: Dict[str, tuple]) -> Dict[str, int]:
    """One page's worth of photos, spread across the moments that turned up.

    A frame each first -- that is what `yes` means, and a moment that happened
    and is not in the album is the failure this whole group exists to avoid.
    Whatever the page still has room for goes to the moments with the most to
    show, since a send-off burst can carry a page on its own where a two-frame
    processional cannot.

    The page is charged across them in proportion, so the group totals exactly
    one spread however it was divided.
    """
    capacity = max(int(round(lut[event][0])) for event in present)
    picks = {event: 1 for event in present}

    for event in sorted(present, key=lambda e: -available.get(e, 0)):
        while sum(picks.values()) < capacity and picks[event] < available.get(event, 0):
            picks[event] += 1

    total = sum(picks.values())
    for event, count in picks.items():
        focus_table[event]['photos'] = count
        focus_table[event]['spreads'] = count / total
    return picks


def _charge_to_ceremony(focus_table: dict, present: List[str], logger=None) -> Optional[int]:
    """Take the `yes` photos out of the ceremony's allowance, not the album's.

    The ceremony keeps its pages: these moments *are* ceremony, so they occupy
    the space it was already given. Only the photo count moves.

    Returns the photos charged, or None when there is no ceremony to charge --
    a gallery with a send-off and no ceremony class, which the detectors do not
    produce but nothing forbids.
    """
    cost = int(sum(focus_table[event]['photos'] for event in present))
    ceremony = focus_table.get(CEREMONY)

    if cost <= 0:
        return 0
    if not isinstance(ceremony, dict) or 'photos' not in ceremony:
        if logger:
            logger.warning(
                f"Ceremony highlights {present} cost {cost} photos with no "
                f"'{CEREMONY}' budget to charge them to; album grows instead"
            )
        return None

    before = ceremony['photos']
    ceremony['photos'] = max(0, before - cost)
    if logger:
        logger.info(
            f"Ceremony highlights {present} charged {cost} photos to "
            f"'{CEREMONY}' ({before:.1f} -> {ceremony['photos']:.1f}); album keeps its length"
        )
    return cost


def redistribute(focus_table: dict, lut: Dict[str, tuple], shortfall: int) -> int:
    """Hand unfilled pages to the categories that have photos to spare.

    One page to each category with a spare page's worth, walking the table, and
    round again until the album is full. Returns what is still unfilled; the
    original's ``> 1`` thresholds are kept, so a single missing page is left
    alone.

    **A `yes` category sits out the first round.** `yes` means one photo if the
    thing happened -- that is what it is worth when the album can be built from
    the categories the profile actually weighted. Only once a full walk of those
    has failed to fill the album is a `yes` category worth a page of its own.

    Without that, `yes` competed on the first walk like anything else, and on a
    gallery short of material the album filled with whatever sat high in the
    file: measured on 53273032, `settings` and `food` each took a full page on
    the *first* pass, `food` reaching 5 photos out of the 6 it had. Both now
    wait, and are reached only if the shortfall survives the first round.

    The ceremony highlights are already out of both rounds -- `settle_ceremony_yes`
    zeroes their surplus, because a send-off burst is large enough to fill
    several pages on its own.
    """
    surplus = sum(config['over_spreads'] for config in focus_table.values())
    first_round = True

    while surplus > 1 and shortfall > 1:
        for event, config in focus_table.items():
            if first_round and isinstance(config['value'], str):
                continue  # a 'yes' event is one photo until the second round
            if config['over_spreads'] > 1 and shortfall > 1:
                config['over_spreads'] -= 1
                config['over_photos'] -= lut[event][0]
                config['photos'] += lut[event][0]
                config['spreads'] += 1
                surplus -= 1
                shortfall -= 1
        if not any(config['over_spreads'] > 1 for config in focus_table.values()):
            break
        first_round = False

    return shortfall


# --------------------------------------------------------------------------
# The whole thing
# --------------------------------------------------------------------------


def allocate(available: Dict[str, int], focus_table: dict, lookup_table: Dict[str, tuple],
             density: int, photos: pd.DataFrame, logger=None) -> Allocation:
    """The per-category budget for one request.

    ``focus_table`` is mutated -- it is the working state, as it was in the
    original. `load_event_mapping` reads a fresh copy per request, so nothing
    leaks between galleries.
    """
    min_total, max_total = define_min_max_spreads(photos, focus_table, available, logger)
    if min_total is None:
        raise ValueError("define_min_max_spreads could not size the album")

    lut = density_scaled(lookup_table, density)
    budget_each(focus_table, available, lut, target=min_total)

    shortfall = max(0, sum(config['miss_spreads'] for config in focus_table.values()))

    present = ceremony_yes_classes(focus_table, available)
    settled = settle_ceremony_yes(focus_table, present, available, lut, shortfall, logger)

    unfilled = redistribute(focus_table, lut, max(0, shortfall - settled.pages))

    allocation = Allocation(
        images={event: round(focus_table[event]['photos']) for event in available},
        spreads={event: focus_table[event]['spreads'] for event in available},
        min_total_spreads=min_total,
        max_total_spreads=max_total,
        lookup_table=lut,
        shortfall=shortfall,
        unfilled=unfilled,
        ceremony_yes=present,
        yes_categories=yes_categories(focus_table, available),
        ceremony_page_granted=bool(settled.pages),
        charged_to_ceremony=settled.photos_charged,
    )

    if logger and unfilled > 1:
        logger.warning(f"Unable to fill desired spreads. Total miss spreads: {unfilled}")
    return allocation
