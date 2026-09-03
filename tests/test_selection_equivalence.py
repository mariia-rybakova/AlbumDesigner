"""Equivalence tests for the selection decomposition.

`src.selection.ai_wedding_selection.smart_wedding_selection` is the monolith the
`src/pipeline/select` substages were extracted from. It is untouched, so it
doubles as the reference implementation: for the same synthetic gallery both
paths must choose the same photos, in the same order, with the same budget.

**Every deliberate departure is switched off for these comparisons** -- see
`as_the_monolith`. It is a deliberate
behaviour change -- it commits hand-picked photos, identity coverage, the covers
and the `yes` categories before any ranking, which the monolith did not -- so
holding it to the monolith would be asserting the change had not been made.
Switching it off and still matching is the stronger statement available here: it
says the decomposition reproduces the monolith exactly, and that every departure
comes from the constraints rather than from drift in the machinery underneath.
What the constraints themselves do is `tests/test_preselect.py`.

Run from the repo root (the focus profile is read from a relative path)::

    python -m pytest tests/test_selection_equivalence.py -v
    python tests/test_selection_equivalence.py          # no pytest needed
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import AlbumContext, build_select  # noqa: E402
from src.pipeline.contracts import AiHints, GalleryFacts  # noqa: E402
from src.selection.ai_wedding_selection import smart_wedding_selection  # noqa: E402
from utils.configs import CONFIGS  # noqa: E402

#: Every preselect constraint off, so the pipeline runs the monolith's rules.
NO_CONSTRAINTS = {'user_picks': False, 'identities': False, 'key_pages': False,
                  'yes_categories': False}


@contextmanager
def as_the_monolith():
    """Turn off every deliberate departure, so the comparison means something.

    Three so far, each a change the monolith did not make, so holding it to
    them would be asserting they had not been made:

    * `select.preselect`'s constraints
    * `budget_normalise_present_only` -- the profile's percentages spread over
      the categories the gallery has rather than over the whole profile
    * `bride_prep_by_identity` -- the getting-ready subject chosen by `bride_id`
      rather than by a substring match on the subquery text
    """
    original = CONFIGS['preselect']
    normalisation = CONFIGS.get('budget_normalise_present_only', True)
    prep = CONFIGS.get('bride_prep_by_identity', True)
    CONFIGS['preselect'] = {**original, **NO_CONSTRAINTS}
    CONFIGS['budget_normalise_present_only'] = False
    CONFIGS['bride_prep_by_identity'] = False
    try:
        yield
    finally:
        CONFIGS['preselect'] = original
        CONFIGS['budget_normalise_present_only'] = normalisation
        CONFIGS['bride_prep_by_identity'] = prep

BRIDE_ID = 101
GROOM_ID = 202

#: Enough categories to exercise every strategy, plus the early exits.
CATEGORIES = [
    ("bride and groom", 40),
    ("bride", 22),
    ("groom", 18),
    ("bride party", 14),
    ("groom party", 12),
    ("ceremony", 45),
    ("dancing", 60),
    ("walking the aisle", 16),
    ("first dance", 12),
    ("cake cutting", 9),
    ("full party", 10),
    ("portrait", 30),
    ("very large group", 11),
    ("speech", 13),
    ("parents portrait", 12),
    ("bride getting dressed", 20),
    ("getting hair-makeup", 14),
    ("detail", 25),
    ("settings", 18),
    ("food", 10),
    ("rings", 4),
    ("accessories", 5),
    ("wedding dress", 6),
    ("invite", 2),
    ("other", 8),
]

FORMAL_PORTRAIT_QUERY = (
    "formal studio-style wedding portrait, bride and groom centered, attendants standing still, "
    "bouquets held, symmetrical but there are no people behind them"
)
PARENT_CATEGORIES = [
    "bride and groom with parents",
    "bride with her parents",
    "groom with his parents",
]


def make_gallery(seed: int = 7) -> pd.DataFrame:
    """A synthetic wedding gallery with every column selection reads."""
    rng = np.random.default_rng(seed)
    rows = []
    image_id = 900_000
    clock = 1_700_000_000  # a fixed epoch so timestamps are reproducible

    for category, count in CATEGORIES:
        for i in range(count):
            image_id += 1
            # Cluster photos in time so the temporal narrowing has real groups
            # to find rather than a uniform smear.
            clock += int(rng.integers(20, 200)) + (600 if i % 7 == 0 else 0)

            people = _people_for(category, i, rng)
            embedding = rng.normal(size=32)
            embedding = embedding / np.linalg.norm(embedding)

            rows.append({
                "image_id": image_id,
                "embedding": embedding,
                "model_version": 2,
                "image_class": int(rng.integers(0, 12)),
                "cluster_label": int(rng.integers(0, 6)),
                "cluster_class": int(rng.integers(0, 12)),
                "cluster_context": category,
                "ranking": float(rng.random()),
                "image_order": int(rng.integers(1, 500)),
                "persons_ids": people,
                "main_persons": [BRIDE_ID, GROOM_ID],
                "bride_id": BRIDE_ID,
                "groom_id": GROOM_ID,
                "n_faces": len(people),
                "number_bodies": len(people),
                "image_time": clock,
                "image_as": 1.5 if i % 3 else 0.66,
                "image_color": 0 if i % 11 == 0 else 1,
                "image_orientation": "landscape" if i % 3 else "portrait",
                "scene_order": i // 8,
                "image_query_content": category,
                "image_subquery_content": _subquery_for(category, i),
                "parent_category": (
                    PARENT_CATEGORIES[i % 3] if category == "parents portrait" else None
                ),
                "user_rating": int(rng.integers(0, 6)),
            })

    return pd.DataFrame(rows)


def _people_for(category: str, i: int, rng) -> list:
    guest = int(rng.integers(300, 340))
    if category in ("bride", "bride getting dressed", "getting hair-makeup"):
        return [BRIDE_ID] if i % 4 else [BRIDE_ID, guest]
    if category == "groom":
        return [GROOM_ID] if i % 4 else [GROOM_ID, guest]
    if category in ("bride and groom", "first dance", "cake cutting", "walking the aisle"):
        return [BRIDE_ID, GROOM_ID] if i % 5 else [BRIDE_ID]
    if category == "bride party":
        return [BRIDE_ID, guest, guest + 1]
    if category == "groom party":
        return [GROOM_ID, guest, guest + 1]
    if category == "parents portrait":
        return [BRIDE_ID, GROOM_ID, 401 + (i % 4), 411 + (i % 4)]
    if category in ("portrait", "very large group", "full party"):
        return [BRIDE_ID, GROOM_ID] + [guest + k for k in range(i % 5)]
    if category in ("detail", "settings", "food", "rings", "invite", "accessories", "wedding dress"):
        return []
    return [guest, guest + 1] if i % 2 else [guest]


def _subquery_for(category: str, i: int) -> str:
    if category == "portrait" and i % 3 == 0:
        return FORMAL_PORTRAIT_QUERY
    if category in ("bride getting dressed", "getting hair-makeup"):
        return "bride getting ready with her dress" if i % 2 else "hairstylist working"
    return f"{category} shot {i % 4}"


def _quiet_logger() -> logging.Logger:
    logger = logging.getLogger("selection-equivalence")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    return logger


def run_reference(df, *, ten_photos, person_ids, focus, density, artificial, rating):
    """The pre-refactor monolith."""
    return smart_wedding_selection(
        df.copy(), list(ten_photos), list(person_ids), list(focus), [],
        density, artificial, _quiet_logger(), rating=rating,
    )


def run_pipeline(df, *, ten_photos, person_ids, focus, density, artificial, rating):
    """The decomposed substages, driven the way the service drives them."""
    logger = _quiet_logger()

    context = AlbumContext(
        logger=logger,
        request={'rating': rating or []},
        photos=df.copy(),
        available_photo_ids=[],
        hints=AiHints(
            photo_ids=list(ten_photos),
            person_ids=list(person_ids),
            focus=list(focus),
            # Blank subjects short-circuit the tag-bin load, matching the
            # empty tags_features handed to the reference.
            subjects=[''],
            density=density,
            present=True,
        ),
        facts=GalleryFacts(is_wedding=True, is_artificial_time=artificial, model_version=2),
    )

    with as_the_monolith():
        context = build_select(logger=logger).run(context)
    assert not context.failed, context.error

    outcome = context.selection
    return (
        outcome.photo_ids,
        outcome.spreads,
        outcome.min_total_spreads,
        outcome.max_total_spreads,
        None,
    )


SCENARIOS = {
    "no hints (unscored path)": dict(
        ten_photos=[], person_ids=[], focus=['brideAndGroom'], density=3,
        artificial=False, rating=None,
    ),
    "people + picked photos": dict(
        ten_photos=None, person_ids=[BRIDE_ID, GROOM_ID, 305], focus=['brideAndGroom'],
        density=3, artificial=False, rating=None,
    ),
    "with ratings": dict(
        ten_photos=None, person_ids=[BRIDE_ID], focus=['parents'], density=4,
        artificial=False, rating=None,
    ),
    "artificial time": dict(
        ten_photos=None, person_ids=[GROOM_ID], focus=['everyoneElse'], density=2,
        artificial=True, rating=None,
    ),
    "low density, everyone else": dict(
        ten_photos=[], person_ids=[], focus=['everyoneElse'], density=1,
        artificial=False, rating=None,
    ),
}


def _resolve(scenario, df):
    """Fill in the scenario's data-dependent fields."""
    resolved = dict(scenario)
    if resolved["ten_photos"] is None:
        resolved["ten_photos"] = df["image_id"].iloc[::37].tolist()[:10]
    return resolved


def check_scenario(name: str, scenario: dict, df: pd.DataFrame) -> None:
    resolved = _resolve(scenario, df)
    if name == "with ratings":
        resolved["rating"] = [
            {"photoId": int(pid), "rating": int(r)}
            for pid, r in zip(df["image_id"].iloc[::5], df["user_rating"].iloc[::5])
        ]

    ref_ids, ref_spreads, ref_min, ref_max, ref_err = run_reference(df, **resolved)
    new_ids, new_spreads, new_min, new_max, new_err = run_pipeline(df, **resolved)

    assert ref_err is None, f"{name}: reference errored: {ref_err}"
    assert new_err is None, f"{name}: pipeline errored: {new_err}"
    assert ref_ids, f"{name}: reference selected nothing — the fixture is not exercising the code"

    assert new_ids == ref_ids, (
        f"{name}: selected photos differ\n"
        f"  only in reference: {sorted(set(ref_ids) - set(new_ids))}\n"
        f"  only in pipeline:  {sorted(set(new_ids) - set(ref_ids))}\n"
        f"  order differs:     {new_ids != ref_ids and set(new_ids) == set(ref_ids)}"
    )
    assert new_spreads == ref_spreads, f"{name}: spread budget differs"
    assert (new_min, new_max) == (ref_min, ref_max), f"{name}: spread envelope differs"


def test_selection_matches_reference():
    df = make_gallery()
    for name, scenario in SCENARIOS.items():
        check_scenario(name, scenario, df)


def test_selection_matches_reference_other_seeds():
    for seed in (11, 23, 42):
        df = make_gallery(seed=seed)
        check_scenario("people + picked photos", SCENARIOS["people + picked photos"], df)


def test_pipeline_is_composable():
    """The point of the decomposition: a substage can be swapped out."""
    from src.pipeline.select.strategies import default_registry
    from src.pipeline.select.strategies.base import CategoryStrategy
    from src.pipeline.select.contracts import CategoryPicks

    class SkipCategory(CategoryStrategy):
        handles = ("dancing",)

        def pick(self, request):
            # `skip` abandons the category outright. Returning an empty
            # `preferred` instead would still let the driver top the shortfall
            # up from greyscale, which is the intended behaviour there.
            return CategoryPicks(skip=True)

    df = make_gallery()
    logger = _quiet_logger()

    def run(registry):
        context = AlbumContext(
            logger=logger, request={}, photos=df.copy(),
            hints=AiHints(subjects=[''], density=3, present=True),
            facts=GalleryFacts(is_wedding=True, model_version=2),
        )
        pipeline = build_select(
            logger=logger, options={"select.pick": {"strategies": registry}}
        )
        return pipeline.run(context).selection.photo_ids

    baseline = run(default_registry())
    swapped = run(default_registry().set("dancing", SkipCategory()))

    dancing_ids = set(df[df["cluster_context"] == "dancing"]["image_id"])
    assert dancing_ids & set(baseline), "fixture should select some dancing photos"
    assert not (dancing_ids & set(swapped)), "replacement strategy was not used"
    assert set(baseline) - dancing_ids == set(swapped) - dancing_ids, (
        "swapping one category changed another"
    )


if __name__ == "__main__":
    test_selection_matches_reference()
    print("equivalence across scenarios: ok")
    test_selection_matches_reference_other_seeds()
    print("equivalence across seeds: ok")
    test_pipeline_is_composable()
    print("substage replacement: ok")
