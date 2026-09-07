"""Tests for the picker's self-attribution -- Phase 0 of the CP-SAT plan.

`WeddingPicker` settles a category at one of a dozen decision points, and from
the outside they are indistinguishable: a class comes back short because the
gate declined it, because temporal narrowing emptied it, because
`_take_all_distinct` deduplicated it, or because a strategy saturated.
`per_category['bound_by']` names which, and every later phase of
`docs/cpsat_scoring_plan.md` is scored against that.

The risk this guards is drift: a new early return in `_run_category` that
nobody names, or a mechanism renamed out from under
`tools/pick_attribution.py`.

    python -m pytest tests/test_pick_attribution.py -v
"""

from __future__ import annotations

import ast
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import AlbumContext, build_select  # noqa: E402
from src.pipeline.contracts import AiHints, GalleryFacts  # noqa: E402
from tools.pick_attribution import MECHANISMS  # noqa: E402
from utils.configs import CONFIGS  # noqa: E402

PICKER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'src', 'pipeline', 'select', 'pick.py')


def quiet():
    logger = logging.getLogger("attribution-test")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    return logger


def select(df, *, cpsat=False):
    logger = quiet()
    context = AlbumContext(
        logger=logger, request={'rating': []}, photos=df.copy(),
        available_photo_ids=[],
        hints=AiHints(photo_ids=[], person_ids=[], focus=['brideAndGroom'],
                      subjects=[''], density=3, present=True),
        facts=GalleryFacts(is_wedding=True, is_artificial_time=False,
                           model_version=2),
    )
    original = CONFIGS['pick_cpsat']
    CONFIGS['pick_cpsat'] = {**original, 'enabled': cpsat}
    try:
        context = build_select(logger=logger).run(context)
    finally:
        CONFIGS['pick_cpsat'] = original
    assert not context.failed, context.error
    return context.selection.per_category


@pytest.fixture(scope="module")
def gallery():
    from test_selection_equivalence import make_gallery
    return make_gallery()


# -- the record is complete ------------------------------------------------


def test_every_category_names_what_settled_it(gallery):
    per_category = select(gallery)

    assert per_category, "the fixture should exercise several categories"
    unnamed = [c for c, entry in per_category.items() if not entry.get('bound_by')]
    assert not unnamed, f"no mechanism recorded for {unnamed}"


def test_only_known_mechanisms_are_recorded(gallery):
    """`tools/pick_attribution.py` tabulates by mechanism, so a name it does
    not know about silently vanishes from the baseline."""
    per_category = select(gallery)

    recorded = {entry['bound_by'] for entry in per_category.values()}
    assert recorded <= set(MECHANISMS), (
        f"unknown mechanisms {sorted(recorded - set(MECHANISMS))}; "
        f"add them to tools.pick_attribution.MECHANISMS")


def test_the_tool_knows_every_mechanism_the_picker_emits():
    """Read the literals out of `pick.py` rather than waiting for a fixture to
    exercise the branch -- a rarely-taken exit would otherwise go unnoticed."""
    tree = ast.parse(open(PICKER, encoding='utf-8').read())
    emitted = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        name = getattr(target, 'attr', None) or getattr(target, 'id', None)
        if name not in ('_note', 'setdefault'):
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                emitted.add(argument.value)

    emitted -= {'bound_by'}          # the key itself, not a mechanism
    missing = emitted - set(MECHANISMS)
    assert not missing, (
        f"pick.py records {sorted(missing)}, which tools.pick_attribution "
        f"does not list")


def test_every_exit_from_run_category_is_named():
    """A bare `return` in `_run_category` with no `_note` before it leaves the
    category unattributed, which is exactly the blind spot Phase 0 removes."""
    tree = ast.parse(open(PICKER, encoding='utf-8').read())
    function = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == '_run_category')

    # Statement lists, walked so a `return` inside an `if` is checked against
    # the statement immediately before it in the same block.
    def check(body):
        for index, statement in enumerate(body):
            if isinstance(statement, ast.Return):
                before = body[:index]
                named = any(
                    isinstance(earlier, ast.Expr)
                    and isinstance(earlier.value, ast.Call)
                    and getattr(earlier.value.func, 'attr', '') in ('_note', 'setdefault')
                    for earlier in before)
                assert named, (
                    f"the return at line {statement.lineno} of _run_category "
                    f"is not preceded by a _note in its own block")
            for field in ('body', 'orelse', 'finalbody'):
                nested = getattr(statement, field, None)
                if nested:
                    check(nested)

    check(function.body)


# -- the counts mean what the tool assumes ---------------------------------


def test_committed_is_tracked_apart_from_selected(gallery):
    """`selected` counts committed photos too, and their allowance was charged
    in `select.preselect`. The tool subtracts one from the other to get what
    the picker itself chose, so the two must not be conflated."""
    per_category = select(gallery)

    for category, entry in per_category.items():
        committed = entry.get('committed', 0)
        assert committed <= entry.get('selected', 0), (
            f"{category}: {committed} committed but only "
            f"{entry.get('selected', 0)} selected")


def test_the_allowance_is_recorded_wherever_one_was_read(gallery):
    per_category = select(gallery)

    for category, entry in per_category.items():
        if entry['bound_by'] == 'all_committed':
            continue  # settled before the allowance is looked at
        assert 'need' in entry, f"{category} has no recorded allowance"


def test_cpsat_attributes_its_categories_too(gallery):
    """The two pickers are compared class by class, so both have to answer."""
    per_category = select(gallery, cpsat=True)

    recorded = {entry.get('bound_by') for entry in per_category.values()}
    assert recorded <= set(MECHANISMS)


# -- the baseline ----------------------------------------------------------


def test_the_committed_baseline_is_readable():
    """Later phases diff against this file, so a malformed one is a silent
    loss of the only reference point."""
    import json
    path = os.path.join('tools', 'baselines', 'pick_attribution.json')
    if not os.path.exists(path):
        pytest.skip("no baseline recorded yet")

    payload = json.load(open(path, encoding='utf-8'))
    assert payload['galleries'], "a baseline with no galleries says nothing"
    for gallery, result in payload['galleries'].items():
        assert result['categories'], f"{gallery} has no categories"
        for category, row in result['categories'].items():
            assert row['bound_by'] in MECHANISMS or row['bound_by'] is None, (
                f"{gallery}/{category}: unknown mechanism {row['bound_by']!r}")
            assert row['loop_chose'] == row['loop_selected'] - row['committed']


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
