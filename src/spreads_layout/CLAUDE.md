# src/spreads_layout — this package is a tree. Read this before adding a file.

`main.py` is the only way in. Everything else here is reached from above it,
never from outside the package:

```
main.process_group
  └── partitions      how many photos each spread of a sub-group gets
        └── combinations   which photos go together on one spread
              └── spreads/ the spread itself, scored against the layouts
  with math_tools, layouts_tools, group_layouts as leaves
```

`main.select_best_layout_for_subgroup` is the second, narrower entrance, used
where a caller already has one sub-group in hand.

## The rule

**If new code here would be imported from outside this package, it does not
belong here.** Put it in `src/core/`, which exists to be imported from
anywhere, and call it from the tree.

Re-exporting an internal through an `__init__.py` is not a fix. The dependency
is unchanged; only the import path looks tidier, and the next person reads a
tree that is no longer one.

Two modules left for exactly this reason and are the worked examples:
`src/core/design_templates.py` (the design catalogue, read by
`request_processing` before any layout happens) and
`src/core/spread_ordering.py` (chronological order, applied after
`process_group` has returned).

## Known exceptions

These predate the rule. None of them is a precedent, and the rule is not
already true — it is being worked towards:

| Module | Imported from outside by |
|---|---|
| `fold` | `album_processing`, `core/key_pages`, `stages_recorder/fold` |
| `twins.apply_twin_colors` | `album_processing` |
| `spreads/spread.set_neutral_contexts` | `album_processing` |
| `combinations.Combination`, `group_layouts.get_group_single_layouts` | `predefined/processing` |

Tests are not exceptions — a test may import any module it needs.
