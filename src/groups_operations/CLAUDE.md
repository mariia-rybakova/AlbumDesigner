# src/groups_operations — this package is a tree. Read this before adding a file.

`groups_management.process_wedding_illegal_groups` is the only way in.
Everything else here is reached from above it, never from outside the package:

```
groups_management.process_wedding_illegal_groups
  ├── ceremony  the processional, the vows and the send-off become one group
  ├── split     oversized or temporally diverse groups are cut
  └── merge     small and complementary groups are joined
```

`ceremony` runs first and for a different reason than the other two. They
reshape groups that are *illegal* -- too big, too diverse, too small. It
reshapes legal ones, because the ceremony is one narrative told by four classes
and a group is an atomic block downstream. Running it before `split` is what
puts the merged sequence under the same two-spread ceiling as everything else.

`docs/ceremony_sequence.md` is the design note: what was tried before it, which
guards are load-bearing, and why the window is read from the gallery rather than
from the photos selection kept. Read it before changing any of that.

This package *reshapes* groups that already exist. Making them in the first
place is `src/core/group_formation.py` — a different job, deliberately
outside this tree, because `album_processing` calls it directly: it sizes the
lookup table from the raw groups before any splitting happens, and for a
non-wedding album it is the only grouping there is.

## The rule

**If new code here would be imported from outside this package, it does not
belong here.** Put it in `src/core/`, which exists to be imported from
anywhere, and call it from the tree.

Re-exporting an internal through an `__init__.py` is not a fix. The dependency
is unchanged; only the import path looks tidier, and the next person reads a
tree that is no longer one.

A helper used by exactly one function in this package belongs beside that
function, not in a module of its own: `split_groups` and
`assign_special_group_contexts` sit next to `_get_groups`, their only caller.

## Known exceptions

None in production code. `album_processing` enters at
`process_wedding_illegal_groups` and nowhere else — keep it that way.

Tests are not exceptions — a test may import any module it needs.
