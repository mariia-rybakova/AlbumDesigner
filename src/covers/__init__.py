"""Choosing and building the album's first and last pages.

The cover is not picked by the selection that fills the album; it has its own
model, and this is it, in three layers that depend in one direction only:

    pages     -> choosing -> scoring

* `scoring`  how good a photograph is as a cover, as a number
* `choosing` which two photographs it will be
* `pages`    putting them onto a page, with a design and the fold

This is a shelf, not a tree: five callers reach five different functions, and
that is fine here -- unlike `spreads_layout` and `groups_operations`, there is
no single entrance to protect. Nothing is re-exported, so import the layer you
mean and the dependency stays visible at the call site.
"""
