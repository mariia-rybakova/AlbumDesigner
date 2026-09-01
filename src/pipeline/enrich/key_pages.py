"""The photos that open and close the album.

Choosing a cover is an inference about photos -- which frame of the couple best
opens a story, which one best ends it -- not a step of page layout. It has
nevertheless always run inside ProcessStage, in
``src/core/key_pages.py::generate_first_last_pages``, wedged between time
clustering and the layout search. This substage is the first half of moving it
where it belongs.

Weddings only, for now, like the other substages built around the couple --
``enrich.content_class`` and ``enrich.identities``. ProcessStage keeps handling
non-wedding galleries.

**This phase changes nothing about the album.** The substage calls the same
function ProcessStage calls, so there is one implementation of the rule and no
chance of the two drifting; the only difference is the pool it is handed. Enrich
runs before selection, so that pool is the whole gallery instead of the few
hundred frames selection kept. ProcessStage still runs its own copy and still
decides the covers -- nothing downstream reads what this writes yet.

Two properties of the wider pool matter for the second half of the move:

``time_cluster`` does not exist yet
    It is built in ProcessStage. Without it the shared ``_pick_cover_subset``
    helper falls through to its ``image_time`` branch and takes the ten earliest
    and ten latest couple photos rather than the whole first and last time
    cluster. On a gallery whose EXIF is unusable -- the reason ``general_time``
    exists at all -- those ten are not the start of the day.

The pool is not quality-filtered
    Selection is what removes the weak frames. Over the full gallery a mediocre
    early couple shot competes on equal terms with the good one, separated only
    by ``image_order`` and only after the subquery priority has already tied.

Both argue for the consumer resolving the ranked list against what it actually
has, rather than this substage trying to guess the survivors. Hence ``KeyPages``
holding lists.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import pandas as pd

from src.core.key_pages import choose_good_wedding_images
from src.pipeline.contracts import AlbumContext, Col, KeyPages, ctx, photo
from src.pipeline.registry import register
from src.pipeline.substage import SubStage

#: Written into :data:`Col.KEY_PAGE` for the photo that opens the album...
OPENING = "opening"
#: ...and the one that closes it. Every other row keeps the empty string, so
#: the column is safe to compare against without a null check.
CLOSING = "closing"

#: The content category the wedding rule draws its covers from.
COUPLE = "bride and groom"

#: `get_important_imgs` logs from inside its own try/except, so a bare `None`
#: logger does not raise -- it quietly costs the album its covers, which is
#: worse. A context without a logger is normal offline, so stand one in.
_QUIET = logging.getLogger(__name__)
_QUIET.addHandler(logging.NullHandler())


@register
class KeyPagesSubStage(SubStage):
    """Pick the album's opening and closing photos from the whole gallery.

    Wedding-only, like `enrich.content_class` and `enrich.identities`: the rule
    is built around the couple, and it reads the `cluster_context` column that
    non-wedding galleries never get. ProcessStage keeps handling those.

    Optional: an album without a cover is a worse album, not a failed one, and
    the rule reaches into enough columns (subquery tags, identities, embeddings)
    that a gallery missing one of them should lose the covers rather than the
    album.
    """

    name = "enrich.key_pages"
    requires = frozenset({
        photo(Col.IMAGE_ID),
        photo(Col.IMAGE_ORDER),
        photo(Col.IMAGE_ORIENTATION),
        photo(Col.PERSONS_IDS),
        photo(Col.N_FACES),
    })
    provides = frozenset({photo(Col.KEY_PAGE), ctx("key_pages")})
    optional = True

    def applies_to(self, context: AlbumContext) -> bool:
        """Weddings with a first page.

        ``generate_first_last_pages`` gates on the same ``firstPage`` flag. An
        empty ``pagesInfo`` means no design data was read at all -- an offline
        run or a test -- and there the covers are worth computing anyway.
        """
        if not context.facts.is_wedding:
            return False
        pages = context.designs.pages
        if not pages:
            return True
        return bool(pages.get("firstPage"))

    def execute(self, context: AlbumContext) -> AlbumContext:
        photos = context.photos
        photos[Col.KEY_PAGE] = ""

        opening, closing = self._choose(context)

        photos.loc[photos[Col.IMAGE_ID].isin(opening), Col.KEY_PAGE] = OPENING
        photos.loc[photos[Col.IMAGE_ID].isin(closing), Col.KEY_PAGE] = CLOSING
        context.key_pages = KeyPages(opening=opening, closing=closing)

        if context.logger:
            context.logger.info(
                f"Key pages over {len(photos)} gallery photos: "
                f"opening={opening or 'none'}, closing={closing or 'none'}"
            )
        return context

    # -- the rule ----------------------------------------------------------

    def _choose(self, context: AlbumContext) -> Tuple[List[int], List[int]]:
        """Delegate to the same function ProcessStage uses.

        Sorted by ``image_order`` descending because that is the frame
        ProcessStage passes, and ``_pick_cover_subset``'s last-resort branch --
        the one that runs when neither ``time_cluster`` nor ``image_time``
        exists -- reads whatever order it is given.

        The couple frame stands in for ``message.content['bride and groom']``,
        which ``select.publish`` caches for ProcessStage out of the *selected*
        photos. Here it is the whole gallery's worth.
        """
        pool = context.photos.sort_values(Col.IMAGE_ORDER, ascending=False)
        couple = pool[pool[Col.CLUSTER_CONTEXT] == COUPLE]

        _, opening, _, closing, _ = choose_good_wedding_images(
            pool, couple, context.logger or _QUIET)

        return _ids(opening), _ids(closing)


def _ids(value) -> List[int]:
    """Normalise the several empties the rule can return to a list.

    ``get_important_imgs`` answers ``None`` when it raises internally, and
    ``_select_cover_image_ids`` answers ``[]``.
    """
    if value is None:
        return []
    if isinstance(value, pd.Series):
        return value.tolist()
    return list(value)
