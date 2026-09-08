"""How many albums this gallery is worth, and what each one should be.

Phase 3 of `docs/multi_album_plan.md`. The count cannot come off the request,
because whether a second album is worth making is a fact about the *gallery* --
which identities were found, which content classes exist, whether the
embeddings are the width a policy needs. So it is derived here, at the end of
ENRICH, like every other derived fact: with declared requirements, checked at
the boundary, and testable on its own.

``len(context.variants)`` is N. One variant is the normal case and overrides
nothing, which is what makes a single album byte-identical to having no
variants at all.

The first axis is ``focus``, because it is the cheapest real one. A focus names
a column of `files/focus_csv.csv` -- the per-category spread profile the budget
is built from -- so two focus values compose genuinely different albums out of
one gallery read, with no extra reading and no new selection code.

``autoAlbums`` on the request turns it on, and it is false unless asked for:
two albums where one was expected is a product decision, not a default.
"""

from __future__ import annotations

from typing import List

from src.pipeline.albums import AlbumVariant
from src.pipeline.contracts import AlbumContext, Col, ctx, photo
from src.pipeline.registry import register
from src.pipeline.substage import SubStage

#: The request flag. Absent or false means one album, as today.
AUTO_ALBUMS = "autoAlbums"

#: What ``autoAlbums`` asks for: the couple, and the family. Both are columns
#: of the focus CSV, so each gets its own per-category budget out of the same
#: gallery. `everyoneElse` is the third column and is deliberately not here --
#: it is close to the default profile, so it would produce the least distinct
#: third album.
AUTO_FOCUS = ("brideAndGroom", "parents")

#: The single album a normal request gets: no overrides, so selection sees
#: exactly the request's own hints.
AS_REQUESTED = AlbumVariant(name="requested")


@register
class VariantsSubStage(SubStage):
    """Plan the albums for this gallery.

    Always runs and always produces at least one variant, so nothing
    downstream has to care whether planning happened.
    """

    name = "enrich.variants"
    requires = frozenset({photo(Col.IMAGE_ID)})
    provides = frozenset({ctx("variants")})

    def execute(self, context: AlbumContext) -> AlbumContext:
        context.variants = self._plan(context)
        if context.logger and len(context.variants) > 1:
            names = ", ".join(v.name for v in context.variants)
            context.logger.info(
                f"Variants: {len(context.variants)} albums planned ({names})")
        return context

    @staticmethod
    def _plan(context: AlbumContext) -> List[AlbumVariant]:
        request = context.request or {}
        if not request.get(AUTO_ALBUMS):
            return [AS_REQUESTED]

        # One album per focus. Nothing gallery-dependent yet -- the facts to
        # gate on (which identities `enrich.parents` actually resolved, whether
        # the couple classes exist) are available here and are the obvious next
        # refinement, but a flag that silently produced one album would be
        # worse than one that produces two the gallery cannot fill well.
        return [AlbumVariant(name=focus, focus=(focus,)) for focus in AUTO_FOCUS]
