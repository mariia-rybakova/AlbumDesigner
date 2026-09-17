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

The second album is **seeded** when `enrich.parents` resolved anybody. Focus on
its own changes the album's shape and not its content: a `parents` profile asks
for more family spreads and the picker fills them with whatever ranked best,
which on a gallery the couple dominates is the couple. So where there are
parents to build an album around, the variant also carries a pseudo
``aiMetadata`` -- their photos and their people -- which `select.preselect`
commits before ranking and `person_score` weighs everywhere else. See
`src.pipeline.family`.

Where no parents were resolved there is nothing to seed from, and the plan is
exactly what it was: two focus-only variants. That is the normal outcome on a
gallery whose candidates cannot be separated, not an error, so it must stay a
working two-album plan rather than a failure.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List

from src.pipeline import album_requests, family
from src.pipeline.albums import AlbumVariant
from src.pipeline.contracts import AlbumContext, Col, ctx, photo
from src.pipeline.registry import register
from src.pipeline.substage import SubStage

#: The request flag. Absent or false means one album, as today.
AUTO_ALBUMS = "autoAlbums"

#: The focus whose album gets the parents seed.
PARENTS_FOCUS = "parents"

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
    #: Deliberately still just the photo table's identity column, even though
    #: the parents seed reads `persons_ids` and `cluster_context` too.
    #: Declaring those would make this substage *fail* on a gallery that lacks
    #: them -- and this one is not allowed to fail: planning is what guarantees
    #: there is at least one album, so a gallery with no identity data must
    #: still come out of here with a plan. The seed checks for its own columns
    #: and stands down when they are missing, which is the correct degradation:
    #: no seed, still an album.
    requires = frozenset({photo(Col.IMAGE_ID)})
    provides = frozenset({ctx("variants")})

    def execute(self, context: AlbumContext) -> AlbumContext:
        context.variants = self._plan(context)
        if context.unfulfilled and isinstance(context.request, dict):
            context.request[album_requests.UNFULFILLED_KEY] = [
                decline.as_content() for decline in context.unfulfilled]
        if context.logger and len(context.variants) > 1:
            names = ", ".join(v.name for v in context.variants)
            context.logger.info(
                f"Variants: {len(context.variants)} albums planned ({names})")
        return context

    @staticmethod
    def _plan(context: AlbumContext) -> List[AlbumVariant]:
        request = context.request or {}

        briefs, declines = album_requests.parse(request)
        if briefs:
            return VariantsSubStage._from_briefs(briefs, declines, context)

        context.unfulfilled = tuple(declines)
        if not request.get(AUTO_ALBUMS):
            return [AS_REQUESTED]

        plan = [AlbumVariant(name=focus, focus=(focus,)) for focus in AUTO_FOCUS]
        return [VariantsSubStage._seed(variant, context) for variant in plan]

    @staticmethod
    def _from_briefs(briefs, declines, context: AlbumContext) -> List[AlbumVariant]:
        """Answer every brief the caller sent: compose it, or decline it.

        The list guides; the gallery decides. A brief whose requirement this
        gallery cannot meet is declined by name and reason rather than composed
        into something that is not what was asked for -- a parents album on a
        gallery with no resolved parents is a second couple album wearing the
        wrong label, and the caller cannot tell.

        Every brief leaves with an answer, and a gallery that can meet nothing
        still returns at least one album: declining the lot would turn a
        product decision into an empty reply.
        """
        plan: List[AlbumVariant] = []
        refused = list(declines)

        for brief in briefs:
            reason = VariantsSubStage._refuse(brief, context)
            if reason is not None:
                refused.append(album_requests.Decline(
                    brief.album_request_id, reason[0], reason[1]))
                continue
            variant = AlbumVariant(
                name=brief.logical_name or brief.album_request_id,
                focus=brief.focus,
                photo_ids=brief.photo_ids,
                person_ids=brief.person_ids,
                fulfils=brief.album_request_id,
            )
            plan.append(VariantsSubStage._seed(variant, context))

        if not plan:
            # Nothing survived. The gallery still gets the album the request
            # itself asked for, and the reply says what was refused and why.
            plan = [AS_REQUESTED]

        context.unfulfilled = tuple(refused)
        if context.logger and refused:
            summary = ", ".join(f"{d.album_request_id}: {d.reason}" for d in refused)
            context.logger.info(f"Variants: {len(refused)} brief(s) declined ({summary})")
        return plan

    @staticmethod
    def _refuse(brief, context: AlbumContext):
        """``(reason, detail)`` when this gallery cannot honour a brief.

        Keyed on the *objective* -- the focus the composer already translated
        its product name into -- and not on the product name itself, so a new
        product that maps onto an existing focus needs no change here.
        """
        for requirement in REQUIREMENTS:
            answer = requirement(brief, context)
            if answer is not None:
                return answer
        return None


    @staticmethod
    def _seed(variant: AlbumVariant, context: AlbumContext) -> AlbumVariant:
        """Give the parents album a pseudo selection, where there is one.

        Every other variant is returned untouched, so this cannot reach the
        album the request actually asked for.
        """
        if PARENTS_FOCUS not in (variant.focus or ()) or not family.enabled():
            return variant

        facts = context.facts
        photo_ids, person_ids = family.parents_seed(
            context.photos,
            getattr(facts, "bride_parents", ()),
            getattr(facts, "groom_parents", ()),
            getattr(facts, "bride_id", None),
            getattr(facts, "groom_id", None),
        )
        if not photo_ids:
            # Parents resolved but nothing to show them in, or none resolved at
            # all. A focus-only variant is still a valid album.
            return variant

        if not family.settings().get("replace_user_picks", True):
            existing = tuple(context.hints.photo_ids or ())
            photo_ids = tuple(dict.fromkeys(existing + photo_ids))

        if context.logger:
            context.logger.info(
                f"Variants: seeded the '{variant.name}' album with "
                f"{len(photo_ids)} parent photo(s) and {len(person_ids)} "
                f"identity/identities")
        return replace(variant, photo_ids=photo_ids, person_ids=person_ids)


def _resolved_parents(context: AlbumContext) -> bool:
    """Did `enrich.parents` separate anybody? The same facts `_seed` reads."""
    facts = context.facts
    return bool(getattr(facts, "bride_parents", ()) or getattr(facts, "groom_parents", ()))


def _needs_parents(brief, context: AlbumContext):
    """A parents album with no parents is not a parents album.

    Composing it anyway produces a second couple album wearing the wrong
    label -- the focus alone changes the album's shape and not its content --
    and the caller has no way to tell. Declining says so.
    """
    if PARENTS_FOCUS not in (brief.focus or ()):
        return None
    if _resolved_parents(context):
        return None
    return ("no_parents_resolved",
            "enrich.parents resolved nobody, so there is no family to build around")


#: The known set a brief can be refused by. Config may select from these; it
#: cannot define one, because a requirement is a question about the gallery
#: rather than a value -- which is the whole reason this stays in code.
REQUIREMENTS = (_needs_parents,)
