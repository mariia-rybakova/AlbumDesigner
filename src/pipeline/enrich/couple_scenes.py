"""Whole scenes filed as the couple that are not of the couple.

`enrich.identities` already re-files the single frame holding one of them and a
recognised third person. It cannot see the case this substage is for, because
there the third person is never recognised at all: on 49996919 the album opened
on the bride being kissed by her **father**, in a first look shot against the
same wall as the couple portraits, and across the whole scene the identity model
names the bride four times and the father not once.

Frame by frame every one of those photos reads as "the couple with one face
missed", which is ordinary, common, and must stay cheap -- 29 of 120 couple
frames on 49995684 are exactly that, and demoting them would cost a quarter of
the class to a detection gap. Over the scene the difference is plain: a real
couple scene names the groom in *some* of its frames.

Why a substage of its own rather than another test inside `enrich.identities`:
the question needs a timeline, and `enrich.temporal` -- which rebuilds one when
the EXIF is unusable -- runs after identities. Declaring `general_time` in
`requires` puts it where it can actually be asked.
"""

from __future__ import annotations

from src.pipeline import subject
from src.pipeline.contracts import AlbumContext, Col, photo
from src.pipeline.registry import register
from src.pipeline.substage import SubStage


@register
class CoupleScenesSubStage(SubStage):
    """Re-file couple-class scenes that hold one of them and never the other.

    Optional and wedding-only. The rule reads `cluster_context`, which
    non-wedding galleries never get, and a gallery whose identities never
    resolved should lose the correction rather than the album --
    `lopsided_couple_scenes` returns ``None`` in every such case and nothing
    moves.
    """

    name = "enrich.couple_scenes"
    requires = frozenset({
        photo(Col.PERSONS_IDS),
        photo(Col.CLUSTER_CONTEXT),
        # The axis the scenes are cut along. `image_order` is a quality rank
        # and `image_time` may be unusable; only this one is a day.
        photo(Col.GENERAL_TIME),
    })
    optional = True

    def applies_to(self, context: AlbumContext) -> bool:
        return bool(context.facts.is_wedding) and subject.enabled()

    def execute(self, context: AlbumContext) -> AlbumContext:
        context.photos, _moved = subject.refile_lopsided_couple_scenes(
            context.photos,
            context.facts.bride_id,
            context.facts.groom_id,
            context.logger,
        )
        return context
