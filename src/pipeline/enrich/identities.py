"""Who is in the photos: the couple, the people-composition key, the parents."""

from __future__ import annotations

from src.pipeline.contracts import AlbumContext, Col, photo
from src.pipeline.registry import register
from src.pipeline.substage import SubStage
from src.request_processing import identify_parents
from utils.read_protos_files import add_people_cluster, resolve_bride_groom


@register
class IdentitiesSubStage(SubStage):
    """Resolve bride and groom identity ids and stamp them on every row.

    Wedding-only: it reads the ``bride`` / ``groom`` content categories, which
    ``enrich.content_class`` only produces for weddings.
    """

    name = "enrich.identities"
    requires = frozenset({photo(Col.CLUSTER_CONTEXT), photo(Col.MAIN_PERSONS)})
    provides = frozenset({photo(Col.BRIDE_ID), photo(Col.GROOM_ID)})

    def applies_to(self, context: AlbumContext) -> bool:
        return bool(context.facts.is_wedding)

    def execute(self, context: AlbumContext) -> AlbumContext:
        context.photos = resolve_bride_groom(context.photos, context.logger)

        if not context.photos.empty:
            context.facts.bride_id = context.photos.iloc[0][Col.BRIDE_ID]
            context.facts.groom_id = context.photos.iloc[0][Col.GROOM_ID]
        return context


@register
class PeopleClusterSubStage(SubStage):
    """Key each photo by its people composition (``2_pple_17_44``, ...)."""

    name = "enrich.people_cluster"
    requires = frozenset({photo(Col.PERSONS_IDS), photo(Col.NUMBER_BODIES)})
    provides = frozenset({photo(Col.PEOPLE_CLUSTER)})

    def execute(self, context: AlbumContext) -> AlbumContext:
        context.photos = add_people_cluster(context.photos)
        return context


@register
class ParentsSubStage(SubStage):
    """Re-label portraits that show the couple with a set of parents.

    Cross-references the social circles with per-identity age and gender, then
    rewrites ``cluster_context`` to ``parents portrait`` and records which of
    the three parent groupings it is in ``parent_category``.
    """

    name = "enrich.parents"

    def execute(self, context: AlbumContext) -> AlbumContext:
        context.photos = identify_parents(
            context.social_circles,
            context.person_details,
            context.photos,
            logger=context.logger,
        )
        return context
