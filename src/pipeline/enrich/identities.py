"""Who is in the photos: the couple, the people-composition key, the parents."""

from __future__ import annotations

import pandas as pd

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


#: The content model's own classes for a same-sex couple. Looked up by name so
#: a reordering of `label_list` cannot silently point this at the wrong class.
SAME_SEX_LABELS = ('two brides', 'two grooms')

#: The two solo-portrait classes. On a same-sex gallery the model puts *both*
#: partners' solo portraits into one of them and leaves the other empty.
SOLO_CLASSES = ('bride', 'groom')


def _same_sex_classes():
    from utils.reading_tools import label_list
    return {i for i, label in enumerate(label_list) if label in SAME_SEX_LABELS}


@register
class SameSexCoupleSubStage(SubStage):
    """Give each partner of a same-sex couple their own solo-portrait class.

    `map_cluster_label` folds the model's `two brides` / `two grooms` into
    `bride and groom`, which is right for the couple shots. The damage is to the
    *solo* shots. The model puts both partners' portraits into a single class --
    on gallery 52894932, `bride` held 68 photos that split exactly 32/32 between
    the two brides -- and `groom` was empty.

    That loses one partner entirely, because `CoupleTimelineStrategy` filters
    the `bride` category to ``persons_ids == [bride_id]`` and the `groom`
    category to ``persons_ids == [groom_id]``. The 32 portraits of the second
    bride sat in the `bride` class where that filter rejects them, while the
    `groom` category that would have taken them had no photos at all. Her 12% of
    the album bought nothing.

    So the second partner's solo shots are moved into the empty counterpart
    class. Everything downstream then works untouched and already correctly:
    two 12% shares in `focus_csv.csv`, the identity filter in the strategy, the
    lookup table, the per-class thresholds. Nothing needs to know the couple is
    same-sex -- which is the point of doing it here.

    Only exact solo frames move (``persons_ids == [other]``), which is precisely
    what the receiving category accepts, so nothing is moved into a class that
    would reject it.
    """

    name = "enrich.same_sex_couple"
    requires = frozenset({photo(Col.CLUSTER_CLASS), photo(Col.CLUSTER_CONTEXT),
                          photo(Col.BRIDE_ID), photo(Col.GROOM_ID),
                          photo(Col.PERSONS_IDS)})

    def applies_to(self, context: AlbumContext) -> bool:
        return bool(context.facts.is_wedding)

    def execute(self, context: AlbumContext) -> AlbumContext:
        photos = context.photos
        classes = _same_sex_classes()
        if not classes or not photos[Col.CLUSTER_CLASS].isin(classes).any():
            return context

        context.facts.same_sex_couple = True
        counted = int(photos[Col.CLUSTER_CLASS].isin(classes).sum())

        identities = {'bride': context.facts.bride_id, 'groom': context.facts.groom_id}
        moved = 0
        for holder, target in (('bride', 'groom'), ('groom', 'bride')):
            other = identities[target]
            if other is None or pd.isna(other):
                continue
            # Only when the counterpart class really is empty -- that emptiness
            # is the whole signature, and a populated one needs no help.
            if (photos[Col.CLUSTER_CONTEXT] == target).any():
                continue

            held = photos[photos[Col.CLUSTER_CONTEXT] == holder]
            solo = held[held[Col.PERSONS_IDS].apply(
                lambda ids: isinstance(ids, list) and ids == [other])]
            if solo.empty:
                continue

            photos.loc[solo.index, Col.CLUSTER_CONTEXT] = target
            moved += len(solo)
            if context.logger:
                context.logger.info(
                    f"Same-sex couple: moved {len(solo)} solo photos of identity "
                    f"{other} from '{holder}' to the empty '{target}' class, so both "
                    f"partners have their own share of the album")

        if context.logger and not moved:
            context.logger.info(
                f"Same-sex couple ({counted} photos), but no solo frames needed "
                f"moving: both solo classes are already populated")
        return context
