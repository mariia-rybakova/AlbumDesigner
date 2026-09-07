"""Data-quality gates between enrichment steps."""

from __future__ import annotations

from src.pipeline.contracts import AlbumContext, Col, photo
from src.pipeline.registry import register
from src.pipeline.enrich.dedupe import shot_groups
from src.pipeline.substage import SubStage
from utils.read_protos_files import require_cluster_data


@register
class DuplicateShotsSubStage(SubStage):
    """Keep one copy of each shot the photographer uploaded more than once.

    A photographer often uploads a second copy of their best frames with a
    different treatment -- black and white, or a blue or brown tone. Gallery
    53273032 is 1028 photos that are really **514 shots, each uploaded twice**,
    and one album spread ended up showing the same dance frame in colour and in
    black and white.

    This has to run before anything counts the gallery, which is why it is here
    and not in the picker. Left until selection, the budget sizes the album
    against a supply twice as large as it really is, then every category runs
    out of distinct photos and the album comes up short -- measured, 100 photos
    down to 86. Removed up front, the counts are simply right: `select.budget`
    allocates against 514, the layout cannot pad a group with a twin either,
    and no later stage needs to know.

    See :mod:`src.pipeline.enrich.dedupe` for why the test is the capture
    second rather than the pixels, and for the guard that stops it firing on a
    gallery whose timestamps are broken.

    The copy kept is the best-ranked one -- `image_order` is `selectionOrder`,
    where 0 is best -- so the choice is the content model's, not the upload
    order's, and it is independent of which treatment came first.
    """

    name = "enrich.duplicate_shots"
    requires = frozenset({photo(Col.IMAGE_TIME), photo(Col.IMAGE_AS),
                          photo(Col.IMAGE_ORDER)})

    def execute(self, context: AlbumContext) -> AlbumContext:
        photos = context.photos
        groups = shot_groups(photos, context.logger)
        if not groups:
            return context

        keep = set()
        dropped = 0
        for members in groups.values():
            ranked = photos.loc[photos[Col.IMAGE_ID].isin(members)]
            ranked = ranked.sort_values(Col.IMAGE_ORDER, ascending=True)
            keep.add(ranked[Col.IMAGE_ID].iloc[0])
            dropped += len(members) - 1

        if not dropped:
            return context

        spoken_for = {i for members in groups.values() for i in members}
        context.photos = photos[
            ~photos[Col.IMAGE_ID].isin(spoken_for - keep)
        ]
        if context.logger:
            context.logger.info(
                f"Duplicate shots: {len(photos)} photos are {len(photos) - dropped} "
                f"distinct shots; dropped {dropped} second copies"
            )
        return context


@register
class RequireClusterDataSubStage(SubStage):
    """Drop photos missing the cluster columns everything downstream assumes.

    Runs after the semantic tagging, matching the original order: tagging is
    what drops rows with unusable embeddings, and this drops rows the
    content-clustering model had nothing to say about.
    """

    name = "enrich.require_cluster_data"
    requires = frozenset({photo(Col.RANKING), photo(Col.CLUSTER_LABEL)})

    def execute(self, context: AlbumContext) -> AlbumContext:
        context.photos = require_cluster_data(context.photos, context.logger)
        return context
