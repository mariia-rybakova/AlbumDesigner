"""Data-quality gates between enrichment steps."""

from __future__ import annotations

import pandas as pd

from src.pipeline.contracts import AlbumContext, Col, photo
from src.pipeline.registry import register
from src.pipeline.enrich.dedupe import shot_groups, treatment_twins
from src.pipeline.substage import SubStage
from utils.read_protos_files import require_cluster_data


@register
class TreatmentTwinsSubStage(SubStage):
    """Flag the photos whose shot is also in the gallery in the other treatment.

    Before `enrich.duplicate_shots` drops the second copy, so the pairing
    survives it: every photo of a shot uploaded in colour and in black and white
    records the other version's id and colour flag in `treatment_twin` and
    `twin_color`. Nothing about the photo changes -- it is the same shot, the
    same people, the same moment -- so selection, scoring and grouping read it
    as before. What the flag makes possible is a later swap of one version for
    the other where only colour matters: a black and white frame among colour
    ones costs its page 1e-9 in the layout, and on 49994361 that is what still
    splits `ceremony`, `kiss` and `speech` into two spreads each.

    The same detection as `enrich.duplicate_shots`, restricted to the groups
    that mix treatments -- a gallery uploaded twice in the *same* treatment has
    no other version to swap to.
    """

    name = "enrich.treatment_twins"
    requires = frozenset({photo(Col.IMAGE_TIME), photo(Col.IMAGE_AS),
                          photo(Col.IMAGE_ORDER)})

    def execute(self, context: AlbumContext) -> AlbumContext:
        photos = context.photos
        photos[Col.TREATMENT_TWIN] = pd.Series([None] * len(photos), index=photos.index, dtype=object)
        photos[Col.TWIN_COLOR] = pd.Series([None] * len(photos), index=photos.index, dtype=object)
        twins = treatment_twins(photos, shot_groups(photos, None))
        if not twins:
            return context

        ids = photos[Col.IMAGE_ID]
        # Object dtype, or pandas turns an id column with gaps into floats --
        # 5000.0, and NaN rather than None where there is no twin.
        photos[Col.TREATMENT_TWIN] = pd.Series(
            [twins[i][0] if i in twins else None for i in ids], index=photos.index, dtype=object)
        photos[Col.TWIN_COLOR] = pd.Series(
            [twins[i][1] if i in twins else None for i in ids], index=photos.index, dtype=object)
        if context.logger:
            context.logger.info(f"Treatment twins: {len(twins)} photos have a version of their "
                                f"shot in the other treatment")
        return context


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
        dropping = photos[Col.IMAGE_ID].isin(spoken_for - keep)
        # A dropped copy that a kept photo names as its other treatment is kept
        # aside whole, for the swap `treatment_twin` exists for.
        if Col.TREATMENT_TWIN in photos.columns:
            wanted = set(photos.loc[~dropping, Col.TREATMENT_TWIN].dropna())
            aside = photos[dropping & photos[Col.IMAGE_ID].isin(wanted)]
            context.treatment_twins.update(
                {row[Col.IMAGE_ID]: row for _, row in aside.iterrows()})
        context.photos = photos[~dropping]
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

    #: Photos per spread the thinnest album still needs. Below `2 * min_spreads`
    #: there is not a second photo for every spread the design asks for, and
    #: what comes out is not an album of this gallery but of the fragment of it
    #: that happened to be ready.
    PHOTOS_PER_SPREAD_FLOOR = 2

    def execute(self, context: AlbumContext) -> AlbumContext:
        logger = context.logger
        before = len(context.photos)
        context.photos = require_cluster_data(context.photos, logger)
        after = len(context.photos)

        if after < before and logger is not None:
            # Logged at WARNING, not DEBUG: this is the content model not having
            # finished, and everything downstream -- the gallery type, the
            # budget, the groups -- is then decided on a fragment.
            logger.warning(
                f"enrich.require_cluster_data: dropped {before - after} of {before} "
                f"photos with no content data; {after} remain")

        needed = self._minimum_photos(context)
        if needed is not None and after < needed:
            raise ValueError(
                f"Gallery has {after} photos with content data, and the design's "
                f"smallest album needs {needed} "
                f"({self.PHOTOS_PER_SPREAD_FLOOR} per spread over "
                f"{needed // self.PHOTOS_PER_SPREAD_FLOOR} spreads). Too few to "
                f"compose: the gallery is either still being ingested -- "
                f"{before - after} of {before} photos had no content data -- or "
                f"smaller than this design allows")

        return context

    def _minimum_photos(self, context: AlbumContext):
        """`2 * min_spreads` for this design, or None when it does not say.

        Sized through `size_album`, the same function the layout budget uses, so
        the floor moves with the design rather than with a number kept here.
        """
        designs = getattr(context.designs, 'designs', None) or {}
        if 'minPages' not in designs or 'maxPages' not in designs:
            return None
        try:
            from src.album_processing import size_album

            min_spreads, _ = size_album(designs)
        except Exception:  # noqa: BLE001 - a missing floor must not lose the album
            return None
        return self.PHOTOS_PER_SPREAD_FLOOR * int(min_spreads)
