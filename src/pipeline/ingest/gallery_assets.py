"""Decode the gallery protobufs into the photo table."""

from __future__ import annotations

from src.pipeline.contracts import AlbumContext, Col, photo
from src.pipeline.registry import register
from src.pipeline.substage import SubStage
from utils.read_protos_files import load_gallery_assets


@register
class GalleryAssetsSubStage(SubStage):
    """Read every per-photo protobuf and organise them into one table.

    Pure ingest: the classification, CLIP tagging and identity resolution that
    used to happen in the same function now live in ``src/pipeline/enrich``.
    """

    name = "ingest.gallery_assets"
    provides = frozenset({
        photo(Col.IMAGE_ID),
        photo(Col.EMBEDDING),
        photo(Col.MODEL_VERSION),
        photo(Col.PERSONS_IDS),
        photo(Col.IMAGE_CLASS),
        photo(Col.CLUSTER_LABEL),
        photo(Col.CLUSTER_CLASS),
        photo(Col.RANKING),
        photo(Col.IMAGE_ORDER),
        photo(Col.IMAGE_TIME),
        photo(Col.IMAGE_COLOR),
        photo(Col.IMAGE_ORIENTATION),
        photo(Col.SCENE_ORDER),
    })

    def execute(self, context: AlbumContext) -> AlbumContext:
        photos, person_details, social_circles, error = load_gallery_assets(
            project_base_url=context.project_url,
            logger=context.logger,
            clip_df=context.clip_embeddings,
        )
        if error is not None:
            return context.fail(error)

        context.photos = photos
        context.person_details = person_details
        context.social_circles = social_circles

        if not photos.empty:
            context.facts.model_version = photos.iloc[0][Col.MODEL_VERSION]

        return context
