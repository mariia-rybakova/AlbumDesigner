"""Fetch CLIP vectors from Qdrant, when the project has been vectorised."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.contracts import AlbumContext, Col
from src.pipeline.registry import register
from src.pipeline.substage import SubStage
from src.request_processing import fetch_vectors_from_qdrant
from utils.configs import CONFIGS


@register
class EmbeddingsSubStage(SubStage):
    """Load embeddings from the vector database into ``context.clip_embeddings``.

    Skipped when the project is not in the vector DB; ``ingest.gallery_assets``
    then reads them from the gallery's ``ai_search_matrix.pai`` instead.
    """

    name = "ingest.embeddings"

    def applies_to(self, context: AlbumContext) -> bool:
        return context.is_in_vector_db is True

    def execute(self, context: AlbumContext) -> AlbumContext:
        logger = context.logger
        model_version = context.facts.model_version

        logger.info(
            f'Project {context.project_id} has isInVectorDB = True, loading Clip embeddings from qdrant'
        )
        collection_name = CONFIGS["QDRANT_COLLECTION"][model_version]

        try:
            clip_dict = fetch_vectors_from_qdrant(
                context.services.qdrant_client, collection_name, context.project_id, logger=logger
            )
            if context.message is not None:
                context.message.clip_version = model_version

            clip_df = pd.DataFrame([
                {Col.IMAGE_ID: photo_id, Col.EMBEDDING: np.array(data)}
                for photo_id, data in clip_dict.items()
            ])
            clip_df[Col.MODEL_VERSION] = model_version

            # The request names the photos the album is to be built from; the
            # vector DB holds whatever has been vectorised so far. When it holds
            # fewer, the gallery is still being ingested and every count from
            # here on describes a fragment -- worth one line, because nothing
            # else in the run says so.
            requested = (context.request or {}).get('photos') or []
            if requested and len(clip_df) < len(requested):
                logger.warning(
                    f"ingest.embeddings: {len(clip_df)} vectors for the "
                    f"{len(requested)} photos this request names -- the gallery "
                    f"is still being vectorised, and the album will be composed "
                    f"from the part that is ready")
        except Exception as ex:
            if context.message is not None:
                context.message.error = True
            raise Exception('Qdrant fetch error: {}'.format(ex))

        context.clip_embeddings = clip_df
        return context
