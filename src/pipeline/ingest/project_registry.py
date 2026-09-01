"""Look the project up in Mongo: is it vectorised, and under which model."""

from __future__ import annotations

from bson.objectid import ObjectId

from src.pipeline.contracts import AlbumContext
from src.pipeline.registry import register
from src.pipeline.substage import SubStage


@register
class ProjectRegistrySubStage(SubStage):
    """Fetch ``isInVectorDatabase`` / ``imageModelVersion`` for the project.

    A lookup failure is not fatal: the original code logged it and fell back to
    reading embeddings from the gallery's own ``.pai`` file, which is what the
    ``None`` values below cause downstream.
    """

    name = "ingest.project_registry"

    def execute(self, context: AlbumContext) -> AlbumContext:
        collection = context.services.project_status_collection
        project_id = context.project_id
        logger = context.logger

        try:
            logger.info(f"Fetch the document from the collection {collection}")
            key = project_id if isinstance(project_id, int) else ObjectId(project_id)
            doc = collection.find_one({"_id": key}, {"isInVectorDatabase": 1, "imageModelVersion": 1})

            if doc is None:
                logger.info(f"doc not found for project_id {project_id}")
                is_in_vector_db = None
                image_model_version = None
            else:
                logger.info(f"doc found for project_id {project_id}: {doc}")
                is_in_vector_db = doc.get("isInVectorDatabase")
                image_model_version = doc.get("imageModelVersion")
        except Exception as ex:  # noqa: BLE001 - tolerated, see docstring
            logger.warning(f"Failed to read one message: {ex}")
            is_in_vector_db = None
            image_model_version = None

        context.is_in_vector_db = is_in_vector_db
        context.facts.model_version = image_model_version
        return context
