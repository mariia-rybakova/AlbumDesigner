import uvicorn
import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from schema.image_selection_schema import AlbumResponse,AlbumRequest
from src.image_selection import auto_selection
from protos.get_data_protos import get_info_protobufs
from ptinfra import intialize, get_logger

#from ptinfra.config import get_variable

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize FastAPI and add variables"""
    logger = get_logger(__name__, 'DEBUG')

    # add model to app state
    app.package = {'logger': logger}

    print("App is starting up...")
    yield
    print("App is shutting down...")


app = FastAPI(title='AI Album Designer Service',
              description='Smart Album Designer',
              version='1',
              allow_origins=["http://localhost:3000"],
              terms_of_service=None,
              contact=None,
              license_info=None,
              lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"])

settings_filename = os.environ.get('HostingSettingsPath',
                                   '/ptinternal/pictures/hosting/ai_settings_audiobeat.json.txt')
intialize('ContextCluster', settings_filename)


@app.get("/")
async def root():
    return {"Message": "This page is for Album design"}


@app.post("/selectPhotos/", response_model=AlbumResponse)
async def select_images(request: AlbumRequest):
    logger = app.get('logger', None)
    debug = app.get('debug', False)

    # File paths
    queries_file_path = r'files/queries_features.pkl'
    tags_features_file = r'files/tags.pkl'

    # Call image selection function (assuming it's defined elsewhere)
    gallery_info_df, is_wedding = get_info_protobufs(project_base_url= request.base_url, logger=logger)

    images_selected, errors = auto_selection(
        request.base_url,
        request.photoId,  # Adjusted parameter name
        request.tags,
        request.peopleIds,
        request.relation,
        queries_file_path,
        tags_features_file,
        debug,
        logger=logger
    )

    if errors:
        if logger:
            logger.error(f"Error in auto-selection: {errors}")
        return AlbumResponse(error=True, error_description=str(errors), result=None)

    if images_selected:
        if logger:
            logger.info(f"Number of Selected Images: {len(images_selected)}")
        return AlbumResponse(
            error=False,
            result={"photos": images_selected}
        )

    if logger:
        logger.error("No images were selected")
    return AlbumResponse(error=True, error_description="No images selected", result=None)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080, timeout_keep_alive=15000)


