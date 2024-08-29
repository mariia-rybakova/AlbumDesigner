import os
import sys
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from schema import albumResponse
from config import DEPLOY_CONFIGS
from src.smart_selection import auto_selection
from src.layouts_processing import generate_layouts_file
from src.album_processing import create_AI_album


#from ptinfra import intialize, get_logger, AbortRequested
#from ptinfra.config import get_variable

# sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")

app = FastAPI(title='AI Album Designer Service',
              description='Smart Album Designer',
              version='1',
              terms_of_service=None,
              contact=None,
              license_info=None)

app.add_middleware(CORSMiddleware, allow_origins=["*"])

# settings_filename = os.environ.get('HostingSettingsPath', '/ptinternal/pictures/hosting/ai_settings_audiobeat.json.txt')
# intialize('albumdesinger', settings_filename)


@app.on_event("startup")
async def startup_event():
    """Initialize FastAPI and add variables"""
    #logger = get_logger(__name__, 'DEBUG')

    # add model to app state
    app.package = {"image_query": 1,
                   'design_id': DEPLOY_CONFIGS['design_id'],
                   'logger': None}

@app.get("/")
async def root():
    return {"Message": "This page is for Album design"}

@app.post("/album", response_model=albumResponse)
async def create_album(project_base_url:str, request:Request):
    data: bytes = await request.body()
    # Select images for creating an album
    images_selected, gallery_photos_info = auto_selection(project_base_url, data.ten_photos, data.tags, data.people_ids, data.user_relation)
    layouts_path = genereate_layouts_path(design_path,desings_ids,save_path)
    images_data_dict = get_info_only_for_selected_images(images_selected,gallery_photos_info)
    album_json_result, error = create_album_AI(images_data_dict,layouts_path)

    if album_json_result:
        return {"error": False,'error_description':None, "result": album_json_result}
    else:
        return {"error": True, 'error_description': error, "result": None}

