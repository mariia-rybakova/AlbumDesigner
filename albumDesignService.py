import uvicorn
import os
import sys
import json
import psutil
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from schema import albumResponse
from config import DEPLOY_CONFIGS
from src.smart_selection import auto_selection
from src.album_processing import create_automatic_album
from utils.generate_layout_file import genereate_layouts_path
from utils.get_images_data import get_info_only_for_selected_images
from ptinfra import intialize, get_logger, AbortRequested

#from ptinfra.config import get_variable

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize FastAPI and add variables"""
    logger = get_logger(__name__, 'DEBUG')

    # add model to app state
    app.package = {"image_query": 1,
                   'design_id': DEPLOY_CONFIGS['design_id'],
                   'logger': logger}

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


@app.post("/album/", response_model=albumResponse)
async def create_album(project_base_url: str):
    #data: bytes = await request.body()
    logger = app.package['logger']

    # Convert bytes to string (assuming UTF-8 encoding)
    #data_str = data.decode("utf-8")
    data_str = ''
    if data_str:
        # Convert string to dictionary
        data_dict = json.loads(data_str)
    else:
        data_dict = {
            'ten_photos': [9863268097,9863268114,9863268124,9863268136,9863268140,9863268145,9863268148,9863268164,9863268228,9863268257],
            'people_ids': [5,7],
            'tags': ['ceremony', 'dancing', 'bride and groom', 'walking the aisle', 'parents', 'first dance', 'kiss'],
            'user_relation': 'bride_groom'  # or 'spouse' or 'children' # designs ids
        }

    gallery_path = 'dataset\\40919535'
    design_path = r'files\designs.json'
    desings_ids = [3444, 3415, 3417, 3418, 3419, 3420, 3421, 3423, 3424, 3425, 3426, 3427, 3428, 3429, 3430, 3431, 3432,
                   3433, 3434, 3435, 3436, 3437, 3438, 3439, 3440, 3441, 3442, 3443, 3445, 3449, 3450, 3451, 3452, 3453,
                   3454, 3455, 3456, 3457, 3458, 3459, 3460, 3461, 3462, 3463, 3464, 3465, 3466, 3467, 3468, 3469, 3470,
                   3471, 3472, 3473, 3474, 3475, 3476, 3477, 3478, 3479, 3480, 3481, 3482, 3483, 3484, 3485, 3486, 3487,
                   3488, 3489, 3490, 3491, 3492, 3494, 3495, 3496, 15971, 15972, 15973, 15974, 15975, 15976, 15977,
                   15978, 15979, 15980, 15981, 15982, 15983, 15984, 15990, 15991, 15992, 15994, 15995, 15997, 15998,
                   15999, 16000, 16001, 16002, 16003, 16004, 16111, 16112, 17109, 17110]
    save_path = r'files'
    queries_file_path = r'files\queries_features.pkl'
    tags_features_file = r'files\tags.pkl'

    # Select images for creating an album
    images_selected, gallery_photos_info, errors = auto_selection(project_base_url, data_dict['ten_photos'],
                                                                  data_dict['tags'], data_dict['people_ids'],
                                                                  data_dict['user_relation'], queries_file_path,
                                                                  tags_features_file, logger=app.package['logger'])

    print("Number of Selected Images ", len(images_selected))
    if errors:
        return {"error": True, 'error_description': str(errors), "result": None}

    layouts_path = genereate_layouts_path(design_path, desings_ids, save_path, logger=logger)
    images_data_dict = get_info_only_for_selected_images(images_selected, gallery_photos_info, logger=logger)
    print("Selected images data info", len(images_data_dict))
    if len(images_data_dict) == 0:
        return {"error": True, 'error_description': "Theres no data for images that have veen selected", "result": None}

    # Get CPU usage percentage
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()

    logger.info(f"CPU Usage Before creating an Album: {cpu_usage}%")
    logger.info(f"Memory Usage Before creating an Album: {memory_info.percent}%")

    album_json_result, error = create_automatic_album(images_data_dict, layouts_path,gallery_path, logger=logger)

    if album_json_result:
        logger.info("Album created Sucessfully")
        logger.info("Result {}".format(album_json_result))
        return {"error": False, 'error_description': None, "result": album_json_result}
    else:
        logger.error("The ADD process did not succeed")
        return {"error": True, 'error_description': str(errors), "result": None}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080, timeout_keep_alive=15000)


