# Read Mongo data
import json, re
import shutil

from bson import json_util
import os

def read_mongoextjson_file(filename):
    with open(filename, "r") as f:
        bsondata = f.read()
        # Convert Mongo object(s) to regular strict JSON
        jsondata = re.sub(r'NumberInt\s*\(\s*(\S+)\s*\)',
                          r'\1',
                          bsondata)

        jsondata = re.sub(r'NumberLong\s*\(\s*(\S+)\s*\)',
                          r'\1',
                          jsondata)

        jsondata = re.sub(r'ISODate\s*\(\s*"([^"]+)"\s*\)',
                          r'"\1"',
                          jsondata)

        data = json.loads(jsondata, object_hook=json_util.object_hook)
        return data

# Read images ids and spreads number of gallery for the albumn and order of images
galleries = [30246437,30430170,30518750,30127105]
for gallery_id in galleries:
    if not os.path.exists(f'json_files/{gallery_id}.json'):
        print("Json file for {gallery_id} is not exists!")
        continue
    data = read_mongoextjson_file(f'json_files/{gallery_id}.json')
    number_spread = data['countCompositions']
    imges_listof_dict = data['placementsImg']
    imges_ids =[(img['photoId'],img['compositionId'], img['placementImgId']) for img in imges_listof_dict]
    print(f"Number of spreads for Album Number gallery {gallery_id}: {number_spread}")
    print(f'Number of images for the album is {len(imges_ids)} : {imges_ids}')

    # copy images pics that selected to new folder
    for i, (img_id,spread_id, place_id) in enumerate(imges_ids):
        img_path = os.path.join(f'./album_galleries/{gallery_id}', f'{img_id}.jpg')
        if not os.path.exists(img_path):
            print(f"Image number {img_id} is not exists in gallery {gallery_id}")
            continue
        os.makedirs(f"./selected_imges/{gallery_id}", exist_ok=True)
        shutil.copy(img_path,f"./selected_imges/{gallery_id}/{img_id}_S{spread_id}_O{place_id}.jpg")
