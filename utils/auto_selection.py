import os
import struct
import numpy as np
from utils.parser import CONFIGS
from ptinfra.azure.pt_file import PTFile
from io import BytesIO

from utils.ai_non_wedding_selection import smart_non_wedding_selection
from utils.ai_wedding_selection import smart_wedding_selection

def load_pre_queries_embeddings(pre_queries_name,version):

    pre_query_file_name = CONFIGS['bin_name_dictionary'].get(pre_queries_name,pre_queries_name)

    try:
        if version == 1:
            file = os.path.join('pictures/photostore/4/pre_queries', f'{pre_query_file_name}.bin')
        else:
            file = os.path.join('pictures/photostore/32/pre_queries/v2', f'{pre_query_file_name}.bin')

        pai_file_bytes = PTFile(file)  # load file
        fileBytes = pai_file_bytes.read_blob()
        fb = BytesIO(fileBytes)
        b_obs = fb.read()
    except Exception as ex:
        if version == 1:
            file_path = os.path.join(r'files/pre_queries/v1/', f'{pre_query_file_name}.bin')
        else:
            file_path = os.path.join(r'files/pre_queries/v2/', f'{pre_query_file_name}.bin')
        with open(file_path, 'rb') as f:
            b_obs = f.read()

    emb_size = struct.unpack_from('<2i', b_obs)  # [512, num_of_embeddings]
    obs = struct.unpack_from(f'<{emb_size[0] * emb_size[1]}f', b_obs[8:])
    embd_matrix = np.array(obs).reshape(emb_size[1], emb_size[0])  # rows are the embeddings -> [num_of_embeddings, 512]
    return embd_matrix


def get_tags_bins(tags,version):
    if not any(s.strip() for s in tags):
        return []
    #make the check for the model version
    tags_features = {}
    for tag in tags:
        embeddings = load_pre_queries_embeddings(tag,version)
        if tag not in tags_features:
            tags_features[tag] = []
        tags_features[tag] = embeddings

    return tags_features

def ai_selection(df, selected_photos, people_ids, focus,tags,is_wedding,density,
                          logger):
    try:
        if is_wedding:
            # Select images for creating an album
            model_version =  df.iloc[0]['model_version']
            tags_features = get_tags_bins(tags,model_version)
            ai_images_selected, errors = smart_wedding_selection(df, selected_photos, people_ids, focus,
                                                                 tags_features,density, logger)
        else:
            # Select images for creating an album
            ai_images_selected, errors = smart_non_wedding_selection(df, logger=logger)

    except Exception as e:
        logger.error(e)
        return [], 'Error:{}'.format(e)

    return ai_images_selected, errors