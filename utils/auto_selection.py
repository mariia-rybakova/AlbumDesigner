import os
import struct
import numpy as np

from ptinfra.azure.pt_file import PTFile
from io import BytesIO

from utils.selection_tools import smart_wedding_selection,smart_non_wedding_selection

def load_pre_queries_embeddings(pre_queries_name):
    file = os.path.join('pictures/photostore/4/pre_queries', f'{pre_queries_name}.bin')
    pai_file_bytes = PTFile(file)  # load file
    fileBytes = pai_file_bytes.read_blob()
    fb = BytesIO(fileBytes)
    b_obs = fb.read()
    emb_size = struct.unpack_from('<2i', b_obs)  # [512, num_of_embeddings]
    obs = struct.unpack_from(f'<{emb_size[0] * emb_size[1]}f', b_obs[8:])
    embd_matrix = np.array(obs).reshape(emb_size[1], emb_size[0])  # rows are the embeddings -> [num_of_embeddings, 512]
    return embd_matrix



def get_tags_bins(tags):
    if not any(s.strip() for s in tags):
        return []

    tags_features = {}
    for tag in tags:
        embeddings = load_pre_queries_embeddings(tag)
        if tag not in tags_features:
            tags_features[tag] = []
        tags_features[tag] = embeddings

    return tags_features

def ai_selection(df, selected_photos, people_ids, focus,tags,is_wedding,density,
                          logger):
    try:
        if is_wedding:
            # Select images for creating an album
            tags_features = get_tags_bins(tags)
            ai_images_selected, errors = smart_wedding_selection(df, selected_photos, people_ids, focus,
                                                                 tags_features,density, logger)
        else:
            # Select images for creating an album
            ai_images_selected, errors = smart_non_wedding_selection(df, logger=logger)

    except Exception as e:
        logger.error(e)
        return [], 'Error:{}'.format(e)

    return ai_images_selected, errors