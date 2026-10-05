import os
import struct
from functools import lru_cache

import numpy as np
from utils.configs import CONFIGS
from ptinfra.azure.pt_file import PTFile
from io import BytesIO

from src.selection.ai_non_wedding_selection import smart_non_wedding_selection
from src.selection.ai_wedding_selection import smart_wedding_selection

@lru_cache(maxsize=256)
def load_pre_queries_embeddings(pre_queries_name, version):
    """Load a concept's CLIP text embeddings, as a (n_phrases, dim) matrix.

    Cached on (name, version): every call otherwise costs a blob round-trip
    (~130 ms warm, and a failed one plus a local-disk fallback for any concept
    not uploaded to storage). Selection asks for one bin per subject -- 26 on
    the default list -- so the same handful of bins is fetched over and over
    within a single gallery. The bins are immutable build artefacts, so caching
    for the life of the process is safe; the returned array is shared, so
    callers must not mutate it in place.

    **The repo's copy first, blob storage only for a concept it lacks.** The
    v2 blob copies were rewritten on 2026-05-24 in a space the image embeddings
    are not in: 75 of 83 differ from the repo's, and against a fresh text
    embedding of their own name they score ~0.0 where the repo's score 1.0.
    Read first, they were what production scored with. Telling a class's photos
    from the rest of the gallery (AUC, 0.5 is chance) on 49994361 and
    53951381: `cake` 0.99 local / 0.63 blob, `kissing` 0.99 / 0.37,
    `centerpiece` 0.92 / 0.16, `dancing` 0.75 / 0.29 -- and on the cover side
    `kissing` scored a laughing couple above every kiss. v1 is identical in
    both places.
    """

    pre_query_file_name = CONFIGS['bin_name_dictionary'].get(pre_queries_name,pre_queries_name)

    local_dir = 'files/pre_queries/v1' if version == 1 else 'files/pre_queries/v2'
    file_path = os.path.join(local_dir, f'{pre_query_file_name}.bin')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            b_obs = f.read()
    else:
        if version == 1:
            file = os.path.join('pictures/photostore/4/pre_queries', f'{pre_query_file_name}.bin')
        else:
            file = os.path.join('pictures/photostore/32/pre_queries/v2', f'{pre_query_file_name}.bin')
        # Forward slashes: `os.path.join` on Windows would ask for a key the
        # service never wrote.
        b_obs = BytesIO(PTFile(file.replace('\\', '/')).read_blob()).read()

    emb_size = struct.unpack_from('<2i', b_obs)  # [512, num_of_embeddings]
    obs = struct.unpack_from(f'<{emb_size[0] * emb_size[1]}f', b_obs[8:])
    embd_matrix = np.array(obs).reshape(emb_size[1], emb_size[0])  # rows are the embeddings -> [num_of_embeddings, 512]
    return embd_matrix


def get_tags_bins(tags,version,logger):
    try:
        if not any(s.strip() for s in tags):
            return []
        #make the check for the model version
        tags_features = {}
        for tag in tags:
            embeddings = load_pre_queries_embeddings(tag,version)
            if tag not in tags_features:
                tags_features[tag] = []
            tags_features[tag] = embeddings

    except Exception as e:
        logger.error(e)
        return [], None, 'Error inside get_tags_bins Function:{}'.format(e)

    return tags_features

def ai_selection(df, selected_photos, people_ids, focus,tags,is_wedding,density,is_artificial_time,
                          logger, rating=None):
    try:
        spreads_dict = {}
        min_total_spreads = None
        max_total_spreads = None
        if is_wedding:
            # Select images for creating an album
            model_version =  df.iloc[0]['model_version']
            tags_features = get_tags_bins(tags,model_version,logger)
            ai_images_selected, spreads_dict, min_total_spreads, max_total_spreads, errors = smart_wedding_selection(
                df, selected_photos, people_ids, focus,
                tags_features,density,is_artificial_time, logger, rating=rating)
        else:
            # Select images for creating an album
            ai_images_selected, errors = smart_non_wedding_selection(df, logger=logger)

    except Exception as e:
        logger.error(e)
        return [], None, None, None, 'Error inside the ai_selection Function:{}'.format(e)

    return ai_images_selected, spreads_dict, min_total_spreads, max_total_spreads, errors