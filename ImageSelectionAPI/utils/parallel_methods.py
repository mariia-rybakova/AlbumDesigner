import copy
import pandas as pd
from multiprocessing import Pool

from ImageSelectionAPI.utils.cluster_classes_table import map_cluster_label

def process_content(row_dict):
        row_dict = copy.deepcopy(row_dict)
        cluster_class = row_dict.get('cluster_class')
        cluster_class_label = map_cluster_label(cluster_class)
        row_dict['cluster_context'] = cluster_class_label
        return row_dict

def parallel_content_processing(df):
    rows = df[['image_id', 'cluster_class']].to_dict('records')

    # make process content in parallel to get content cluster
    with Pool(processes=4) as pool:
        processed_rows = pool.map(process_content, rows)

    return pd.DataFrame(processed_rows)