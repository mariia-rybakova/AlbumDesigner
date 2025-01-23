import pandas as pd
from multiprocessing import Pool

from utils.process_content_df import process_content


def parallel_content_processing(sorted_df):
    rows = sorted_df[['image_id', 'cluster_class']].to_dict('records')

    # make process content in parallel to get content cluster
    with Pool(processes=4) as pool:
        processed_rows = pool.map(process_content, rows)

    return pd.DataFrame(processed_rows)