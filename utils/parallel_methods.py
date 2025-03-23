import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from utils.process_content_df import process_content


def parallel_content_processing(sorted_df):
    rows = sorted_df[['image_id', 'cluster_class']].to_dict('records')

    # Use threads instead of multiprocessing pool
    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     processed_rows = list(executor.map(process_content, rows))
    processed_rows =[]
    for row in rows:
        processed_rows.append(process_content(row))




    return pd.DataFrame(processed_rows)