import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import read_pkl_file


def process_row(idx, row, loaded_tags_features):
    """
    Function to process a single row and return the necessary updates.
    Returns (idx, best_tag, best_sub_tag) or (idx, None, None).
    """
    image_features = row.get('embedding', None)

    # Normalize image_features -> 1D numpy vector
    if image_features is None:
        return idx, None, None
    image_features = np.asarray(image_features, dtype=float).ravel()
    if image_features.size == 0:
        return idx, None, None

    # Build a non-destructive search space (don't overwrite loaded_tags_features)
    cluster_ctx = row.get('cluster_context', None)
    if cluster_ctx is not None and cluster_ctx in loaded_tags_features:
        search_space = {cluster_ctx: loaded_tags_features[cluster_ctx]}
    else:
        search_space = loaded_tags_features

    tags_similarities = {}

    for tag, sub_features in search_space.items():
        sub_tags = {}

        # sub_features is expected to be: {sub_tag: feature_values}
        for tag_feature, feature_values in sub_features.items():
            F = np.asarray(feature_values, dtype=float)

            # Handle both single-vector and matrix-of-vectors cases
            if F.ndim == 1:
                # (dim,) · (dim,) -> scalar similarity
                if F.shape[0] != image_features.shape[0]:
                    continue  # skip dim mismatch
                max_query_similarity = float(np.dot(F, image_features))
            elif F.ndim == 2:
                # (num_queries, dim) @ (dim,) -> (num_queries,)
                if F.shape[1] != image_features.shape[0]:
                    continue  # skip dim mismatch
                sims = F @ image_features
                # ensure 1D vector
                sims = np.asarray(sims).ravel()
                if sims.size == 0:
                    continue
                max_query_similarity = float(np.max(sims))
            else:
                # unexpected shape; skip
                continue

            sub_tags[tag_feature] = max_query_similarity

        if sub_tags:
            # pick sub_tag with highest similarity (deterministic via key order on ties)
            highest_sub_tag = max(sub_tags, key=sub_tags.get)
            tags_similarities[tag] = (sub_tags[highest_sub_tag], highest_sub_tag)

    if not tags_similarities:
        return idx, None, None

    # Get the most similar tag (primary: similarity score)
    # If several tags tie exactly, Python's max is stable w.r.t. insertion order of dicts.
    max_value_tag = max(tags_similarities, key=lambda k: tags_similarities[k][0])

    return idx, max_value_tag, tags_similarities[max_value_tag][1]


def generate_query(tags_file, df, num_workers=4, logger=None):
    """
    Process the DataFrame in parallel using ThreadPoolExecutor.
    """
    loaded_tags_features = read_pkl_file(tags_file)

    results = {}
    delete_empty_images = set()

    for idx, row in df.iterrows():
        idx, image_query_content, image_subquery_content = process_row (idx, row, loaded_tags_features)
        if image_query_content is None:
            delete_empty_images.add(idx)
        else:
            results[idx] = (image_query_content, image_subquery_content)

    # with ThreadPoolExecutor(max_workers=num_workers) as executor:
    #     futures = {executor.submit(process_row, idx, row, loaded_tags_features): idx for idx, row in df.iterrows()}
    #
    #     for future in as_completed(futures):
    #         try:
    #             idx, image_query_content, image_subquery_content = future.result()
    #             if image_query_content is None:
    #                 delete_empty_images.add(idx)
    #             else:
    #                 results[idx] = (image_query_content, image_subquery_content)
    #         except Exception as e:
    #             if logger:
    #                 logger.error(f"Error processing row {futures[future]}: {e}")

    # Update DataFrame efficiently
    if results:
        df.loc[results.keys(), ['image_query_content', 'image_subquery_content']] = list(results.values())

    # Drop rows with empty embeddings
    df.drop(index=list(delete_empty_images), inplace=True)

    return df

# def generate_query(tags_file, df, logger=None):
#     # Load the dictionary from the binary file
#     loaded_tags_features = read_pkl_file(tags_file)
#
#     # List to store indices of rows with empty embeddings
#     delete_empty_images = []
#
#     # Iterate over each row in the DataFrame
#     for idx, row in df.iterrows():
#         image_features = row['embedding']
#
#         if isinstance(image_features, list):
#             image_features = np.array(image_features)
#
#         if image_features.shape[0] == 0:
#             delete_empty_images.append(idx)
#             continue
#
#         tags_similarities = {}
#         for tag in loaded_tags_features:
#             sub_tags = {}
#             for tag_feature in loaded_tags_features[tag]:
#                 similarity = loaded_tags_features[tag][tag_feature] @ image_features.T
#                 max_query_similarity = np.max(similarity, axis=0)
#                 sub_tags[tag_feature] = max_query_similarity
#             highest_sub_tag = max(sub_tags, key=sub_tags.get)
#             tags_similarities[tag] = (sub_tags[highest_sub_tag], highest_sub_tag)
#
#         # Get the most similar tag
#         max_value_tag = max(tags_similarities, key=lambda k: tags_similarities[k][0])
#
#         # Update the DataFrame with the new information
#         df.at[idx, 'image_query_content'] = max_value_tag
#         df.at[idx, 'image_subquery_content'] = tags_similarities[max_value_tag][1]
#
#     # Drop rows with empty embeddings
#     df = df.drop(delete_empty_images)
#
#     return df
