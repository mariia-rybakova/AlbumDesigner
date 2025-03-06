import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.read_files_types import read_pkl_file

def process_row(idx, row, loaded_tags_features):
    """
    Function to process a single row and return the necessary updates.
    """
    image_features = row['embedding']

    if isinstance(image_features, list):
        image_features = np.array(image_features)

    if image_features is None or image_features.size == 0:
        return idx, None, None  # Mark for deletion

    tags_similarities = {}

    for tag, sub_features in loaded_tags_features.items():
        sub_tags = {}
        for tag_feature, feature_values in sub_features.items():
            similarity = feature_values @ image_features.T
            max_query_similarity = np.max(similarity, axis=0)
            sub_tags[tag_feature] = max_query_similarity

        if sub_tags:  # Ensure sub_tags is not empty
            highest_sub_tag = max(sub_tags, key=sub_tags.get)
            tags_similarities[tag] = (sub_tags[highest_sub_tag], highest_sub_tag)

    if not tags_similarities:  # Handle edge case where no tags match
        return idx, None, None

    # Get the most similar tag
    max_value_tag = max(tags_similarities, key=lambda k: tags_similarities[k][0])

    return idx, max_value_tag, tags_similarities[max_value_tag][1]


def generate_query(tags_file, df, num_workers=4, logger=None):
    """
    Process the DataFrame in parallel using ThreadPoolExecutor.
    """
    loaded_tags_features = read_pkl_file(tags_file)

    results = {}
    delete_empty_images = set()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_row, idx, row, loaded_tags_features): idx for idx, row in df.iterrows()}

        for future in as_completed(futures):
            try:
                idx, image_query_content, image_subquery_content = future.result()
                if image_query_content is None:
                    delete_empty_images.add(idx)
                else:
                    results[idx] = (image_query_content, image_subquery_content)
            except Exception as e:
                if logger:
                    logger.error(f"Error processing row {futures[future]}: {e}")

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
