import numpy as np
from utils.read_files_types import read_pkl_file

def generate_query(tags_file, df, logger=None):
    # Load the dictionary from the binary file
    loaded_tags_features = read_pkl_file(tags_file)

    # List to store indices of rows with empty embeddings
    delete_empty_images = []

    # Iterate over each row in the DataFrame
    for idx, row in df.iterrows():
        image_id = row['image_id']
        image_features = row['embedding']

        if isinstance(image_features, list):
            image_features = np.array(image_features)

        if image_features.shape[0] == 0:
            delete_empty_images.append(idx)
            continue

        tags_similarities = {}
        for tag in loaded_tags_features:
            sub_tags = {}
            for tag_feature in loaded_tags_features[tag]:
                similarity = loaded_tags_features[tag][tag_feature] @ image_features.T
                max_query_similarity = np.max(similarity, axis=0)
                sub_tags[tag_feature] = max_query_similarity
            highest_sub_tag = max(sub_tags, key=sub_tags.get)
            tags_similarities[tag] = (sub_tags[highest_sub_tag], highest_sub_tag)

        # Get the most similar tag
        max_value_tag = max(tags_similarities, key=lambda k: tags_similarities[k][0])

        # Update the DataFrame with the new information
        df.at[idx, 'image_query_content'] = max_value_tag
        df.at[idx, 'image_subquery_content'] = tags_similarities[max_value_tag][1]

    # Drop rows with empty embeddings
    df = df.drop(delete_empty_images)

    return df
