import numpy as np
from utils.read_files_types import read_pkl_file


def generate_query(tags_file, images_data_dict, logger=None):
    if images_data_dict is None:
        logger.error("images data dict is empty cant process the query")
        return None

    # Load the dictionary from the binary file
    loaded_tags_features = read_pkl_file(tags_file)

    for image in images_data_dict:
        image_features = images_data_dict[image]['embedding']
        tags_similarities = {}
        for tag in loaded_tags_features:
            sub_tags = {}
            for tag_feature in loaded_tags_features[tag]:
                similarity = loaded_tags_features[tag][tag_feature] @ image_features.T
                # maximum similarity by query
                max_query_similarity = np.max(similarity, axis=0)
                sub_tags[tag_feature] = max_query_similarity
            highest_sub_tag = max(sub_tags, key=sub_tags.get)
            tags_similarities[tag] = (sub_tags[highest_sub_tag], highest_sub_tag)
        max_value_tag = max(tags_similarities, key=lambda k: tags_similarities[k][0])
        images_data_dict[image].update(
            {'image_query_content': max_value_tag, 'image_subquery_content': tags_similarities[max_value_tag][1]})

    return images_data_dict
