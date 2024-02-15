from pprint import pprint

import clip
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from .category_queries import category_queries as CAT_QUERIES
from utils.files_utils import get_file_paths

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
CLIP_MODEL = 'ViT-B/32'


class ImageContext:
    def __init__(self, tags: list, clip_path: str = '',
                 model_name: str = 'ViT-B/32', cat_queries: dict = CAT_QUERIES) -> None:
        print('model name', model_name)
        self.tags = tags
        self.model_path = clip_path
        # load clip model
        self.model, self.preprocess = clip.load(model_name, device, jit=False)
        if clip_path:
            checkpoint = torch.load(clip_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.tokenize = clip.tokenize
        if device == 'cpu':
            self.model.float()
        self.tags_features = {}
        self.cat_queries = cat_queries
        self.comp_tags_features()

    def create_queries(self, tag: str) -> list:
        if tag in self.cat_queries.keys():
            queries = self.cat_queries[tag]
        else:
            queries = ['{}'.format(tag.lower()), ]
        return queries

    def comp_tag_features(self, tag: str) -> np.array:
        queries = self.create_queries(tag)
        text_tokens = self.tokenize(queries)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens.to(device)).float()
        text_features /= text_features.norm(dim=1, keepdim=True)
        return text_features.cpu().numpy()

    def comp_tags_features(self):
        for tag in self.tags:
            self.tags_features[tag] = self.comp_tag_features(tag)

    def comp_images_features(self, image_paths: list, batch_size: int = 64) -> np.array:
        image_paths.sort()
        # images = []
        image_features = None

        for i in range(0, len(image_paths), batch_size):
            batch_image_paths = image_paths[i: i + batch_size]
            batch_images = []
            for image_path in batch_image_paths:
                image = Image.open(image_path).convert('RGB')
                # image_p = preprocess(image)
                batch_images.append(self.preprocess(image))
            batch_input = torch.tensor(np.stack(batch_images)).to(device)

            with torch.no_grad():
                batch_image_features = self.model.encode_image(batch_input).float()
            batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)
            batch_image_features = batch_image_features.cpu().numpy()
            # concatenate image_features
            if image_features is None:
                image_features = batch_image_features
            else:
                image_features = np.concatenate((image_features, batch_image_features), axis=0)
        return image_features

    def comp_tag_similarity(self, tag, image_features):
        # print('tag {} features shape'.format(tag), self.tags_features[tag].shape)
        # print('images features shape', image_features.shape)
        similarity = self.tags_features[tag] @ image_features.T
        # maximum similarity by query
        max_query_similarity = np.max(similarity, axis=0)
        # mean similarity by image
        mean_image_similarity = np.mean(max_query_similarity, axis=0)
        return mean_image_similarity

    @staticmethod
    def define_category(category_similarities):
        return max(category_similarities, key=category_similarities.get)

    def classify(self, image_paths: list) -> tuple:
        '''
        image_paths: list - list of image paths to classify context
        '''
        # print('Processing images ...')
        image_features = self.comp_images_features(image_paths)
        # print('Computing similarity ...')
        tags_similarities = {}
        for tag in self.tags_features.keys():
            tags_similarities[tag] = self.comp_tag_similarity(tag, image_features)
        # pprint(tags_similarities)
        category = self.define_category(tags_similarities)
        return category, tags_similarities[category]


def run():
    image_dir = 'H:/Data/pic_time/ordered_images/photos_8/33287564'
    image_paths = list(get_file_paths(image_dir))
    image_paths = image_paths[: 1]
    tags = list(CAT_QUERIES.keys())
    # tags = ['bride party', 'groom party', 'full party', ]
    image_context = ImageContext(tags)
    context = image_context.classify(image_paths)
    print(context)


if __name__ == '__main__':
    run()
