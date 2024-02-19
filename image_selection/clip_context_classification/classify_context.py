from pprint import pprint
import os
import clip
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import pandas as pd

from .category_queries import category_queries as CAT_QUERIES
from image_selection.utils.files_utils import get_file_paths

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
CLIP_MODEL = 'ViT-B/32'

"""This file Cluster the selected images based on content "Queries" Only and saved in csv file"""

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


def plot_images(score_list: list, title: str, n_row=4, n_col=5) -> plt.Figure:
    # n_row, n_col = 3, 3
    fig, axs = plt.subplots(n_row, n_col, figsize=(15, 10.5))
    for i in range(n_row):
        for j in range(n_col):
            try:
                img_num = i*n_col + j
                image_path = score_list[img_num]
                image_name = os.path.basename(image_path)[-25:]
                img_title = '{}'.format(image_name)
                image = plt.imread(image_path)
                axs[i][j].imshow(image)
                axs[i][j].set_title(img_title, fontsize=8)
                axs[i][j].set_axis_off()
            except IndexError:
                axs[i][j].set_axis_off()
    fig.suptitle(title)
    return fig

def plot_images2(score_list: list, title: str, n_row=4, n_col=5) -> plt.Figure:
        # n_row, n_col = 3, 3
        fig, axs = plt.subplots(n_row, n_col, figsize=(15, 10.5))
        for i in range(n_row):
            for j in range(n_col):
                try:
                    img_num = i * n_col + j
                    if len(score_list[img_num]) == 3:
                        image_path, score, std = score_list[img_num]
                        image_name = os.path.basename(image_path)[-25:]
                        if type(std) == float:
                            img_title = '{} - {:,.3f} +- ({:,.3f})'.format(image_name, score, std)
                        else:
                            img_title = '{} - {:,.3f} group {}'.format(image_name, score, std)
                    elif len(score_list[img_num]) == 2:
                        image_path, score = score_list[img_num]
                        image_name = image_path.split('\\')[-1]
                        if type(score) == float:
                            img_title = '{} - {:,.4f}'.format(image_name, score)
                        else:
                            img_title = '{} - score {:,.2f}'.format(image_name, float(score))
                    else:
                        image_path = score_list[img_num]
                        image_name = os.path.basename(image_path)[-25:]
                        img_title = '{}'.format(image_name)
                    image = plt.imread(image_path)
                    axs[i][j].imshow(image)
                    axs[i][j].set_title(img_title, fontsize=8)
                    axs[i][j].set_axis_off()
                except IndexError:
                    axs[i][j].set_axis_off()
        fig.suptitle(title)

        return fig


def save_clustered(cluster_dict, res_path, n_row=4, n_col=5):
    print('Save clustered pdf')
    pdf = PdfPages(res_path)
    # plot images of clusters
    for label in tqdm(cluster_dict.keys()):
        image_paths = cluster_dict[label]
        for i in range(0, len(image_paths), n_row*n_col):
            batch_image_paths = image_paths[i: i + n_row * n_col]
            title = f'cluster: {label}'
            fig = plot_images2(batch_image_paths, title=title, n_row=n_row, n_col=n_col)
            pdf.savefig(fig)
            plt.close()
    pdf.close()

def save_clusters_info(gallery_num, cluster_dict, csv_path):
    print('Save cluster info to csv file')
    image_cluster_df = pd.DataFrame(columns=['image_name', 'cluster_label', 'gallery_number'])
    for label, image_paths in tqdm(cluster_dict.items()):
        for image_path, score in image_paths:
            base_name = os.path.basename(image_path)
            image_name = base_name.split('.')[0]
            image_cluster_df.loc[len(image_cluster_df)] = [image_name, label, gallery_num]
    image_cluster_df = image_cluster_df.sort_values('image_name').reset_index(drop=True)
    image_cluster_df.to_csv(csv_path)


def run():
    gallery_num = 30127105
    image_dir = f'C:\\Users\\karmel\\Desktop\\PicTime\\Projects\\AlbumDesign_dev\\datasets\\selected_imges\\selected_imges\\{gallery_num}'
    cluster_path = '../results/ordered_clustered_images/{}.pdf'.format(gallery_num)
    csv_path = '../results/ordered_clustered_images/{}.csv'.format(gallery_num)
    image_paths = list(get_file_paths(image_dir))
    tags = list(CAT_QUERIES.keys())
    if os.path.exists(cluster_path):
        os.remove(cluster_path)
    image_context = ImageContext(tags,clip_path='C:\\Users\\karmel\\Desktop\\PicTime\\Projects\\AlbumDesign_dev\\models\\clip_model_v1.pt')
    clustering_content = {}
    for image_path in image_paths:
          image_path = [image_path]
          context = image_context.classify(image_path)
          category = context[0]
          if category not in clustering_content:
              clustering_content[category] = []
          clustering_content[category].append((image_path[0],context[1]))

    return clustering_content
    # print(clustering_content)
    # save_clustered(clustering_content, cluster_path)
    # save_clusters_info(gallery_num, clustering_content, csv_path)


if __name__ == '__main__':
    run()
