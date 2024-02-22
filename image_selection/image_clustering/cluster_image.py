from collections import OrderedDict
from datetime import datetime
import json
import os
from pprint import pprint
import warnings

import clip
import cv2
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.decomposition import PCA
from sklearn import metrics as skmetrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
import torch
from tqdm import tqdm

from image_selection.clip_context_classification.category_queries import category_queries
from image_selection.clip_context_classification.classify_context import ImageContext
from compute_features import load_resnet101, load_resnet152, load_trained_encoder, \
    load_full_encoder, load_contrastive_encoder, load_trunk_encoder
from color_features import ColorDescriptor
from image_selection.utils.files_utils import get_file_paths
from image_selection.utils.plot_utils import plot_images


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
os.environ['OMP_NUM_THREADS'] = '1'
sns.set_theme()
warnings.simplefilter('ignore')


# ORDERED_IMAGES_PATH = 'H:/Data/pic_time/ordered_images/export (6).csv'
TIME_IMAGES_PATH = 'C:\\Users\\karmel\\Desktop\\PicTime\\Projects\\AlbumDesign_dev\\datasets\\selected_imges\\time_files\\27807822_times.txt'


# def get_ordered_ids():
#     id_df = pd.read_csv(ORDERED_IMAGES_PATH, header=0)
#     ordered_ids = list(id_df['photo_id'].values)
#     return ordered_ids

def parse_date(date_string):
    try:
        # Attempt to parse the date with the first format
        date_object = datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%SZ')
    except ValueError:
        try:
            # If parsing with the first format fails, try the second format
            date_object = datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S.%fZ')
        except ValueError:
            # If both formats fail, return None or handle the error as needed
            print("Error: Date format not recognized")
            return None
    return date_object

def preprocess_timestamp(timestamp):
    hour = timestamp.hour
    if hour <= 7:
        hour = timestamp.hour + 12

    new_dt_obj = timestamp.replace(hour=hour)

    return new_dt_obj

class ImageSimCalculator:
    '''Class for computing similarity between images'''
    def __init__(self, image_encoder, similarity_metric='cosine',
                 color_features=False, time_image_path='', image_weight=1.0,
                 color_weight=1.0, num_reduced_image_fs=0, seq_weight=0.05):
        # define image encoder
        if image_encoder == 'CLIP_ViT-B/32':
            self.model, self.preprocess = clip.load('ViT-B/32', device, jit=False)
            checkpoint = torch.load('../../models/clip_model_v1.pt')
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif image_encoder == 'Resnet101':
            self.model, self.preprocess = load_resnet101()
        elif image_encoder == 'Resnet152':
            self.model, self.preprocess = load_resnet152()
        elif image_encoder == 'TrainEnc':
            self.model, self.preprocess = load_trained_encoder()
        elif image_encoder == 'FullEnc':
            self.model, self.preprocess = load_full_encoder()
        elif image_encoder == 'ContEnc':
            self.model, self.preprocess = load_contrastive_encoder()
        elif image_encoder == 'TruncEnc':
            self.model, self.preprocess = load_trunk_encoder()
        else:
            raise ValueError('Unexpected encoder')

        # define similarity function
        if similarity_metric == 'cosine similarity':
            self.sim_function = skmetrics.pairwise.cosine_similarity
        elif similarity_metric == 'cosine':
            self.sim_function = skmetrics.pairwise.cosine_distances
        elif similarity_metric == 'euclidean':
            self.sim_function = skmetrics.pairwise.euclidean_distances
        else:
            raise ValueError('Unexpected similarity metric')

        # compute time features
        if time_image_path:
            self.time_image_dict = self.create_time_image_dict(time_image_path)
            self.time_features = True
        else:
            self.time_features = False

        # parameters
        self.color_features = color_features
        self.image_weight = image_weight
        self.color_weight = color_weight
        self.num_reduced_image_fs = num_reduced_image_fs
        self.seq_weight = seq_weight

    @staticmethod
    def compute_image_features(image_paths, model, preprocess, batch_size=64):
        image_features = None
        print('Compute image features ...')
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_image_paths = image_paths[i: i + batch_size]
            batch_images = []
            for image_path in batch_image_paths:
                image = Image.open(image_path).convert('RGB')
                # image_p = self.preprocess(image)
                batch_images.append(preprocess(image))
            batch_input = torch.tensor(np.stack(batch_images)).to(device)
            with torch.no_grad():
                batch_image_features = model.encode_image(batch_input).float()
            batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)
            batch_image_features = batch_image_features.cpu().numpy()
            # concatenate image_features
            if image_features is None:
                image_features = batch_image_features
            else:
                image_features = np.concatenate((image_features, batch_image_features), axis=0)
        # print('Features shape', image_features.shape)
        return image_features

    @staticmethod
    def compute_color_features(image_paths):
        color_features = None
        print('Compute color features ...')
        color_descriptor = ColorDescriptor((3, 3, 3))
        for image_path in tqdm(image_paths):
            image = cv2.imread(image_path)
            image_color_features = np.array([color_descriptor.describe(image=image)])
            if color_features is None:
                color_features = image_color_features
            else:
                color_features = np.concatenate((color_features, image_color_features), axis=0)
        return color_features

    @staticmethod
    def reduce_features(image_features, n_components=100):
        pca = PCA(n_components=n_components, random_state=22)
        pca.fit(image_features)
        X = pca.transform(image_features)
        # X = MinMaxScaler().fit_transform(X)
        X = StandardScaler().fit_transform(X)
        return X

    def compute_features(self, image_paths):
        image_features = self.compute_image_features(image_paths, self.model, self.preprocess)
        if self.num_reduced_image_fs:
            image_features = self.reduce_features(image_features, n_components=self.num_reduced_image_fs)
        # image_features = MinMaxScaler().fit_transform(image_features)
        image_features = image_features * self.image_weight
        print('IMAGE FEATURES MEAN', np.mean(image_features))
        print('IMAGE FEATURES SHAPE', image_features.shape)
        if self.color_features:
            color_features = self.compute_color_features(image_paths)
            # color_features = MinMaxScaler().fit_transform(color_features)
            print('COLOR_WEIGHGS', self.color_weight)
            color_features = color_features * self.color_weight
            print('COLOR FEATURES MEAN', np.mean(color_features))
            print('COLOR FEATURES SHAPE', color_features.shape)
            features = np.concatenate((image_features, color_features), axis=-1)
        else:
            features = image_features
        print('FEATURES SHAPE', features.shape)
        return features

    @staticmethod
    def compute_seq_features(image_paths):
        image_seq = [float(item) for item in range(len(image_paths))]
        seq_features = np.array(image_seq)
        seq_features = StandardScaler().fit_transform(seq_features.reshape(-1, 1))
        return seq_features

    @staticmethod
    def create_time_image_dict(time_images_path):
        print('Create time image dict')
        with open(time_images_path, 'r') as fp:
            time_image_list = json.load(fp)
        time_image_dict = {}
        for item in tqdm(time_image_list):
            time_image_dict[item['photoId']] = item['dateTaken']
        return time_image_dict

    def compute_time_features(self, image_paths):
        print('Compute time features')
        time_list = []
        for image_path in image_paths:
            base_name = os.path.basename(image_path)
            image_name = base_name.split('.')[0]
            try:
                image_id = int(image_name)
            except ValueError:
                #print('image name contains version', image_name)

                image_id = int(image_name.split('_')[0])
            time_list.append(self.time_image_dict[image_id])
        time_list = [parse_date(item) for item in time_list]
        time_list = [preprocess_timestamp(item) for item in time_list]
        time_list = [[(item.hour*60 + item.minute + item.second) for item in time_list]]
        time_arr = np.array(time_list)
        time_arr = StandardScaler().fit_transform(time_arr.reshape(-1, 1))
        return time_arr

    def cluster(self, image_paths, eps=0.20, metric='cosine', n_components=50):
        print('Clustering images')
        # prepare features
        # image_features = self.compute_image_features(image_paths, self.model, self.preprocess)
        image_features = self.compute_features(image_paths)
        image_features = self.reduce_features(image_features, n_components=n_components)
        if (self.seq_weight != 0.0) and (not self.time_features):
            seq_features = self.compute_seq_features(image_paths)
            features = np.concatenate((image_features, seq_features), axis=-1)
            features = normalize(features, axis=1)
            features[:, -1:] = features[:, -1:] * self.seq_weight * n_components
        elif self.time_features:
            print('USE TIME FEATURES')
            time_features = self.compute_time_features(image_paths)
            features = np.concatenate((image_features, time_features), axis=-1)
        else:
            features = image_features

        # cluster
        cluster_tool = DBSCAN(eps=eps, min_samples=1, metric=metric)
        cluster_tool.fit(features)
        labels = cluster_tool.labels_
        cluster_dict = {label: [] for label in set(labels)}
        for i in range(len(labels)):
            cluster_dict[labels[i]].append(image_paths[i])
        selected_image_paths = sorted([cluster_dict[key][0] for key in cluster_dict.keys()])
        # sort image paths by sequence of names
        for key in cluster_dict.keys():
            cluster_dict[key] = sorted(cluster_dict[key])
        cluster_list = [value for key, value in cluster_dict.items()]
        cluster_list = sorted(cluster_list, key=lambda x: x[0])
        # create list of cluster labels
        cluster_labels = [None] * len(image_paths)
        for cluster_num, one_cluster_list in enumerate(cluster_list):
            for image_path in one_cluster_list:
                image_num = image_paths.index(image_path)
                cluster_labels[image_num] = cluster_num
        print('NUMBER OF IMAGES', len(image_paths))
        print('NUMBER OF CLUSTERS', len(cluster_dict.keys()))

        return cluster_dict, cluster_labels

    def cluster_images(self, image_paths, eps=0.20, metric='cosine', n_components=50, n_iterations=1,
                       sort_by_cluster_size=False):
        '''
        Iterative cluster images with increasing epsilon
        '''
        if len(image_paths) >= 100:
            n_cluster_items = 50
        elif len(image_paths) <= 50:
            n_cluster_items = 12
        else:
            n_cluster_items = 20

        eps_step = 0.025
        full_cluster_dict = {}
        start_id = 0
        rest_image_paths = image_paths
        for iteration in range(n_iterations):
            print('iteration {}, number of images for clustering {}'.format(iteration, len(rest_image_paths)))
            cluster_dict, cluster_labels = self.cluster(rest_image_paths, eps=eps,
                                                        metric=metric, n_components=n_components)
            print(f'iteration {iteration} number of labels created {cluster_labels} with eps {eps}')
            rest_image_paths = []
            if (iteration + 1) < n_iterations:
                for key, value in cluster_dict.items():
                    if len(value) < n_cluster_items:
                        rest_image_paths.extend(value)
                    else:
                        full_cluster_dict[start_id + key] = value
                start_id += len(full_cluster_dict.keys())
                eps = eps + eps_step
            else:
                full_cluster_dict.update(cluster_dict)
        if sort_by_cluster_size:
            # sort by size of cluster
            full_cluster_dict = OrderedDict(sorted(full_cluster_dict.items(),
                                                   key=lambda x: len(x[1]), reverse=True))
        else:
            # sort by first image_name of cluster
            full_cluster_dict = OrderedDict(sorted(full_cluster_dict.items(),
                                                   key=lambda x: x[1][0], reverse=False))

        return full_cluster_dict


def analyze_results(cluster_dict, cluster_context_dict, sort_by_cluster_size=False):
    print('Analyze clusters')
    cluster_df = pd.DataFrame(columns=['cluster', 'num_images', 'num_ordered', 'context'], )
    cluster_ordered_dict = {}
    for key, values in tqdm(cluster_dict.items()):
        num_images = len(values)
        # compute ordered in cluster
        ordered_images = []
        for image_path in values:
            base_name = os.path.basename(image_path)
            ordered_images.append(image_path)
        cluster_ordered_dict[key] = ordered_images
        num_ordered = len(ordered_images)
        cluster_df.loc[len(cluster_df)] = [key, num_images, num_ordered, cluster_context_dict[key]]
    if sort_by_cluster_size:
        cluster_df = cluster_df.sort_values(['num_images', 'num_ordered'], ascending=[False, False])
        cluster_df = cluster_df.reset_index(drop=True)
    return cluster_ordered_dict, cluster_df


def classify_context(cluster_dict):
    print('Classify cluster context')
    image_context = ImageContext(tags=list(category_queries.keys()))
    cluster_context_dict = {}
    for label, image_paths in tqdm(cluster_dict.items()):
        context, sim = image_context.classify(image_paths)

        cluster_context_dict[label] = context
    return cluster_context_dict


def save_clustered(cluster_dict, cluster_ordered_dict, cluster_df, cluster_context_dict, res_path, n_row=4, n_col=5):
    print('Save clustered pdf')
    pdf = PdfPages(res_path)
    # plot table with number of images and number of ordered images
    for i in range(0, len(cluster_df), 25):
        batch_cluster_df = cluster_df[i: i+25]
        if not batch_cluster_df.empty:
            fig, ax = plt.subplots()
            ax.axis('off')
            table = pd.plotting.table(ax, batch_cluster_df, loc='center', cellLoc='center')
            pdf.savefig(fig)
            plt.close()
    # plot images of clusters
    for label in tqdm(cluster_dict.keys()):
        image_paths = cluster_dict[label]
        ordered_image_paths = cluster_ordered_dict[label]
        for i in range(0, len(image_paths), n_row*n_col):
            batch_image_paths = image_paths[i: i + n_row * n_col]
            batch_ordereds = [item in ordered_image_paths for item in batch_image_paths]
            batch_scores = [(x, y) for x, y in zip(batch_image_paths, batch_ordereds)]
            title = f'cluster: {label}, context: {cluster_context_dict[label]}'
            fig = plot_images(batch_scores, title=title, n_row=n_row, n_col=n_col)
            pdf.savefig(fig)
            plt.close()
    pdf.close()


def save_clusters_info(gallery_num, cluster_dict, cluster_context_dict, csv_path):
    print('Save cluster info to csv file')
    image_cluster_df = pd.DataFrame(columns=['image_name', 'cluster_label', 'num_cluster_images',
                                             'cluster_context', 'gallery_number'])
    for label, image_paths in tqdm(cluster_dict.items()):
        cluster_size = len(image_paths)
        for image_path in image_paths:
            base_name = os.path.basename(image_path)
            image_name = base_name.split('.')[0]
            image_cluster_df.loc[len(image_cluster_df)] = [image_name, label, cluster_size,
                                                           cluster_context_dict[label], gallery_num]
    image_cluster_df = image_cluster_df.sort_values('image_name').reset_index(drop=True)
    image_cluster_df.to_csv(csv_path)


def run():
    gallery_num = 27807822
    image_dir = 'C:\\Users\\karmel\\Desktop\\PicTime\\Projects\\AlbumDesign_dev\\datasets\\selected_imges\\selected_imges\\{}'.format(gallery_num)
    cluster_path = '../results/embedding/{}/{}_features.pdf'.format(gallery_num,gallery_num)
    csv_path = '../results/embedding/{}/{}_features.csv'.format(gallery_num,gallery_num)
    image_paths = list(get_file_paths(image_dir))
    image_paths = image_paths[:]
    os.makedirs(f'../results/embedding/{gallery_num}', exist_ok=True)
    max_plot_images = 40

    sim_calculator = ImageSimCalculator(image_encoder='CLIP_ViT-B/32',
                                        similarity_metric='cosine',
                                        color_features=False,
                                        time_image_path=TIME_IMAGES_PATH,
                                        image_weight=1.0,
                                        color_weight=0.0,
                                        num_reduced_image_fs=0,
                                        seq_weight=1.0)
    cluster_dict = sim_calculator.cluster_images(image_paths, eps=0.5, metric='cosine',
                                                 n_components=30, n_iterations=8)
    cluster_context_dict = classify_context(cluster_dict)
    cluster_ordered_dict, cluster_df = analyze_results(cluster_dict, cluster_context_dict)
    save_clustered(cluster_dict, cluster_ordered_dict, cluster_df, cluster_context_dict, cluster_path)
    save_clusters_info(gallery_num, cluster_dict, cluster_context_dict, csv_path)


if __name__ == '__main__':
    run()
