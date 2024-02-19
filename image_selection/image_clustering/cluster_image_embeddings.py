import os
import pandas as pd
from tqdm import tqdm
import json
import torch
import clip
from PIL import Image
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from image_selection.utils.files_utils import get_file_paths
from image_selection.utils.plot_utils import plot_images
from image_selection.clip_context_classification.category_queries import category_queries
from image_selection.clip_context_classification.classify_context import ImageContext

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load('ViT-B/32', device, jit=False)
checkpoint = torch.load('../../models/clip_model_v1.pt')
model.load_state_dict(checkpoint['model_state_dict'])

def create_time_image_dict(time_images_path):
    print('Create time image dict')
    with open(time_images_path, 'r') as fp:
        time_image_list = json.load(fp)
    time_image_dict = {}
    for item in tqdm(time_image_list):
        time_image_dict[item['photoId']] = item['dateTaken']
    return time_image_dict

time_image_dict = create_time_image_dict('C:\\Users\\karmel\\Desktop\\PicTime\\Projects\\AlbumDesign_dev\\datasets\\selected_imges\\time_files\\27807822_times.txt')
time_features = True

def read_csv(path):
    imges_labeled = {}
    data = pd.read_csv(path)
    data = data.to_numpy()
    for i in data:
        _, image_name, cluster, gallery_id = i
        if gallery_id not in imges_labeled:
            imges_labeled[gallery_id] = {}
        if cluster not in imges_labeled[gallery_id]:
            imges_labeled[gallery_id][cluster] = []
        imges_labeled[gallery_id][cluster].append(image_name)

    return imges_labeled



def compute_time_features(image_paths):
    print('Compute time features')
    time_list = []
    for image_path in image_paths:
        base_name = os.path.basename(image_path)
        image_name = base_name.split('.')[0]
        try:
            image_id = int(image_name)
        except ValueError:
            print('image name contains version', image_name)
            image_id = int(image_name.split('_')[0])
        time_list.append(time_image_dict[image_id])
    time_list = [datetime.strptime(item, '%Y-%m-%dT%H:%M:%S.%fZ') for item in time_list]
    time_list = [[(item.hour*60 + item.minute) for item in time_list]]
    time_arr = np.array(time_list)
    time_arr = StandardScaler().fit_transform(time_arr.reshape(-1, 1))
    return time_arr

def reduce_features(image_features, n_components=100):
    pca = PCA(n_components=n_components, random_state=22)
    pca.fit(image_features)
    X = pca.transform(image_features)
    X = StandardScaler().fit_transform(X)
    return X

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
def compute_features(image_paths):
    image_features = compute_image_features(image_paths, model, preprocess)
    image_features = image_features * 1.0
    print('FEATURES SHAPE', image_features.shape)
    return image_features
def cluster(image_paths, eps=0.20, metric='cosine', n_components=50):
    print('Clustering images')
    # prepare features
    # image_features = self.compute_image_features(image_paths, self.model, self.preprocess)
    image_features = compute_features(image_paths)
    #image_features = reduce_features(image_features, n_components=n_components)

    print('USE TIME FEATURES')
    time_features = compute_time_features(image_paths)
    features = np.concatenate((image_features, time_features), axis=-1)

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

def classify_context(cluster_dict):
    print('Classify cluster context')
    image_context = ImageContext(tags=list(category_queries.keys()))
    cluster_context_dict = {}
    for label, image_paths in tqdm(cluster_dict.items()):
        context, sim = image_context.classify(image_paths)
        cluster_context_dict[label] = context
    return cluster_context_dict

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
            try:
                image_id = int(base_name.split('.')[0])
            except ValueError:
                image_id = 0
                print('image name {} is not converted to int'.format(base_name))

            ordered_images.append(image_path)
        cluster_ordered_dict[key] = ordered_images
        num_ordered = len(ordered_images)
        cluster_df.loc[len(cluster_df)] = [key, num_images, num_ordered, cluster_context_dict[key]]
    if sort_by_cluster_size:
        cluster_df = cluster_df.sort_values(['num_images', 'num_ordered'], ascending=[False, False])
        cluster_df = cluster_df.reset_index(drop=True)
    return cluster_ordered_dict, cluster_df

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


if __name__ == "__main__":
    # read csv file to get images with thier labels
    csv_path = 'C:\\Users\\karmel\\Desktop\\PicTime\\Projects\\AlbumDesign_dev\\image_selection\\results\\ordered_clustered_images\\27807822.csv'
    imges_labeled_dict = read_csv(csv_path)
    gallery_num = 27807822
    image_dir = 'C:\\Users\\karmel\\Desktop\\PicTime\\Projects\\AlbumDesign_dev\\datasets\\selected_imges\\selected_imges\\{}'.format(gallery_num)
    cluster_path = '../results/ordered_clustered_images/{}_features.pdf'.format(gallery_num)
    csv_path = '../results/ordered_clustered_images/{}_features.csv'.format(gallery_num)
    image_paths = list(get_file_paths(image_dir))
    cluster_dict, cluster_labels = cluster(image_paths, eps=0.025, metric='cosine', n_components=50)
    cluster_context_dict = classify_context(cluster_dict)
    cluster_ordered_dict, cluster_df = analyze_results(cluster_dict, cluster_context_dict)
    save_clustered(cluster_dict, cluster_ordered_dict, cluster_df, cluster_context_dict, cluster_path)
    save_clusters_info(gallery_num, cluster_dict, cluster_context_dict, csv_path)
