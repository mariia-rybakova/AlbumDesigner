import os
import pickle
import shutil

import clip
from clip import tokenize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image

import torch
from tqdm import tqdm
from utils import hparams as hp
from utils.define_sim_threshold import define_threshold
from utils.get_story_sequence import get_tag_list
from utils.last_keywords import keywords

from utils.get_file_name import get_file_names
from utils.plot_utils import plot_images


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class ImageSearch:

    def __init__(self, tags: list, image_paths: list, clip_path: str,
                 img_pt_model_path: str = '', txt_pt_model_path: str = '') -> None:
        self.tags = tags
        self.image_paths = image_paths
        self.model_path = clip_path
        # load clip model
        self.model, self.preprocess = clip.load(hp.clip_model, device, jit=False)
        checkpoint = torch.load(clip_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if device == 'cpu':
            self.model.float()
        self.tag_features = {}
        self.image_features = None
        self.tag_similarities = {}
        self.sorted_images = {}


    def comp_tag_features(self, tag: str) -> np.array:
        queries_df = dict()
        queries = list(queries_df['query'])
        text_tokens = tokenize(queries)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=1, keepdim=True)
        return text_features.cpu().numpy()


    def comp_tags_features(self):
        for tag in self.tags:
            self.tag_features[tag] = self.comp_tag_features(tag)

    def comp_images_features(self):
        self.image_paths.sort()
        images = []
        for image_file in tqdm(self.image_paths[:]):
            image = Image.open(image_file).convert('RGB')
            images.append(preprocess(image))
        image_input = torch.tensor(np.stack(images)).to(device)
        with torch.no_grad():
            self.image_features = self.model.encode_image(image_input).float()
        self.image_features /= self.image_features.norm(dim=-1, keepdim=True)
        self.image_features = self.image_features.cpu().numpy()


    def comp_tag_similarity(self, tag):
        similarity = self.tag_features[tag] @ self.image_features.T
        max_similarity = np.max(similarity, axis=0)
        return max_similarity

    def sort_tag_images(self, tag):
        sorted_sim, sorted_img_names = map(list, zip(*sorted(zip(self.tag_similarities[tag], self.image_paths),
                                                             reverse=True)))
        return sorted_sim, sorted_img_names

    def search(self) -> dict:
        print('Processing tags ...')
        self.comp_tags_features()
        print('Processing images ...')
        self.comp_images_features()
        print('Computing similarity ...')
        for tag in self.tag_features.keys():
            self.tag_similarities[tag] = self.comp_tag_similarity(tag)
        print('Ranging images ...')
        for tag in self.tag_similarities.keys():
            sorted_sim, sorted_img_names = self.sort_tag_images(tag)
            self.sorted_images[tag] = [sorted_img_names, sorted_sim]
        return self.sorted_images


def keywords_search():
    version = '1'
    category_name = 'Wedding'
    category_num = 1
    gallery = 26775277
    image_dir = 'f:\\Data\\pic_time\\gallery_categories\\categories_datasets\\categories_test_v4\\cat{}\\{}'\
        .format(category_num, gallery)
    clip_model_path = 'F:\\Data\\pic_time\\models\\clip_finetuned_used\\clip_models\\clip_model_v{}.pt'.format(version)
    img_pt_model_path = 'F:\\models\\clip-image-encoder\\0\\model.pt'
    txt_pt_model_path = 'F:\\models\\clip-text-encoder\\0\\model.pt'
    keywords_path = 'f:\\Programming\\Projects\\pic_time\\info\\keywords\\vendor_keywords.xlsx'
    _, image_path_list = get_file_names(image_dir)

    print('number of images', len(image_path_list))

    # read tags
    # tags = get_tag_list(vendor=True)[:5]
    tags = keywords[category_name][:]

    # search images
    image_path_list = image_path_list[:5000]
    num_images = len(image_path_list)
    image_search = ImageSearch(tags, image_path_list, clip_model_path, )
                               # img_pt_model_path=img_pt_model_path, txt_pt_model_path=txt_pt_model_path)
    sorted_images = image_search.search()

    # ### analyze sorter similarity to define threshold
    sorted_images_dir = 'f:\\Data\\pic_time\\image_search_threshold\\Wedding\\{}'.format(gallery)
    if os.path.exists(sorted_images_dir):
        shutil.rmtree(sorted_images_dir)
    os.makedirs(sorted_images_dir)
    sim_pickle_dir = 'f:\\Projects\\pic_time\\results\\image_search_threshold\\sorted_sims_pkl'
    for tag in tags:
        tag_sorted_images_dir = os.path.join(sorted_images_dir, tag)
        pickle_name = f'{category_num}_{gallery}_{tag}_sim.pkl'
        pickle_path = os.path.join(sim_pickle_dir, pickle_name)
        if os.path.exists(tag_sorted_images_dir):
            shutil.rmtree(tag_sorted_images_dir)
        os.makedirs(tag_sorted_images_dir)
        print('Copy images for {} ...'.format(tag))
        count = 1
        (up_id, up_sim), (low_id, low_sim) = define_threshold(sorted_images[tag][1], plot=False)
        selected_images, selected_sims = sorted_images[tag][0][:low_id], sorted_images[tag][1][:low_id]
        print('Number of selected images', len(selected_images))
        for image_path, sim in tqdm(zip(selected_images, selected_sims)):
            label = 'high' if sim > up_sim else 'low'
            image_name = '{}_{}_{}_{}'.format(count, label, sim, image_path.split('\\')[-1])
            tar_image_path = os.path.join(tag_sorted_images_dir, image_name)
            shutil.copy(image_path, tar_image_path)
            count += 1

        sorted_sims = sorted_images[tag][1]
        # save sorted similarity to pickle file
        with open(pickle_path, 'wb') as fp:
            pickle.dump(sorted_sims, fp)
        # plot sorted similarity
        plt.plot(sorted_sims)
        plt.grid(True)
        # plt.show()
    # ###

    # create pdf
    results_path = 'f:\\Projects\\pic_time\\results\\image_search_keywords\\{}_{}.pdf'.format(category_name, num_images)
    pdf = PdfPages(results_path)
    for tag in sorted_images.keys():
        image_paths, sims = sorted_images[tag][0], sorted_images[tag][1]
        tag_fig = plot_images(image_paths[:20], sims[:20], tag)
        pdf.savefig(tag_fig)
        plt.close()
        # tag_fig = plot_images(image_paths[20:40], sims[20:40], tag)
        # pdf.savefig(tag_fig)
        # plt.close()
    pdf.close()


def run():
    clip_model_path = 'F:\\Data\\pic_time\\models\\clip_finetuned_model\\clip_keywords_selected\\iteration_2\\model_0.pt'
    image_dir = 'I:\\Projects\\pic_time\\results\\CLIP_fine_tuned\\keywords_queries_settings\\Wedding_Dress'
    _, image_path_list = get_file_names(image_dir)
    # tags = ['Flowers', 'Suits', 'Shoes', 'Wedding Cake', 'Dishes', 'Wedding Dress', 'Food', 'Dancing',
    #         'Jewelry', 'Hair', 'Indoors', 'Outdoors', 'Makeup', 'Silverware', 'Wedding Ceremony', 'Tables']
    tags = get_tag_list(vendor=True)
    image_search = ImageSearch(tags, image_path_list, clip_model_path)
    sorted_images = image_search.search()
    # show results
    for tag in sorted_images.keys():
        image_paths, sims = sorted_images[tag][0], sorted_images[tag][1]
        fig = plot_images(image_paths, sims, tag)
        plt.show()


if __name__ == '__main__':
    keywords_search()