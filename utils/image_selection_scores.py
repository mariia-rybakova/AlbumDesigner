import os
import io
import sys
import time
import pickle
import shutil
import random
import clip
import torch
import math
import numpy as np
import scipy.stats as stats
from collections import defaultdict
from clusters_labels import label_list
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity_scores(im_embedding, ten_photos_embeddings):
    # Ensure im_embedding is a 2D array
    im_embedding_2d = np.array(im_embedding).reshape(1, -1)

    # Convert ten_photos_embeddings to a 2D array
    ten_photos_embeddings_2d = np.array(ten_photos_embeddings)

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(im_embedding_2d, ten_photos_embeddings_2d)

    # Flatten the result to a 1D array
    similarity_scores = similarity_scores.flatten()

    return similarity_scores

def map_cluster_label(cluster_label):
    if cluster_label == -1:
        return "None"
    elif cluster_label >= 0 and cluster_label < len(label_list):
        return label_list[cluster_label]
    else:
        return "Unknown"

def comp_tag_features(tag: str) -> np.array:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    CLIP_MODEL = 'ViT-B/32'
    model, preprocess = clip.load(CLIP_MODEL, device, jit=False)

    tokenize = clip.tokenize
    text_tokens = tokenize(tag)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens.to(device)).float()
    text_features /= text_features.norm(dim=1, keepdim=True)
    return text_features.cpu().numpy()

def calcuate_text_embedding(tags):
    tags_features = []
    for tag in tags:
        feature = comp_tag_features(tag)
        tags_features.append(feature)
    return tags_features

def calcuate_tags_score(tags, image_features):
    tags_features = calcuate_text_embedding(tags)

    tags_scores = []
    for tag_feature in tags_features:
        similarity = tag_feature @ image_features.T
        # maximum similarity by query
        max_query_similarity = np.max(similarity, axis=0)
        tags_scores.append(max_query_similarity)
    # get the highest score of tags similarity
    sorted_scores = sorted(tags_scores, reverse=True)
    return sorted_scores[0]

def calculate_scores(image, gallery_photos_info,ten_photos,people_ids, tags):
    # persons score
    if 'persons_ids' in gallery_photos_info[image]:
        persons_in_image = gallery_photos_info[image]['persons_ids']
        person_exists = 0
        missed_person = 0
        for person in persons_in_image:
            if person in people_ids:
                person_exists += 1
            else:
                missed_person += 1
        person_score = person_exists / (len(persons_in_image) + 0.00000000001) * 2
    else:
        person_score = 0.0000001

    # 10 images similarity score
    if 'embedding' in gallery_photos_info[image]:
        im_embedding = gallery_photos_info[image]['embedding']
        ten_photos_embeddings = [gallery_photos_info[im]['embedding'] for im in ten_photos]
        similarity_scores = calculate_similarity_scores(im_embedding, ten_photos_embeddings)
        similarity_score = abs(similarity_scores.mean())
    else:
        similarity_score = 0.00001

    # class matching between 10 selected images and the intent image
    if 'image_class' in gallery_photos_info[image]:
        image_class = gallery_photos_info[image]['image_class']
        ten_photos_class = [gallery_photos_info[im]['image_class'] for im in ten_photos]
        class_match_counts = ten_photos_class.count(image_class)
        if class_match_counts == 0:
            class_matching_score = 0.0001
        else:
            class_matching_score = class_match_counts / len(ten_photos_class)
    else:
        class_matching_score = 0.001

    tags_score = calcuate_tags_score(tags, gallery_photos_info[image]['embedding'])

    total_score = class_matching_score * similarity_score * person_score * gallery_photos_info[image]['ranking'] * tags_score

    return total_score
