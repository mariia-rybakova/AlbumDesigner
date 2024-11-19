import os
import shutil
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import colorsys
import time


def get_image(image_path):
    return Image.open(image_path)


def is_grayscale(image):
    image = image.convert('RGB')
    pixels = np.array(image)
    return np.all(pixels[..., 0] == pixels[..., 1]) and np.all(pixels[..., 1] == pixels[..., 2])


def get_colors_hsv(image, number_of_colors=10):
    image = image.copy()
    image.thumbnail((400, 400))
    image = image.convert("RGB")
    pixels = np.array(image).reshape(-1, 3)

    # Convert RGB to HSV
    hsv_pixels = np.array([colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0) for r, g, b in pixels])

    kmeans = KMeans(n_clusters=number_of_colors, n_init=20)
    kmeans.fit(hsv_pixels)
    colors_hsv = kmeans.cluster_centers_

    # Convert HSV back to RGB for display purposes
    colors_rgb = np.array([colorsys.hsv_to_rgb(*hsv) for hsv in colors_hsv])
    colors_rgb = (colors_rgb * 255).astype(int)

    return colors_rgb, colors_hsv


def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def plot_colors(colors_rgb):
    sorted_colors = sorted(colors_rgb,
                           key=lambda rgb: colorsys.rgb_to_hsv(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)[0])
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('off')
    for i, color in enumerate(sorted_colors):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=rgb_to_hex(color)))
    for i, color in enumerate(sorted_colors):
        hex_code = rgb_to_hex(color)
        ax.text(i + 0.5, -0.1, hex_code, ha='center', va='center', fontsize=10, fontweight='bold')
    plt.xlim(0, len(colors_rgb))
    plt.ylim(-0.5, 1)
    plt.tight_layout()
    return fig


def extract_palette(image_path, number_of_colors=6):
    image = get_image(image_path)
    colors_rgb, colors_hsv = get_colors_hsv(image, number_of_colors)
    return colors_rgb, colors_hsv


def get_color_histogram(colors_hsv):
    hist = np.zeros(64)
    for color in colors_hsv:
        h, s, v = color
        h_bin = int(h * 4)
        s_bin = int(s * 4)
        v_bin = int(v * 4)
        index = h_bin * 16 + s_bin * 4 + v_bin
        hist[int(index)] += 1
    return hist / np.sum(hist)


def cluster_images(folder_path, number_of_colors=6,pca_components=0.95):
    image_palettes = {}
    grayscale_images = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            image = get_image(image_path)

            if is_grayscale(image):
                grayscale_images.append(filename)
            else:
                _, colors_hsv = extract_palette(image_path, number_of_colors)
                image_palettes[filename] = get_color_histogram(colors_hsv)

    data = list(image_palettes.items())
    filenames, histograms = zip(*data) if data else ([], [])

    if histograms:
        similarity_matrix = cosine_similarity(histograms)

        # Compute linkage matrix
        linkage_matrix = linkage(1 - similarity_matrix, method='average')

    else:
        linkage_matrix = None

    # similarity_matrix = cosine_similarity(histograms) if histograms else []
    #
    # linkage_matrix = linkage(similarity_matrix, method='average') if histograms else None

    return linkage_matrix, filenames, grayscale_images

def calculate_inertia(linkage_matrix, labels):
    n_samples = linkage_matrix.shape[0] + 1
    cluster_distances = defaultdict(list)

    for i, (cluster1, cluster2, distance, _) in enumerate(linkage_matrix):
        cluster1, cluster2 = int(cluster1), int(cluster2)
        label1 = labels[cluster1] if cluster1 < n_samples else labels[cluster1 - n_samples]
        label2 = labels[cluster2] if cluster2 < n_samples else labels[cluster2 - n_samples]

        if label1 == label2:
            cluster_distances[label1].append(distance)

    inertia = sum(sum(distances) / len(distances) for distances in cluster_distances.values())
    return inertia


def get_clusters_from_linkage(linkage_matrix, filenames, n_clusters):
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

    clusters = defaultdict(list)
    for label, filename in zip(cluster_labels, filenames):
        clusters[label].append(filename)

    return dict(clusters)

def get_optimal_clusters(linkage_matrix, max_clusters=10):
    n_samples = linkage_matrix.shape[0] + 1
    range_n_clusters = range(2, min(max_clusters, n_samples) + 1)
    inertias = []

    for n_clusters in range_n_clusters:
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        inertia = calculate_inertia(linkage_matrix, labels)
        inertias.append(inertia)

    # Use elbow method to determine optimal number of clusters
    optimal_n_clusters = range_n_clusters[inertias.index(min(inertias))] if inertias else 1

    return optimal_n_clusters

# The rest of the functions (get_optimal_clusters, calculate_inertia, get_clusters_from_linkage)
# remain the same as they don't directly deal with color representations

if __name__ == '__main__':
    start_time = time.time()
    number_of_colors = 15
    folder_path = r'C:\Users\karmel\Desktop\AlbumDesigner\dataset\non-weddings\non-weddings\3\photos'
    linkage_matrix, filenames, grayscale_images = cluster_images(folder_path, number_of_colors=number_of_colors)

    optimal_n_clusters = get_optimal_clusters(linkage_matrix) + 1 if linkage_matrix is not None else 1

    clusters = get_clusters_from_linkage(linkage_matrix, filenames,
                                         optimal_n_clusters) if linkage_matrix is not None else {}
    # End the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print(f"Clustering took {elapsed_time:.2f} seconds.")

    if grayscale_images:
        clusters[optimal_n_clusters + 1] = grayscale_images

    base_save_path = r"C:\Users\karmel\Desktop\AlbumDesigner\results\color"

    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)

    for cluster_id, images in clusters.items():
        cluster_folder = os.path.join(base_save_path, f"Cluster_{cluster_id}")

        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)

        for image in images:
            image_id = os.path.splitext(image)[0]
            image_path = os.path.join(folder_path, image)

            save_image_path = os.path.join(cluster_folder, f"{image_id}.png")
            shutil.copy(image_path, save_image_path)

            if image not in grayscale_images:
                colors_rgb, _ = extract_palette(image_path, number_of_colors)
                plt.figure(figsize=(15, 3))
                plot_colors(colors_rgb)
                plt.title(image_id, fontsize=8)
                plt.tight_layout()

                save_palette_path = os.path.join(cluster_folder, f"{image_id}_palette.png")
                plt.savefig(save_palette_path)
                plt.close()