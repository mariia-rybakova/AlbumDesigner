import os

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def plot_images(image_paths: list, similarities: list, title: str) -> plt.Figure:

    n_row, n_col = 4, 5
    fig, axs = plt.subplots(n_row, n_col, figsize=(15, 10.5))
    for i in range(n_row):
        for j in range(n_col):
            img_num = i*n_col + j
            if img_num < len(image_paths):
                image = plt.imread(image_paths[img_num])
                image_name = os.path.basename(image_paths[img_num])[-16:]
                axs[i][j].imshow(image)
                if similarities:
                    similarity = similarities[img_num]
                else:
                    similarity = 0
                axs[i][j].set_title('{} {:.5f}'.format(image_name, similarity), fontsize=8)
            axs[i][j].set_axis_off()
    fig.suptitle(title)

    return fig


def plot_table(df):

    fig = plt.figure(figsize=(15, 10.5))
    ax = plt.subplot(111)
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=df.columns, bbox=[0, 0, 1, 1])

    return fig


def plot_distribution(tag_counter: dict, title, rel=True):
    s_tag_counter = {k: v for k, v in sorted(tag_counter.items(), key=lambda item: item[1], reverse=True)}

    total = sum(s_tag_counter.values())
    if rel:
        values = [v / total for v in s_tag_counter.values()]
    else:
        values = s_tag_counter.values()
    fig = plt.figure(figsize=(15, 10.5))
    plt.subplot(111)
    plt.bar(s_tag_counter.keys(), values, color='salmon')
    if rel:
        plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))
    plt.xticks(range(len(s_tag_counter.keys())), list(s_tag_counter.keys()), rotation='vertical')
    plt.subplots_adjust(bottom=0.32)
    plt.grid(axis='y')
    plt.title(title)
    plt.show()
    return fig


def plot_two_distribution(true_tag_counter: dict, pred_tag_counter: dict, title, rel=True):
    true_counter = {k: v for k, v in sorted(true_tag_counter.items(), key=lambda item: item[1], reverse=True)}
    pred_counter = {}
    for key in true_counter.keys():
        pred_counter[key] = pred_tag_counter[key]
    fig = plt.figure(figsize=(15, 10.5))
    plt.subplot(111)
    plt.bar(true_counter.keys(), true_counter.values(), color='g', alpha=0.5, label='True')
    plt.bar(pred_counter.keys(), pred_counter.values(), color='b', alpha=0.5, label='Predicted')
    plt.xticks(range(len(true_counter.keys())), list(true_counter.keys()), rotation='vertical')
    plt.subplots_adjust(bottom=0.32)
    plt.grid(axis='y')
    plt.legend(loc=1)
    plt.title(title)
    plt.show()
    return fig