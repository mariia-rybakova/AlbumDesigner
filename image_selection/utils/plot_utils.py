import os

import matplotlib.pyplot as plt
import numpy as np


def plot_images(score_list: list, title: str, n_row=4, n_col=5) -> plt.Figure:

    # n_row, n_col = 3, 3
    fig, axs = plt.subplots(n_row, n_col, figsize=(15, 10.5))
    for i in range(n_row):
        for j in range(n_col):
            try:
                img_num = i*n_col + j
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
                        img_title = '{} - ordered {}'.format(image_name, score)
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


def plot_confusion_matrix(confusion_matrix, categories):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(confusion_matrix, cmap='coolwarm', interpolation='nearest')
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=90)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.subplots_adjust(bottom=0.25)
    for (j, i), label in np.ndenumerate(confusion_matrix):
        ax.text(i, j, label, ha='center', va='center')
    return fig
