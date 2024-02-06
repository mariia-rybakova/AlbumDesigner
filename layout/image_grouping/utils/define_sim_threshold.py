import os
import pickle

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splrep, splev

from gallery_categories.category_queries import category_nums

DER_SLOPE_THRESH = 5e-7


def point_slope(x1, y1, x2, y2):
    m = (y2-y1)/(x2-x1)
    return m


def curve_abs_slope(points):
    slopes = []
    for i in range(len(points) - 1):
        slope_item = point_slope(i, points[i], i+1, points[i+1])
        slopes.append(abs(slope_item))
    return slopes


def define_threshold(sims, plot=False, plot_title=None):
    sims_x = list((range(0, len(sims))))
    bspl = splrep(sims_x, sims, s=2e-4, k=5)
    # define lower threshold point
    bspl_sims = splev(sims_x, bspl, der=0)
    bspl_sims_slope = curve_abs_slope(bspl_sims)
    lower_point = bspl_sims_slope.index(min(bspl_sims_slope))
    # define upper threshold point
    bspl_der = splev(sims_x, bspl, der=1)
    bspl_der_slope = curve_abs_slope(bspl_der)
    upper_point = np.argwhere(np.array(bspl_der_slope) < DER_SLOPE_THRESH)[0][0]

    if plot:
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        axes[0].plot(sims, label='sorted similarity')
        axes[0].plot(upper_point, sims[upper_point], marker='o', color='r', markersize=8)
        axes[0].text(upper_point+3, sims[upper_point]+0.01, '{}, {:.2f}'.format(upper_point, sims[upper_point]))
        axes[0].plot(lower_point, sims[lower_point], marker='o', color='g', markersize=8)
        axes[0].text(lower_point+3, sims[lower_point]+0.01, '{}, {:.2f}'.format(lower_point, sims[lower_point]))
        axes[0].grid(True)
        axes[0].legend(loc=1)
        axes[1].plot(bspl_sims, label='interpolated similarity')
        axes[1].plot(lower_point, bspl_sims[lower_point], marker='o', color='g', markersize=8)
        axes[1].grid(True)
        axes[1].legend(loc=1)
        axes[2].plot(bspl_der, label='interpolated first derivative')
        axes[2].plot(upper_point, bspl_der[upper_point], marker='o', color='r', markersize=8)
        axes[2].grid(True)
        axes[2].legend(loc=1)
        fig.suptitle(plot_title)
        plt.show()
        return (upper_point, sims[upper_point]), (lower_point, sims[lower_point]), fig

    return (upper_point, sims[upper_point]), (lower_point, sims[lower_point])


def run():
    sim_pickle_dir = 'f:\\Projects\\pic_time\\results\\image_search_threshold\\sorted_sims_pkl'
    pdf_path = 'f:\\Projects\\pic_time\\results\\image_search_threshold\\similarity_plots_pdf\\sims.pdf'
    sim_files = os.listdir(sim_pickle_dir)
    pdf = PdfPages(pdf_path)
    for sim_file in sim_files[:]:
        print(sim_file)
        sim_path = os.path.join(sim_pickle_dir, sim_file)
        with open(sim_path, 'rb') as fp:
            sims = pickle.load(fp)
        # define text query and category from which selected
        cat_num, query, _ = sim_file.split('_')
        plot_title = f'text query <{query}> is selected from category: <{category_nums[int(cat_num)]}>'
        upper_point, lower_point, fig = define_threshold(sims, plot=True, plot_title=plot_title)
        pdf.savefig(fig)
        plt.close()
        print(upper_point, lower_point)
    pdf.close()


if __name__ == '__main__':
    run()