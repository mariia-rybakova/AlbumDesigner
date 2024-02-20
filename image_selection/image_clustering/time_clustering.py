import os
from tqdm import tqdm
import json
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
from image_selection.utils.files_utils import get_file_paths
from image_selection.clip_context_classification.classify_context_2 import run
from image_selection.utils.plot_utils import plot_images

def read_csv(path):
    "Get context clustering for images"
    print("Get context results")
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

def sort_images_by_time(images):
    images.sort(key=lambda x: x[1])
    return images

def create_time_image_dict(time_images_path):
    "Create time for image by reading the json file"
    print('Create time image dict')
    with open(time_images_path, 'r') as fp:
        time_image_list = json.load(fp)
    time_image_dict = {}
    for item in tqdm(time_image_list):
        time_image_dict[item['photoId']] = item['dateTaken']
    return time_image_dict

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

def compute_time(image_paths,time_image_dict):
    "Compute time Seq for each image by hour and min"
    images_with_time = list()
    print('Compute time counter')
    for image_path in image_paths:
        base_name = os.path.basename(image_path)
        image_name = base_name.split('.')[0]
        try:
            image_id = int(image_name)
        except ValueError:
            print('image name contains version', image_name)
            image_id = int(image_name.split('_')[0])
        if image_id not in time_image_dict:
            print(image_id, " doesnt have time")
        time = time_image_dict[image_id]
        time = parse_date(time)
        time_counted = time.hour*60 + time.minute + time.second
        if image_id not in images_with_time:
            images_with_time.append((image_path,time))
    return images_with_time

# Function to define time intervals
def define_time_intervals(timestamps):
    # Your logic to dynamically define time intervals
    # This could involve calculating time gaps, clustering timestamps, etc.
    # For simplicity, you can divide the total time span into equal intervals.
    if len(timestamps) >= 100:
        num_intervals = 6 # maybe random between 5 to 7
    else:
        num_intervals = 3 # maybe random between 3 and 5
    interval_length = len(timestamps) // num_intervals
    intervals = [timestamps[i * interval_length:(i + 1) * interval_length] for i in range(num_intervals)]
    return intervals

# Function to assign images to time intervals
def assign_images_to_intervals(images, intervals):
    assigned_intervals = []
    for image in images:
        for i, interval in enumerate(intervals):
            if image[1] <= interval[-1][1]:
                assigned_intervals.append((image[0],image[1], i))
                break
    return assigned_intervals

def plot_images2(score_list: list, title: str, n_row=4, n_col=5) -> plt.Figure:
    # n_row, n_col = 3, 3
    fig, axs = plt.subplots(n_row, n_col, figsize=(15, 10.5))
    for i in range(n_row):
        for j in range(n_col):
            try:
                img_num = i * n_col + j
                if len(score_list[img_num]) == 3:
                    image_path, score, time = score_list[img_num]
                    image_name = os.path.basename(image_path)[-25:]
                    img_title = '{} - \n score {:,.4f} \n time {}'.format(image_name, float(score), time)

                elif len(score_list[img_num]) == 2:
                    image_path, score = score_list[img_num]
                    image_name = image_path.split('\\')[-1]
                    if type(score) == float:
                        img_title = '{} - {:,.4f}'.format(image_name, score)
                    else:
                        img_title = '{} - \n  {}'.format(image_name, score)
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

def preprocess_timestamp(timestamp):
    hour = timestamp.hour
    if hour <= 7:
        hour = timestamp.hour + 12

    new_dt_obj = timestamp.replace(hour=hour)

    return new_dt_obj



def determine_wedding_type(timestamps):
    # Extract the first and last timestamps
    start_time = timestamps[0][1]
    end_time = timestamps[-1][1]

    # Set thresholds for morning and afternoon
    morning_threshold = datetime.strptime('12:00:00', '%H:%M:%S')
    evening_threshold = datetime.strptime('18:00:00', '%H:%M:%S')

    # Check if the wedding starts in the morning or afternoon
    if start_time.time() < morning_threshold.time():
        wedding_start = "morning"
    else:
        wedding_start = "afternoon"

    # Check if the wedding ends in the evening or at night
    if end_time.time() <= evening_threshold.time():
        wedding_end = "evening"
    else:
        wedding_end = "night"

    return wedding_start, wedding_end

def save_clustered_time(cluster_list, res_path, n_row=4, n_col=5):
    print('Save clustered pdf')
    pdf = PdfPages(res_path)
    # plot images of clusters
    for number in tqdm(range(0,6)):
        image_infos= [image_info for image_info in cluster_list if image_info[2] == number]
        for i in range(0, len(image_infos), n_row*n_col):
            batch_image_paths = image_infos[i: i + n_row * n_col]
            title = f'cluster: {number} interval'
            fig = plot_images2(batch_image_paths, title=title, n_row=n_row, n_col=n_col)
            pdf.savefig(fig)
            plt.close()
    pdf.close()

def save_clustered_content(cluster_dict, res_path, n_row=4, n_col=5):
    print('Save clustered pdf')
    pdf = PdfPages(os.path.join(res_path, 'result.pdf'))
    # plot images of clusters
    for item in list(cluster_dict.keys()):
            image_infos= [image_info for image_info in cluster_dict[item]]
            for i in range(0, len(image_infos), n_row*n_col):
                batch_image_paths = image_infos[i: i + n_row * n_col]
                title = f'cluster: {item} interval'
                fig = plot_images2(batch_image_paths, title=title, n_row=n_row, n_col=n_col)
                pdf.savefig(fig)
                plt.close()
    pdf.close()
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


if __name__ == "__main__":
    # 26065526
    for gallery_num in [27807822, 30127105, 27314637]:
        clustered_csv_path = f'C:\\Users\\karmel\\Desktop\\PicTime\\Projects\\AlbumDesign_dev\\image_selection\\results\\ordered_clustered_images\\{gallery_num}.csv'
        image_dir = 'C:\\Users\\karmel\\Desktop\\PicTime\\Projects\\AlbumDesign_dev\\datasets\\selected_imges\\selected_imges\\{}'.format(gallery_num)
        cluster_result_path = '../results/ordered_clustered_images/{}'.format(gallery_num)
        cluster_path = '../results/ordered_clustered_images/{}_interval_time_sorted.pdf'.format(gallery_num)
        result_csv_path = '../results/ordered_clustered_images/{}_features.csv'.format(gallery_num)
        time_txt_path =  f'C:\\Users\\karmel\\Desktop\\PicTime\\Projects\\AlbumDesign_dev\\datasets\\selected_imges\\time_files\\{gallery_num}_times.txt'
        # Get images path
        image_paths = list(get_file_paths(image_dir))
        # Get images Time
        time_image_dict = create_time_image_dict(time_txt_path)
        # get images clustered by content
        imges_labeled_dict = read_csv(clustered_csv_path)
        timestamps = compute_time(image_paths,time_image_dict)
        # Preprocess timestamps
        preprocessed_timestamps = [(filename, preprocess_timestamp(timestamp)) for filename, timestamp in timestamps]
        # Sort images by time
        images_sorted = sort_images_by_time(preprocessed_timestamps)
        # Check wedding type
        wedding_type = determine_wedding_type(images_sorted)
        print("Wedding type:", wedding_type)
        # Cluster to different intervals
        intervals = define_time_intervals(preprocessed_timestamps)
        assigned_intervals = assign_images_to_intervals(images_sorted, intervals)
        print(assigned_intervals)
        #save_clustered_time(assigned_intervals, cluster_path)

        # Cluster based content through queries for each cluster
        clustering_content = run(assigned_intervals,cluster_result_path)
        # cluster_ordered_dict, cluster_df = analyze_results(cluster_dict, cluster_context_dict)
        save_clustered_content(clustering_content, cluster_result_path)


    # cluster_ordered_dict, cluster_df = order_clustered_images_by_time(time_img_dict_conv, imges_labeled_dict)
