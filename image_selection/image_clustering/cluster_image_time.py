import os
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from image_selection.utils.plot_utils import plot_images
from image_selection.utils.files_utils import get_file_paths
from datetime import datetime
import pytz  # Import pytz module for timezone handling

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
def compute_time(image_paths,time_image_dict):
    "Compute time Seq for each image by hour and min"
    time_img_dict_conv = dict()
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
        # time = datetime.strptime(time, '%Y-%m-%dT%H:%M:%S.%fZ')
        # time_counted = time.hour*60 + time.minute
        if image_id not in time_img_dict_conv:
            time_img_dict_conv[image_id] = 0
        time_img_dict_conv[image_id] = time
    return time_img_dict_conv


def create_time_image_dict(time_images_path):
    "Create time for image by reading the json file"
    print('Create time image dict')
    with open(time_images_path, 'r') as fp:
        time_image_list = json.load(fp)
    time_image_dict = {}
    for item in tqdm(time_image_list):
        time_image_dict[item['photoId']] = item['dateTaken']
    return time_image_dict

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
                        img_title = '{} - \n {}  \n old label {}'.format(image_name, score, std)
                elif len(score_list[img_num]) == 2:
                    image_path, score = score_list[img_num]
                    image_name = image_path.split('\\')[-1]
                    if type(score) == float:
                        img_title = '{} - {:,.4f}'.format(image_name, score)
                    else:
                        img_title = '{} - \n score {}'.format(image_name, score)
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
        image_infos= cluster_dict[label]
        for i in range(0, len(image_infos), n_row*n_col):
            batch_image_paths = image_infos[i: i + n_row * n_col]
            title = f'cluster: {label}'
            fig = plot_images2(batch_image_paths, title=title, n_row=n_row, n_col=n_col)
            pdf.savefig(fig)
            plt.close()
    pdf.close()

def order_clustered_images_by_time(time_img_dict_conv,imges_labeled_dict):
    cluster_df = pd.DataFrame(columns=['context', 'num_images', 'image_id', 'image_order'], )
    cluster_ordered_dict = {}
    for key, values in tqdm(imges_labeled_dict.items()):
        num_images = len(values)
        # compute ordered in cluster
        for label,imges_list in values.items():
            ordered_images = []
            for img_id in imges_list:
                base_name = img_id.split('_')[0]
                time = time_img_dict_conv[int(base_name)]
                image_path = os.path.join(image_dir,f'{img_id}.jpg')
                ordered_images.append((image_path, time))
                cluster_df.loc[len(cluster_df)] = [key, num_images, img_id, time]
            cluster_ordered_dict[label] = sorted(ordered_images, key=lambda x: x[1])

    cluster_df = cluster_df.sort_values(['image_order'], ascending=[False])
    cluster_df = cluster_df.reset_index(drop=True)
    return cluster_ordered_dict, cluster_df




def extract_timezone(datetime_str):
    # Extract timezone from datetime string
    return datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M:%SZ').astimezone().tzinfo
def extract_hour_minute(datetime_str):
    # Extract hour and minute from datetime string
    dt = datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M:%S.%fZ')
    return dt.hour, dt.minute
def reorganize_images_2(image_dict,imges_labeled_dict):
    new_image_dict = {}
    for label, images in image_dict.items():
        for image, datetime_str in images:
            added = False
            for new_label, new_images in new_image_dict.items():
                for idx, (new_image, new_datetime) in enumerate(new_images):
                    # %Y-%m-%dT%H:%M:%S.%fZ
                    if (datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=None),
                        extract_timezone(datetime_str)) == (datetime.strptime(new_datetime, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=None),
                                                             extract_timezone(new_datetime)):
                        new_image_dict[new_label].append((image, datetime_str))
                        added = True
                        break
                if added:
                    break
            if not added:
                new_image_dict[label] = [(image, datetime_str)]

    # Sort images within each cluster based on datetime
    for label, images in new_image_dict.items():
        new_image_dict[label] = sorted(images, key=lambda x: datetime.strptime(x[1], '%Y-%m-%dT%H:%M:%SZ'))

    return new_image_dict

def reorganize_images(image_dict, images_labeled_dict):
    new_image_dict = {}
    for label, images in image_dict.items():
        for image, datetime_str in images:
            added = False
            for new_label, new_images in new_image_dict.items():
                for idx, (new_image, new_datetime) in enumerate(new_images):
                    if (datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=None),
                        extract_timezone(datetime_str)) == (datetime.strptime(new_datetime, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=None),
                                                             extract_timezone(new_datetime)):
                        new_image_dict[new_label].append((image, datetime_str))
                        added = True
                        break
                if added:
                    break
            if not added:
                # Check if the image's time belongs to any existing time in images_labeled_dict
                time_matched = False
                for existing_label, existing_images in image_dict.items():
                    for existing_image, existing_datetime in existing_images:
                        if (datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=None),
                            extract_timezone(datetime_str)) == (datetime.strptime(existing_datetime, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=None),
                                                                 extract_timezone(existing_datetime)):
                            new_label = existing_label
                            time_matched = True
                            break
                    if time_matched:
                        break
                # If the image's time doesn't belong to any related time, create a new label
                if not time_matched:
                    new_label = f'{label}_new'
                    images_labeled_dict[new_label] = []
                new_image_dict.setdefault(new_label, []).append((image, datetime_str))

    # Sort images within each cluster based on datetime
    for label, images in new_image_dict.items():
        new_image_dict[label] = sorted(images, key=lambda x: datetime.strptime(x[1], '%Y-%m-%dT%H:%M:%S.%fZ'))

    return new_image_dict, images_labeled_dict

def reorganize_images_3(image_dict, images_labeled_dict):
    new_image_dict = {}
    for label, images in image_dict.items():
        for image, datetime_str in images:
            # Check if the image's hour and minute belong to any existing cluster in images_labeled_dict
            time_matched = False
            for existing_label, existing_images in image_dict.items():
                for existing_image, existing_datetime in existing_images:
                    if existing_image == image:
                        continue
                    if extract_hour_minute(datetime_str) == extract_hour_minute(existing_datetime): # make it range
                        time_matched = True
                        if label not in new_image_dict:
                            new_image_dict[label] = []
                        if image not in new_image_dict[label] and existing_image not in new_image_dict[label] :
                            new_image_dict[label].extend([(image, datetime_str,label), (existing_image,existing_datetime,existing_label)])
                        break
                if time_matched:
                    break
            # If the image's hour and minute don't belong to any existing cluster, create a new cluster
            if not time_matched:
                new_label = f'{label}_new'
                new_image_dict.setdefault(new_label, []).append((image, datetime_str))

    # Sort images within each cluster based on datetime
    for label, images in new_image_dict.items():
        new_image_dict[label] = sorted(images, key=lambda x: datetime.strptime(x[1], '%Y-%m-%dT%H:%M:%S.%fZ'))

    return new_image_dict, images_labeled_dict

if __name__ == "__main__":
    # read csv file to get images with thier labels
    gallery_num = 30127105
    clustered_csv_path = f'C:\\Users\\karmel\\Desktop\\PicTime\\Projects\\AlbumDesign_dev\\image_selection\\results\\ordered_clustered_images\\{gallery_num}.csv'
    image_dir = 'C:\\Users\\karmel\\Desktop\\PicTime\\Projects\\AlbumDesign_dev\\datasets\\selected_imges\\selected_imges\\{}'.format(gallery_num)
    cluster_path = '../results/ordered_clustered_images/{}_time.pdf'.format(gallery_num)
    result_csv_path = '../results/ordered_clustered_images/{}_features.csv'.format(gallery_num)
    # Get images path
    image_paths = list(get_file_paths(image_dir))
    # Get images Time
    time_image_dict = create_time_image_dict( f'C:\\Users\\karmel\\Desktop\\PicTime\\Projects\\AlbumDesign_dev\\datasets\\selected_imges\\time_files\\{gallery_num}_times.txt')
    # get images clustered by content
    imges_labeled_dict = read_csv(clustered_csv_path)
    time_img_dict_conv = compute_time(image_paths,time_image_dict)

    cluster_ordered_dict, cluster_df = order_clustered_images_by_time(time_img_dict_conv, imges_labeled_dict)
    new_image_dict = reorganize_images_2(cluster_ordered_dict, imges_labeled_dict)
    #new_image_dict, images_labeled_dict = reorganize_images_2(cluster_ordered_dict, imges_labeled_dict)
    print(new_image_dict)
    save_clustered(new_image_dict, cluster_path)