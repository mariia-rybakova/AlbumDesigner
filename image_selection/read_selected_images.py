import pandas as pd
import shutil
import os

gallery_ids = [33946022]

dbscan_df = pd.read_csv(f'/data2/AI/Karmel/AlbumDesgin-dev/datasets/galleries-20240201T133138Z-001/galleries/{gallery_ids[0]}_clip_dbscan.csv')
export_df = pd.read_csv('/data2/AI/Karmel/AlbumDesgin-dev/datasets/galleries-20240201T133138Z-001/galleries/export.csv')

mask = export_df['gallery_id'].isingallery_ids
export_gallerys_df = export_df[mask]

#Search selected image ids in export CSV and collect content values
selected_images_content = []
for image_id in export_gallerys_df['photo_id']:
    export_image_row = dbscan_df['image_name'] == image_id
    if not export_gallerys_df.empty:
        selected_images_content.append(dbscan_df['cluster_context'].values[0])


result_df = pd.DataFrame({'selected-imgs': export_gallerys_df["photo_id"], 'content': selected_images_content , "gallery_id":export_gallerys_df["gallery_id"]})
result_df.to_csv(f'./results/galleries_ordered_images.csv', index=False)

print("Process completed. Results saved in result.csv.")

for data in  export_gallerys_df:
    gallery_id = export_gallerys_df['gallery_id']
    source_dir = f'/data2/AI/Karmel/AlbumDesgin-dev/datasets/galleries-20240201T133138Z-001/galleries/{gallery_id}'
    destination_dir = f'./results./{gallery_id}_selected_imgs'

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    image_id = export_gallerys_df['photo_id']
    source_path = os.path.join(source_dir, f'{image_id}.jpg')  # Adjust the file extension as needed
    destination_path = os.path.join(destination_dir, f'{image_id}.jpg')
    shutil.copy(source_path, destination_path)

print("Process completed. Results saved in result.csv. Selected images copied to the destination directory.")