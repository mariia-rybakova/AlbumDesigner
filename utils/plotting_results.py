import numpy as np
import os
import ast
from PIL import Image

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape
from reportlab.lib import colors

from utils.crop_v2 import smart_cropping


def plot_album(cover_img, cover_img_layout_id,sub_groups, sorted_sub_groups_dict, group_name2chosen_combinations,
                           layouts_df,logger, gallery_path, output_save_path,relation_type):
    gal_num = gallery_path.split('\\')[-1]
    save_folder = os.path.join(output_save_path,gal_num)
    os.makedirs(save_folder, exist_ok=True)
    # Create a PDF canvas with landscape orientation and A4 size (1:2 aspect ratio)
    c = canvas.Canvas(os.path.join(save_folder, f'{relation_type}.pdf'), pagesize=landscape((2000, 1000)))
    page_width, page_height = landscape((2000, 1000))

    if cover_img_layout_id is not None:
        # Plot the cover image
        # Plot the cover image
        cover_layout_info = ast.literal_eval(layouts_df.loc[int(cover_img_layout_id)]['boxes_info'])
        cover_image_id = cover_img['image_id'].values[0]
        cover_img_path = os.path.join(gallery_path, str(cover_image_id) + '.jpg')

        if os.path.exists(cover_img_path):

            img = Image.open(cover_img_path)
            np_img = np.array(img)

            # Resize the image to fit the box
            centroid = cover_img['background_centroid'].values[0]
            for box in cover_layout_info:
                x, y, w, h = box['x'], box['y'], box['width'], box['height']

                # Adjust coordinates and dimensions to fit 1:2 aspect ratio
                x = x * page_width
                y = y * page_height
                w = w * page_width
                h = h * page_height

                # Add text above the image
                text_x = x
                text_y = y + h + 20  # Adjust the 20 to move the text further above the image if needed

                # Set font and draw the text
                c.setFont("Helvetica", 24)  # Set font and size
                c.setFillColor(colors.black)  # Set text color
                #c.drawString(text_x, text_y, f"{cover_img['image_orderInScene'].iloc[0]}")  # Customize the text as needed

                box_aspect_ratio = w / h
                faces = np.copy(cover_img['faces_info'])
                cropped_x, cropped_y, cropped_w, cropped_h = smart_cropping(float(cover_img['image_as'].iloc[0]),
                                                                            faces, centroid,
                                                                            float(cover_img['diameter'].iloc[0]),box_aspect_ratio)
                # Denormalize the coordinates based on the image dimensions
                im_x = int(cropped_x * np_img.shape[1])  # Multiply by image width
                im_y = int(cropped_y * np_img.shape[0])  # Multiply by image height
                im_w = int(cropped_w * np_img.shape[1])  # Multiply by image width
                im_h = int(cropped_h * np_img.shape[0])  # Multiply by image height

                # Ensure the coordinates are within image boundaries
                im_x = max(0, min(im_x, np_img.shape[1] - 1))
                im_y = max(0, min(im_y, np_img.shape[0] - 1))
                im_w = max(0, min(im_w, np_img.shape[1] - im_x))
                im_h = max(0, min(im_h, np_img.shape[0] - im_y))

                # Crop the image using correct indexing
                cropped_image = np_img[im_y:im_y + im_h, im_x:im_x + im_w]

                # Resize the cropped image
                img = Image.fromarray(cropped_image)

                # Save the resized image to a temporary file
                temp_img_path = "cover_temp.jpg"
                img.save(temp_img_path)

                # Plot the cover image inside the box
                c.drawImage(temp_img_path, x, y, w, h)

                # Clean up the temporary image file
                os.remove(temp_img_path)

    for group_name in sorted_sub_groups_dict.keys():
            if group_name not in group_name2chosen_combinations.keys():
                logger.warning(f"Group Name {group_name}  has no results ")
                continue

            group_data = group_name2chosen_combinations[group_name][0]

            number_of_spreads = len(group_data)

            # Loop over each spread
            for spread_index in range(number_of_spreads):
                c.showPage()
                layout_id = group_data[spread_index][0]
                cur_layout_info = ast.literal_eval(layouts_df.loc[layout_id]['boxes_info'])
                left_box_ids = ast.literal_eval(layouts_df.loc[layout_id]['left_box_ids'])
                right_box_ids = ast.literal_eval(layouts_df.loc[layout_id]['right_box_ids'])

                left_page_photos = list(group_data[spread_index][1])
                right_page_photos = list(group_data[spread_index][2])

                all_box_ids = left_box_ids + right_box_ids
                all_photos = left_page_photos + right_page_photos

                orig_group_name = group_name.split('*')
                parts = orig_group_name[0].split('_')
                group_id = (float(parts[0]), '_'.join(parts[1:]))
                c_group = sub_groups.get_group(group_id)

                # Loop over boxes and plot images
                for i, box in enumerate(cur_layout_info):
                    box_id = box['id']
                    if box_id not in all_box_ids:
                        print('Some error, cant find box with id: {}'.format(box_id))

                    element_index = all_box_ids.index(box_id)
                    cur_photo = all_photos[element_index]
                    c_image_id = cur_photo.id

                    c_image_info = c_group[c_group['image_id'] == c_image_id]
                    img_path = os.path.join(gallery_path, str(cur_photo.id) + '.jpg')
                    if not os.path.exists(img_path):
                        continue
                    img = Image.open(img_path)

                    x, y, w, h = box['x'], box['y'], box['width'], box['height']

                    # Adjust coordinates and dimensions to fit 1:2 aspect ratio
                    x = x * page_width
                    y = y * page_height
                    w = w * page_width
                    h = h * page_height

                    # Add text above the image
                    text_x = x
                    text_y = y + h + 20  # Adjust the 20 to move the text further above the image if needed

                    # Set font and draw the text
                    c.setFont("Helvetica", 24)  # Set font and size
                    c.setFillColor(colors.black)  # Set text color
                    #c.drawString(text_x, text_y,
                                 #f"{c_image_info['image_orderInScene'].iloc[0]}")  # Customize the text as needed


                    box_aspect_ratio = w / h
                    centroid = c_image_info['background_centroid'].values[0]
                    cropped_x, cropped_y, cropped_w, cropped_h = smart_cropping(float(c_image_info['image_as'].iloc[0]),
                                                                                c_image_info['faces_info'],
                                                                                centroid,
                                                                                float(c_image_info['diameter'].iloc[0]),box_aspect_ratio)

                    np_img = np.array(img)

                    # Denormalize the coordinates based on the image dimensions
                    im_x = int(cropped_x * np_img.shape[1])  # Multiply by image width
                    im_y = int(cropped_y * np_img.shape[0])  # Multiply by image height
                    im_w = int(cropped_w * np_img.shape[1])  # Multiply by image width
                    im_h = int(cropped_h * np_img.shape[0])  # Multiply by image height

                    # Ensure the coordinates are within image boundaries
                    im_x = max(0, min(im_x, np_img.shape[1] - 1))
                    im_y = max(0, min(im_y, np_img.shape[0] - 1))
                    im_w = max(0, min(im_w, np_img.shape[1] - im_x))
                    im_h = max(0, min(im_h, np_img.shape[0] - im_y))

                    # Crop the image using correct indexing
                    cropped_image = np_img[im_y:im_y + im_h, im_x:im_x + im_w]
                    # Resize the cropped image
                    img = Image.fromarray(cropped_image)

                    #img.thumbnail((cropped_w, cropped_h))
                    # Save the cropped and resized image to a temporary file
                    temp_img_path = f"{cur_photo.id}_temp.jpg"
                    img.save(temp_img_path)

                    # Plot the image inside the box
                    c.drawImage(temp_img_path, x, y, w, h)

                    # Clean up the temporary image file
                    os.remove(temp_img_path)

                # Add spread number as a label (optional)
                c.setFont("Helvetica", 12)
                #c.drawString(30, 30, f"Cluster {group_name} - Spread {spread_index + 1}")


    # Save the PDF
    c.save()