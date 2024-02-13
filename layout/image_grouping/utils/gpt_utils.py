import os
import re
import ast
import shutil



def extract_dict_from_text(text):
    # Define a regular expression pattern to match the dictionary-like structure
    pattern = r'{[^{}]*}'

    # Search for the pattern in the text
    match = re.search(pattern, text)

    if match:
        # If a match is found, extract the matched substring
        dict_str = match.group()

        try:
            # Use ast.literal_eval to safely evaluate the extracted substring as a dictionary
            result_dict = ast.literal_eval(dict_str)
            return result_dict
        except (SyntaxError, ValueError):
            # If the substring cannot be evaluated as a dictionary, return None or handle the error accordingly
            return None
    else:
        # If no match is found, return None or handle the case accordingly
        return None

def organize_images_by_category(imgs_folder,destination_folder,image_dict):
    for category, image_list in image_dict.items():
        # Create directory for the category if it doesn't exist
        result_folder = os.path.join(destination_folder,category)
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        # Move images to their respective category directory
        for image_id in image_list:
            image_path = os.path.join(imgs_folder,image_id)
            if os.path.exists(image_path):
                destination = os.path.join(result_folder, image_id)
                shutil.copy(image_path, destination)
                print(f"Copied {image_id} to {category}")
            else:
                print(f"Image {image_path} not found.")


def create_folders_and_move_images(imgs_folder,destination_dir, organized_dict):
    for spread, image_list in organized_dict.items():
        # Create a folder for each spread if it doesn't exist
        folder_path = os.path.join(destination_dir, spread)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Move images into the respective spread folder
        for image_name in image_list:
            image_path = os.path.join(imgs_folder, image_name)
            if os.path.exists(image_path):
                destination = os.path.join(folder_path, os.path.basename(image_name))
                shutil.copy(image_path, destination)
                print(f"Moved {image_name} to {spread} folder.")
            else:
                print(f"Image {image_name} not found.")

def organize_images(image_names):
    organized_dict = {}

    for image_name in image_names:
        # Split the image name by underscore to separate the components
        components = image_name.split('_')

        # Extract spread number from the second component
        spread_number = components[1][1:]  # Remove the 'S' prefix
        spread_number = f"Spread_{spread_number}"  # Format as "Spread_number"

        # Extract O number from the third component
        o_number = components[2][1:].split('.')[0]  # Remove the 'O' prefix and file extension

        # Add image name to the appropriate key in the dictionary
        if spread_number not in organized_dict:
            organized_dict[spread_number] = []
        organized_dict[spread_number].append((int(o_number), image_name))


    # Sort images within each spread based on O number
    for spread_number in organized_dict:
        organized_dict[spread_number] = [image_name for _, image_name in sorted(organized_dict[spread_number])]

    # Sort the dictionary based on spread number
    organized_dict = dict(sorted(organized_dict.items()))

    return organized_dict


def save_to_file(text, filename):
    with open(filename, 'w') as file:
        file.write(text)
    print("Text saved to", filename)

def read_from_file(filename):
    with open(filename, 'r') as file:
        text = file.read()
    return text

