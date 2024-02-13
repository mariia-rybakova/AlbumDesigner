import os
import re
import ast
import shutil



def extract_dict_from_text(text):
    # Define a regular expression pattern to match the dictionary-like structure
    pattern = r'{(?:[^{}]|(?R))*}'

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

def organize_images_by_category(folder,image_dict):
    for category, image_list in image_dict.items():
        # Create directory for the category if it doesn't exist
        result_folder = os.path.join("gpt_grouping_results",category)
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        # Move images to their respective category directory
        for image_id in image_list:
            image_path = os.path.join(folder,image_id)
            if os.path.exists(image_path):
                destination = os.path.join(result_folder, image_id)
                shutil.copy(image_path, destination)
                print(f"Copied {image_id} to {category}")
            else:
                print(f"Image {image_path} not found.")



def organize_images(image_names):
    organized_dict = {}

    for image_name in image_names:
        # Split the image name by underscore to separate the components
        components = image_name.split('_')

        # Extract spread number from the second component
        spread_number = components[1][1:]  # Remove the 'S' prefix
        spread_number = f"Spread_{spread_number}"  # Format as "Spread_number"

        # Extract image ID from the last component
        image_id = components[-1].split('.')[0]  # Remove the file extension

        # Add image ID to the appropriate key in the dictionary
        if spread_number not in organized_dict:
            organized_dict[spread_number] = [image_id]
        else:
            organized_dict[spread_number].append(image_id)

    # Sort the image IDs within each spread
    for spread_number in organized_dict:
        organized_dict[spread_number] = sorted(organized_dict[spread_number])

    return organized_dict


