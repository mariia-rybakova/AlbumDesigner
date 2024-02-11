import os
import base64
import requests
import time
from utils import gpt_config as config
import cv2


api_key =  config.GPT_TOKEN

# Function to encode the image
def encode_image(image):
    with open(image, "rb") as imagefile:
        image_data = imagefile.read()
        return base64.b64encode(image_data).decode('utf-8')

def process_img(image_path):
    image = cv2.imread(image_path)
    # Encode the image to JPEG format
    _, jpeg_data = cv2.imencode('.jpg', image)
    # Decode the JPEG data
    jpeg_image = cv2.imdecode(jpeg_data, cv2.IMREAD_COLOR)

    if jpeg_image is None:
        print("Error: Unable to load image", image_path)
        return None, 0
    else:
        # Get original image dimensions
        height, width = jpeg_image.shape[:2]

        # Determine if the image is landscape or portrait
        if width > height:  # Landscape
            aspect_ratio = width / float(height)
            new_width = 640
            new_height = int(new_width / aspect_ratio)
        else:  # Portrait
            aspect_ratio = height / float(width)
            new_height = 640
            new_width = int(new_height / aspect_ratio)

        # Resize the image
        resized_image = cv2.resize(jpeg_image, (new_width, new_height))

        resized_image_size_bytes = cv2.imencode('.jpeg', resized_image)[1].tobytes()
        print("Size of the image is", len(resized_image_size_bytes) / (1024 * 1024))
        # Check if adding this resized image would exceed the total size limit
        if len(resized_image_size_bytes) > 20 * 1024 * 1024:  # 20MB limit
            return None, 0  # Return None and 0 if adding this image would exceed the limit


    return encode_image(image_path), resized_image_size_bytes

#"text": "Given the following images, generate a narrative story based on these images, then each part of the generated narrative group images by each part of the sequence, and describe based on what you grouped the images, for example 100 images we grouped them to 4 groups, image 1,5,7 are representing 'wedding ceremony' part of narrative, image 8,9,10 for part 'wedding dancing', last part of narrative 'they lived happily together' image 7,9,10."
def get_response(encoded_image_list):

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    contents = [
    {
        "type": "text",
        "text": "group the following images into groups each group has 3 or 4 images based on the content and color"
    }
]

    for base64_image in encoded_image_list:
        message = { "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "low"
                    }}
        contents.append(message)

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [{
            "role": "user",
            "content": contents,

        }, { "role": "assistant", "content": 'In order to group these images effectively, I will group them considering both content and' }, {"role": "assistant", "content":"color similarities. Here's a possible grouping based on these criteria:\n\nGroup 1 (Images with predominantly green and natural background):\n- Image showing three individuals in suits standing in front of greenery.\n- Image showing two individuals, one in a bridal dress and one in a suit, standing in a forested area.\n- Image showing a couple on a wooden dock with green trees and a blue sky in the background.\n\nGroup 2 (Images with bright, airy indoor or outdoor settings with a focus on bridal attire):\n- Image of an individual in a white robe with a bright interior background.\n- Image of two individuals, one in a wedding dress and the other in a suit, standing inside with a white wall and stairs in the background.\n- Image of a couple on a beach, with one individual in a wedding dress and the other in a suit, in a black and white photo.\n- Image of a couple on the wooden dock where both individuals are smiling and facing each other, with a light and natural color palette.\n\nGroup 3 (Images with a focus on formal group settings):\n- Image showing a large group of people gathered on steps outside a building, many wearing formal attire.\n- Image showing two children in white dresses and others walking in a procession inside a building with blue accents on their clothing.\n\nThe images that do not fit neatly into the above three groups due to differing elements (such as setting or number of people) may either form their own group or be considered separately if no other images match their characteristics. In this case, the remaining image would be:\n\nGroup 4 (Images that are unique in content and do not form a group with others):\n- Image showing an individual in a bridal dress with an individual in a suit, standing together indoors with a hint of greenery in the background.\n\nThese groups are somewhat flexible and subject to interpretation, but these categorizations should provide a coherent way to organize the images by similar content and color themes"},
            {"role":"user", "content":[{"type":"text", "text":"can you give me the images numbers for each group "}]}],
        "max_tokens": 1000,

    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
         print(response.json())
         print(response.json().choices[0].message['content'])
    else:
        print("Error:", response.status_code, response.json())


if __name__ == "__main__":
    print("Uploading images...")
    dir = '../../datasets/selected_imges/selected_imges/test'
    encoded_imges_list = []
    total_resized_image_size_bytes = 0
    for img in os.listdir(dir):
        img_path = os.path.join(dir, img)
        processed_img, size_im = process_img(img_path)
        if processed_img is not None or size_im != 0:
            encoded_imges_list.append(processed_img)
            total_resized_image_size_bytes += len(size_im)

    total_size = total_resized_image_size_bytes  / (1024 * 1024)
    print("total size: ",total_size  / (1024 * 1024))

    if total_size >= 20:
         print("Total size is more than 20MB",total_size)
    else:
        start_time = time.time()
        get_response(encoded_imges_list)
        end_time = time.time()
        response_time = end_time - start_time

        print("Response time:", response_time, "seconds")



