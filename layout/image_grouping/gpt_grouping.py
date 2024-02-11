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


def get_response(encoded_image_list):

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    contents = [
    {
        "type": "text",
        "text": "Given the following images, generate a narrative story based on these images, then each part of the generated narrative group images by each part of the sequence, and describe based on what you grouped the images, for example 100 images we grouped them to 4 groups, image 1,5,7 are representing 'wedding ceremony' part of narrative, image 8,9,10 for part 'wedding dancing', last part of narrative 'they lived happily together' image 7,9,10."
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
            "content": contents
        }],

    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
         print(response.json())
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



