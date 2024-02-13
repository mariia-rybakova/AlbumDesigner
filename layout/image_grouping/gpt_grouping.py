import os
import base64
import requests
import time
from utils import gpt_config as config
from utils.gpt_utils import  organize_images,organize_images_by_category,extract_dict_from_text
from utils.rand_score import rand_index
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
def get_response(encoded_image_list,image_names,folder,ground_truth):

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    contents = [
    {
        "type": "text",
        "text": f"Given a following images with their ids respectively inside the following list {image_names}  to create a beautiful wedding album, organize these images in a logical and visually appealing sequence please Consider the timeline of any weddings as following:  1-Full Cover Image: Select the best image to serve as the cover of the wedding album 2- Preparing Before Wedding:* Groom and His Groomsmen: Images featuring the groom and his friends getting ready.* Bride with Her Bridesmaids: Images of the bride and her bridesmaids preparing for the ceremony. 3- Photoshoot Between Bride and Groom: Images capturing the special moments between the bride and groom before the ceremony. 4- Walking Down the Aisle: Images of the bride walking down the aisle, accompanied by family or friends, leading up to the ceremony. 5-Wedding Ceremony:Exchange of Rings and Kissing: Images from the moment of exchanging rings and the first kiss as a married couple. 6-Photosession for Bride and Groom: Images taken after the ceremony, showcasing the newlyweds together. 7-Reception:Interior Design and Cake Serving: Images of the reception hall's decor and the cake.8-Family Gathering and Dancing Night: Images of family members and guests dancing and celebrating with the couple. 9- bride and groom are together with beautiful background if the number of images inside the gallery are big, then you can add this one as well.  Please return a dictionary which contain each group sequence as a following 'full_cover_image':[image_id1] , 'preparing_Before_Wedding':[image_id3,imageid_6]...and so on based on the list images ids I provided with this text. make sure to make a sequence aims to tell the story of the wedding day in a coherent and visually engaging manner, capturing all the essential moments and emotions."
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
            {"role":"user", "content":[{"type":"text", "text":" "}]}],
        "max_tokens": 2000,

    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
         print(response.json())
         print(response.json().choices[0].message['content'])
         result = response.json().choices[0].message['content']
         predicted_dict = extract_dict_from_text(result)
         if predicted_dict is not None:
             print("Extracted dictionary:", predicted_dict)
             organize_images_by_category(folder,predicted_dict)
             print("Rand Index:", rand_index(ground_truth, predicted_dict))
         else:
             print("No dictionary found in the provided text.")



    else:
        print("Error:", response.status_code, response.json())


if __name__ == "__main__":
    print("Uploading images...")
    dir = '../../datasets/selected_imges/selected_imges/27807822'
    encoded_imges_list = []
    image_names= []
    total_resized_image_size_bytes = 0
    for img in os.listdir(dir):
        img_path = os.path.join(dir, img)
        image_names.append(img)
        processed_img, size_im = process_img(img_path)
        if processed_img is not None or size_im != 0:
            encoded_imges_list.append(processed_img)
            total_resized_image_size_bytes += len(size_im)

    total_size = total_resized_image_size_bytes  / (1024 * 1024)
    print("total size: ",total_size  / (1024 * 1024))

    if total_size >= 20:
         print("Total size is more than 20MB",total_size)
    else:
        ground_truth = organize_images(image_names)
        start_time = time.time()
        get_response(encoded_imges_list, image_names,dir,ground_truth)
        end_time = time.time()
        response_time = end_time - start_time
        print("Response time:", response_time, "seconds")





