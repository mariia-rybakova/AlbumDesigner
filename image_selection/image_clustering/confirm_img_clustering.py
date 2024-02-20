import torch
from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

def confim_clustering():
    prompt = "<image>\nUSER:Does this image representing bride preparation ?  \nASSISTANT:"
    url = "C:\\Users\\karmel\\Desktop\\PicTime\\Projects\\AlbumDesign_dev\\datasets\\selected_imges\\selected_imges\\27807822\\7629431696_S11_O17.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(text=prompt, images=image, return_tensors="pt")

    # Generate
    generate_ids = model.generate(**inputs, max_length=30)
    processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

if __name__ == "__main__":
    # read csv file to get images with thier labels
    csv_path = 'C:\\Users\\karmel\\Desktop\\PicTime\\Projects\\AlbumDesign_dev\\image_selection\\results\\ordered_clustered_images\\27807822.csv'
    confim_clustering()