import clip
import torch
import numpy as np
import pickle

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
CLIP_MODEL = 'ViT-B/32'

model, preprocess = clip.load(CLIP_MODEL, device, jit=False)

clip_path = r'C:\Users\karmel\Desktop\AlbumDesigner\models\clip_model_v1.pt'
if clip_path:
    checkpoint = torch.load(clip_path)
    model.load_state_dict(checkpoint['model_state_dict'])


def comp_tag_features(tag: str) -> np.array:
    text_tokens = clip.tokenize(tag)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens.to(device)).float()
    text_features /= text_features.norm(dim=1, keepdim=True)
    return text_features.cpu().numpy()

tags = ['ceremony', 'dancing', 'bride and groom', 'walking the aisle', 'parents', 'first dance', 'kiss','friends','food', 'portrait', 'group photos']
result = {}
for tag in tags:
    features = comp_tag_features(tag)
    result[tag] = features

with open(r"C:\Users\karmel\Desktop\AlbumDesigner\files\tags.pkl", "wb") as file:
    pickle.dump(result, file)

print("Saved the tags features in  files folder tags.pkl")