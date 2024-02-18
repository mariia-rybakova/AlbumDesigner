import pandas as pd

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model


def read_csv(path):
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

def confim_clustering():
    model_path = "liuhaotian/llava-v1.5-7b"
    #
    # tokenizer, model, image_processor, context_len = load_pretrained_model(
    #     model_path=model_path,
    #     model_base=None,
    #     model_name=get_model_name_from_path(model_path)
    # )

    model_path = "liuhaotian/llava-v1.5-7b"
    prompt = "is this image representing bride preparation ?"
    image_file = "C:\\Users\\karmel\\Desktop\\PicTime\\Projects\\AlbumDesign_dev\\datasets\\selected_imges\\selected_imges\\27807822\\7629431696_S11_O17.jpg"

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    eval_model(args)


if __name__ == "__main__":
    # read csv file to get images with thier labels
    csv_path = 'C:\\Users\\karmel\\Desktop\\PicTime\\Projects\\AlbumDesign_dev\\image_selection\\results\\ordered_clustered_images\\27807822.csv'
    imges_labeled_dict = read_csv(csv_path)
    confim_clustering()