from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms

from autoencoder.pretrained_autoencoder import Encoder as PretrainedEncoder
from autoencoder.conv_autoencoder import Autoencoder, Decoder, Encoder


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

CHECKPOINT_PATH = 'h:\\Data\\pic_time\\models\\conv_autoencoder_model\\encoder_1_15_state_dict.pth'
VGGENCODER_PATH = 'h:\\Data\\pic_time\\models\\trained_conv_ae\\vgg_ae\\encoder.pt'
# MOCOENCODER_PATH = 'f:\\Data\\pic_time\\models\\contrastive_learning\\moco_resnet50\\backbone.pth'
# MOCOENCODER_PATH = 'h:\\Data\\pic_time\\highlights\\backbones\\autoencoder_resnet50\\3\\resnet_encoder.pth'
MOCOENCODER_PATH = 'h:/Data/pic_time/highlights/Ziv_models/resnet_backbone-51_2ft.pth'
TRUNKENCODER_PATH = 'h:/Data/pic_time/highlights/Ziv_models/trunk_ep59.pth'


class TrunkEncoder:
    def __init__(self, checkpoint_path):
        print('use trunk encoder')
        print('checkpoint path ', checkpoint_path)
        self.feature_extractor = models.resnet50()
        self.feature_extractor.fc = torch.nn.Linear(self.feature_extractor.fc.in_features, 512)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # self.feature_extractor.load_state_dict(checkpoint['model_state_dict'])
        self.feature_extractor.load_state_dict(checkpoint)  # for backbone for which only model_state_dict is saved
        self.feature_extractor.to(device)
        self.transforms = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

    def preprocess(self, image):
        preprocessed_image = self.transforms(image)
        return preprocessed_image

    def encode_image(self, preprocessed_image):
        output = self.feature_extractor(preprocessed_image.to(device))
        output = torch.reshape(output, (output.shape[0], output.shape[1]))
        return output


class ContrastiveEncoder:
    def __init__(self, checkpoint_path):
        resnet = models.resnet50()
        self.feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # self.feature_extractor.load_state_dict(checkpoint['model_state_dict'])
        self.feature_extractor.load_state_dict(checkpoint)  # for backbone for which only model_state_dict is saved
        self.feature_extractor.to(device)
        self.transforms = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

    def preprocess(self, image):
        preprocessed_image = self.transforms(image)
        return preprocessed_image

    def encode_image(self, preprocessed_image):
        output = self.feature_extractor(preprocessed_image.to(device))
        output = torch.reshape(output, (output.shape[0], output.shape[1]))
        return output


class TrainedEncoder:
    def __init__(self, encoder_path):
        self.feature_extractor = torch.load(encoder_path)
        self.transforms = transforms.Compose([transforms.Resize((192, 192)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

    def preprocess(self, image):
        preprocessed_image = self.transforms(image)
        return preprocessed_image

    def encode_image(self, preprocessed_image):
        output = self.feature_extractor(preprocessed_image)
        output_copy = copy(output)
        output = output['fc']
        # print('OUTPUT SHAPE', output.shape)
        out_features = torch.flatten(output, start_dim=1)
        # print('OUTPUT SHAPE', out_features.shape)
        out_image = output_copy['r41'][0]
        out_image = torch.mean(out_image, 0)
        out_image = out_image.cpu().detach().numpy()
        # print(out_image.shape)
        # plt.imshow(out_image)
        # plt.show()
        # plt.plot(list(out_features.cpu().detach().numpy()[0]))
        # plt.show()
        return out_features


class FullImageEncoder:
    def __init__(self, checkpoint_path):
        self.feature_extractor = PretrainedEncoder().to(device)
        checkpoint = torch.load(checkpoint_path)
        self.feature_extractor.load_state_dict(checkpoint)
        self.transforms = transforms.Compose([transforms.Resize((192, 192)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])

    def preprocess(self, image):
        preprocessed_image = self.transforms(image)
        return preprocessed_image

    def encode_image(self, preprocessed_image):
        output = self.feature_extractor(preprocessed_image)
        output = output['r41']
        # out_image = torch.mean(output, 1)
        out_features = torch.flatten(output, start_dim=1)
        print('OUTPUT SHAPE', out_features.shape)
        return out_features


class ImageEncoder:
    def __init__(self, in_model, weights):
        model = in_model(weights=weights).to(device)
        if in_model in (models.vgg11, models.vgg13, models.vgg16, models.vgg19):
            self.feature_extractor = model.classifier[6]
            del model.classifier[6]
            self.feature_extractor = model
        else:
            self.feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
        # print('FEATURE_EXTRACTOR', self.feature_extractor)
        self.transforms = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

    def preprocess(self, image):
        preprocessed_image = self.transforms(image)
        return preprocessed_image

    def encode_image(self, preprocessed_image):
        output = self.feature_extractor(preprocessed_image)
        output = torch.reshape(output, (output.shape[0], output.shape[1]))
        # print('OUTPUT SHAPE', output.shape)
        return output


def load_trunk_encoder():
    model = TrunkEncoder(TRUNKENCODER_PATH)
    return model, model.preprocess


def load_contrastive_encoder():
    model = ContrastiveEncoder(MOCOENCODER_PATH)
    return model, model.preprocess


def load_trained_encoder():
    model = TrainedEncoder(VGGENCODER_PATH)
    return model, model.preprocess


def load_full_encoder():
    model = FullImageEncoder(CHECKPOINT_PATH)

    return model, model.preprocess


def load_resnet101():
    model = ImageEncoder(models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2)

    return model, model.preprocess


def load_resnet152():
    model = ImageEncoder(models.resnet152, models.ResNet152_Weights.IMAGENET1K_V2)

    return model, model.preprocess


def load_vgg():
    model = ImageEncoder(models.vgg11, models.VGG11_Weights)

    return model, model.preprocess


def run_trunc_encoder():
    image_path = 'G:/Data/pic_time/Photos/Mila_initial_photos/photos/6932617953.jpg'
    image = Image.open(image_path).convert('RGB')
    model, preprocess = load_trunk_encoder()
    preprocessed_image = preprocess(image)
    inputs = torch.tensor(np.stack([preprocessed_image, preprocessed_image]))
    outputs = model.encode_image(inputs)
    print(outputs.shape)


def run_contrastive_encoder():
    image_path = 'G:/Data/pic_time/Photos/Mila_initial_photos/photos/6932617953.jpg'
    image = Image.open(image_path).convert('RGB')
    model, preprocess = load_contrastive_encoder()
    preprocessed_image = preprocess(image)
    inputs = torch.tensor(np.stack([preprocessed_image, preprocessed_image]))
    outputs = model.encode_image(inputs)
    print(outputs.shape)


def run_trained_encoder():
    image_path = 'G:/Data/pic_time/Photos/Mila_initial_photos/photos/6932617953.jpg'
    image = Image.open(image_path).convert('RGB')
    model, preprocess = load_trained_encoder()
    preprocessed_image = preprocess(image)
    inputs = torch.tensor(np.stack([preprocessed_image, preprocessed_image]))
    model.encode_image(inputs.to(device))
    # recover image
    ae_path = 'g:\\Data\\pic_time\\models\\trained_conv_ae\\vgg_ae\\ae_model.pt'
    encoder = Encoder()
    decoder = Decoder()
    model = Autoencoder(encoder, decoder, lambda_1=25, lambda_2=1, lambda_tv=0.000001)
    model.to(device)
    checkpoint = torch.load(ae_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    out_images, loss, content_feature_loss, per_pixel_loss = model(inputs.to(device))
    print(out_images.shape)
    out_images = out_images.cpu().detach().numpy()
    out_image = np.moveaxis(out_images[0], 0, -1)
    print(out_image.shape)
    preprocessed_image = preprocessed_image.cpu().detach().numpy()
    preprocessed_image = np.moveaxis(preprocessed_image, 0, -1)
    plt.imshow(out_image)
    plt.show()


def run_full_encoder():
    checkpoint_path = 'g:/Data/pic_time/models/conv_autoencoder_model/encoder_1_25_6_state_dict.pth'
    image_path = 'G:/Data/pic_time/Photos/Mila_initial_photos/photos/6932617953.jpg'
    image = Image.open(image_path).convert('RGB')
    model, preprocess = load_full_encoder()
    preprocessed_image = preprocess(image)
    inputs = torch.tensor(np.stack([preprocessed_image, preprocessed_image]))
    model.encode_image(inputs.to(device))


def run_encoder():
    image_path = 'G:/Data/pic_time/Photos/Mila_initial_photos/photos/6932617953.jpg'
    image = Image.open(image_path).convert('RGB')
    model, preprocess = load_resnet152()
    preprocessed_image = preprocess(image)
    inputs = torch.tensor(np.stack([preprocessed_image, preprocessed_image]))
    res = model.encode_image(inputs.to(device))


if __name__ == '__main__':
    run_trunc_encoder()
