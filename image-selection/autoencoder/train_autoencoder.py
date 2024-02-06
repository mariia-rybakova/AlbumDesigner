from datetime import datetime
import random

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

from autoencoder.conv_autoencoder import Autoencoder, Decoder, Encoder
from utils.files_utils import get_file_names

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
random.seed(0)


class ImageData(Dataset):
    def __init__(self, image_path_list, image_transforms):
        self.image_path_list = image_path_list
        self.len = len(self.image_path_list)
        self.image_transforms = image_transforms

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image = Image.open(self.image_path_list[index]).convert('RGB')
        image = self.image_transforms(image)
        return image


class AutoencoderTrainer:
    def __init__(self, checkpoint_path=None, monitoring_path=None):
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.model = Autoencoder(self.encoder, self.decoder, lambda_1=25, lambda_2=1, lambda_tv=0.000001)
        self.model.to(device)
        # define optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.000001)
        # load checkpoint
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # define normalization transformer
        self.norm_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # define writer
        if monitoring_path is not None:
            self.writer = SummaryWriter(monitoring_path)
        # define criterion
        self.criterion = torch.nn.MSELoss()

    def train(self, image_path_list, batch_size=8, num_epochs=1, model_checkpoint_path=None, encoder_path=None,
              encoder_ts_path=None):
        train_image_transforms = T.Compose([T.Resize([192, 192]),
                                            T.ToTensor(),
                                            self.norm_transform])
        train_dataset = ImageData(image_path_list, train_image_transforms)
        print('Data are loaded, number of samples {}'.format(train_dataset.len))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        losses = []
        for epoch in range(num_epochs):
            print('Train model, epoch {}...'.format(epoch))
            for i, images in tqdm(enumerate(train_loader)):
                self.optimizer.zero_grad()
                images = images.to(device)
                out_images, loss_, content_feature_loss, per_pixel_loss = self.model(images)
                loss = self.criterion(out_images, images)
                # backpropagation
                # total_loss = loss.sum()
                total_loss = loss
                total_loss.backward()
                self.optimizer.step()
                losses.append(total_loss.item())
                # print('Batch loss', total_loss.item())
                self.writer.add_scalar('total_loss', total_loss, epoch * len(train_loader) + 1)
                # self.writer.add_scalar('content_feature_loss', content_feature_loss.sum(),
                #                        epoch * len(train_loader) + 1)
                # self.writer.add_scalar('per_pixel_loss', per_pixel_loss.sum(),
                #                        epoch * len(train_loader) + 1)

            # save autoencoder checkpoint
            if model_checkpoint_path is not None:
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': total_loss, },
                           model_checkpoint_path)
                print('Autoencoder checkpoint is saved')
            # save encoder model as torchscript
            if encoder_path is not None:
                torch.save(self.model.encoder, encoder_path)
                print('Encoder model is saved')
            if encoder_ts_path is not None:
                example_image = torch.rand(1, 3, 192, 192).cuda()
                with torch.no_grad():
                    traced_script_model = torch.jit.trace(self.model.encoder, example_image)
                torch.jit.save(traced_script_model, encoder_ts_path)
                print('Encoder serialized model is saved')
        return losses


def run_train():
    # define image paths
    # image_dirs = ['G:\\Data\\pic_time\\Photos\\Imagesets\\photos\\1',
    #               'G:\\Data\\pic_time\\Photos\\Imagesets\\photos\\2']
    image_dirs = ['G:\\Data\\pic_time\\Photos\\Imagesets\\photos', ]
    # define checkpoint paths
    model_checkpoint_path = 'g:\\Data\\pic_time\\models\\trained_conv_ae\\vgg_ae\\ae_model.pt'
    encoder_path = 'g:\\Data\\pic_time\\models\\trained_conv_ae\\vgg_ae\\encoder.pt'
    encoder_ts_path = 'g:\\Data\\pic_time\\models\\trained_conv_ae\\vgg_ae\\encoder.torchscript'
    # define monitoring path
    monitoring_path = 'f:\\Programming\\Projects\\pic_time\\monitoring\\tb_monitoring'
    # get image paths
    image_paths = []
    for image_dir in image_dirs:
        _, item_image_paths = get_file_names(image_dir)
        image_paths += item_image_paths
    image_paths = sorted(image_paths)

    # run trainer
    start = datetime.now()

    trainer = AutoencoderTrainer(checkpoint_path=model_checkpoint_path, monitoring_path=monitoring_path)
    losses = trainer.train(image_paths[:], batch_size=16, num_epochs=5,
                           model_checkpoint_path=model_checkpoint_path, encoder_path=encoder_path,
                           encoder_ts_path=None)

    print('Run time', datetime.now() - start)
    plt.plot(losses)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    run_train()
