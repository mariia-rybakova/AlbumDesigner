import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Autoencoder(nn.Module):
    """
    the autoencoder class
    """
    def __init__(self, encoder, decoder, lambda_1, lambda_2, lambda_tv):
        super(Autoencoder, self).__init__()
        # encoder
        self.encoder = encoder

        # decoder
        self.decoder = decoder

        # loss
        self.l2_loss = L2Loss()

        # loss balancing
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_tv = lambda_tv

    def forward(self, input):
        # encode input
        input_encoded = self.encoder(input)['fc']

        # get output
        output = self.decoder(input_encoded)

        # encode output
        output_encoded = self.encoder(output)['fc']

        # MSELoss(self, input, target) => input_features are the target
        loss, feature_loss, per_pixel_loss = self.calculate_loss(output, input, output_encoded, input_encoded)

        return output, loss, feature_loss, per_pixel_loss

    def calculate_loss(self, input, target, input_features, target_features):
        """
        calculates the network loss (feature loss and per-pixel loss and TV loss)
        @param input: output image of the network
        @param target: original input image of the network
        @param input_features: encoding of image @param input
        @param target_features: encoding of image @param target
        @return:
        """
        # feature loss on relu_4
        content_feature_loss = self.l2_loss(input_features.to(device), target_features.to(device))

        # per pixel loss on the images
        per_pixel_loss = self.l2_loss(input, target).to(device)

        # tv regularizer
        tv_regularizer = self.tv_regularizer(input)

        # loss is sum of losses
        loss = self.lambda_tv * tv_regularizer + \
               self.lambda_1 * content_feature_loss + \
               self.lambda_2 * per_pixel_loss

        return loss.to(device), content_feature_loss, per_pixel_loss

    def tv_regularizer(self, input, beta=2.):
        """
        a total variational regularizer (reduces high frequency structures)
        @param input:
        @param beta:
        @return:
        """
        dy = torch.zeros(input.size())
        dx = torch.zeros(input.size())
        dy[:, 1:, :] = -input[:, :-1, :] + input[:, 1:, :]
        dx[:, :, 1:] = -input[:, :, :-1] + input[:, :, 1:]
        return torch.sum((dx.pow(2) + dy.pow(2)).pow(beta / 2.))


class Decoder(nn.Module):
    """
    the decoder network
    """
    def __init__(self):
        super(Decoder, self).__init__()
        # fc block
        self.conv_fc = nn.ConvTranspose2d(2048, 512, 9, 1, 0)
        self.relu_fc = nn.ReLU(inplace=True)

        self.unpool_fc = nn.UpsamplingNearest2d(scale_factor=2)

        # first block
        self.reflecPad_1_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_1_1 = nn.Conv2d(512, 256, 3, 1, 0)
        self.relu_1_1 = nn.ReLU(inplace=True)

        self.unpool_1 = nn.UpsamplingNearest2d(scale_factor=2)

        # second block
        self.reflecPad_2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_1 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_2_1 = nn.ReLU(inplace=True)

        self.reflecPad_2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_2_2 = nn.ReLU(inplace=True)

        self.reflecPad_2_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_2_3 = nn.ReLU(inplace=True)

        self.reflecPad_2_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_4 = nn.Conv2d(256, 128, 3, 1, 0)
        self.relu_2_4 = nn.ReLU(inplace=True)

        self.unpool_2 = nn.UpsamplingNearest2d(scale_factor=2)

        # third block
        self.reflecPad_3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_1 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu_3_1 = nn.ReLU(inplace=True)

        self.reflecPad_3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_2 = nn.Conv2d(128, 64, 3, 1, 0)
        self.relu_3_2 = nn.ReLU(inplace=True)

        self.unpool_3 = nn.UpsamplingNearest2d(scale_factor=2)

        # fourth block
        self.reflecPad_4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_4_1 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu_4_1 = nn.ReLU(inplace=True)

        self.reflecPad_4_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_4_2 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, input):
        # fc block
        out = self.conv_fc(input)
        out = self.relu_fc(out)
        out = self.unpool_fc(out)

        # first block
        out = self.reflecPad_1_1(out)
        out = self.conv_1_1(out)
        out = self.relu_1_1(out)
        out = self.unpool_1(out)

        # second block
        out = self.reflecPad_2_1(out)
        out = self.conv_2_1(out)
        out = self.relu_2_1(out)
        out = self.reflecPad_2_2(out)
        out = self.conv_2_2(out)
        out = self.relu_2_2(out)
        out = self.reflecPad_2_3(out)
        out = self.conv_2_3(out)
        out = self.relu_2_3(out)
        out = self.reflecPad_2_4(out)
        out = self.conv_2_4(out)
        out = self.relu_2_4(out)
        out = self.unpool_2(out)

        # third block
        out = self.reflecPad_3_1(out)
        out = self.conv_3_1(out)
        out = self.relu_3_1(out)
        out = self.reflecPad_3_2(out)
        out = self.conv_3_2(out)
        out = self.relu_3_2(out)
        out = self.unpool_3(out)

        # fourth block
        out = self.reflecPad_4_1(out)
        out = self.conv_4_1(out)
        out = self.relu_4_1(out)
        out = self.reflecPad_4_2(out)
        out = self.conv_4_2(out)

        return out


class Encoder(nn.Module):
    """
    the encoder network
    """
    def __init__(self):
        super(Encoder, self).__init__()
        # first block
        self.conv_1_1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.reflecPad_1_1 = nn.ReflectionPad2d((1, 1, 1, 1))

        self.conv_1_2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.relu_1_2 = nn.ReLU(inplace=True)

        self.reflecPad_1_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_1_3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu_1_3 = nn.ReLU(inplace=True)

        self.maxPool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # second block
        self.reflecPad_2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu_2_1 = nn.ReLU(inplace=True)

        self.reflecPad_2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu_2_2 = nn.ReLU(inplace=True)

        self.maxPool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # third block
        self.reflecPad_3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.relu_3_1 = nn.ReLU(inplace=True)

        self.reflecPad_3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_3_2 = nn.ReLU(inplace=True)

        self.reflecPad_3_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_3_3 = nn.ReLU(inplace=True)

        self.reflecPad_3_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_3_4 = nn.ReLU(inplace=True)

        self.maxPool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # fourth block
        self.reflecPad_4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_4_1 = nn.Conv2d(256, 512, 3, 1, 0)
        self.relu_4_1 = nn.ReLU(inplace=True)

        self.maxPool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # fc block
        self.conv_fc = nn.Conv2d(512, 2048, 9, 1, 0)
        self.relu_fc = nn.ReLU(inplace=True)

    def forward(self, input):
        output = {}

        # first block
        out = self.conv_1_1(input)
        out = self.reflecPad_1_1(out)
        out = self.conv_1_2(out)
        out = self.relu_1_2(out)

        output['r11'] = out

        out = self.reflecPad_1_3(out)
        out = self.conv_1_3(out)
        out = self.relu_1_3(out)

        out = self.maxPool_1(out)

        # second block
        out = self.reflecPad_2_1(out)
        out = self.conv_2_1(out)
        out = self.relu_2_1(out)

        output['r21'] = out

        out = self.reflecPad_2_2(out)
        out = self.conv_2_2(out)
        out = self.relu_2_2(out)

        out = self.maxPool_2(out)

        # third block
        out = self.reflecPad_3_1(out)
        out = self.conv_3_1(out)
        out = self.relu_3_1(out)

        output['r31'] = out

        out = self.reflecPad_3_2(out)
        out = self.conv_3_2(out)
        out = self.relu_3_2(out)

        out = self.reflecPad_3_3(out)
        out = self.conv_3_3(out)
        out = self.relu_3_3(out)

        out = self.reflecPad_3_4(out)
        out = self.conv_3_4(out)
        out = self.relu_3_4(out)

        out = self.maxPool_3(out)

        # fourth block
        out = self.reflecPad_4_1(out)
        out = self.conv_4_1(out)
        out = self.relu_4_1(out)

        output['r41'] = out

        out = self.maxPool_4(out)

        # fc block
        out = self.conv_fc(out)
        out = self.relu_fc(out)

        output['fc'] = out

        return output


class L2Loss(nn.Module):
    """
    simple L_2-Loss
    """
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, input, target):
        return torch.sqrt(torch.mean(torch.abs(input - target).pow(2)))
