import torchvision
from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import copy

###
# Original implementation courtesy of:
# Dag Lindgren
# Andreas Wallin
# Lowe Lundin

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

def conv1x1(in_, out):
    return nn.Conv2d(in_, out, 1, padding=0)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class UNet(nn.Module):
    def __init__(self, channel_count=3, class_count=2, pre_trained=True):

        num_filters = 32
        """
        :param num_filters:
        :param pre_trained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with VGG11
        """
        super(UNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.convertconv = conv1x1(in_=channel_count, out=3)

        encoder = models.vgg11(pretrained=pre_trained).features

        self.relu = encoder[1]
        self.conv1 = encoder[0]
        self.conv2 = encoder[3]
        self.conv3s = encoder[6]
        self.conv3 = encoder[8]
        self.conv4s = encoder[11]
        self.conv4 = encoder[13]
        self.conv5s = encoder[16]
        self.conv5 = encoder[18]

        self.center = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec3 = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final = nn.Conv2d(num_filters, class_count, kernel_size=1)

        self.downsample2 = nn.Conv2d(in_channels = num_filters * 2, out_channels = num_filters * 2 * 2, kernel_size =1)
        self.downsample3 = nn.Conv2d(in_channels = num_filters * 2 * 2, out_channels = num_filters * 2 * 4, kernel_size = 1)
        self.downsample4 = nn.Conv2d(in_channels = num_filters * 2 * 4, out_channels = num_filters * 2 * 8, kernel_size = 1)
        self.downsample5 = nn.Conv2d(in_channels = num_filters * 2 * 8, out_channels = num_filters * 2 * 8, kernel_size =1)

    def forward(self, x):

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[1] == 2:
            x = torch.cat((x, x[:, :1, :, :]), 1)
            print(x.size())
        if x.shape[1] > 3: 
            x = self.convertconv(x)

        conv1 = self.relu(self.conv1(x))
        conv1p = self.pool(conv1)

        conv2 = self.relu(self.conv2(conv1p) + self.downsample2(conv1p))
        conv2p = self.pool(conv2)

        conv3s = self.relu(self.conv3s(conv2p))
        conv3 = self.relu(self.conv3(conv3s) + self.downsample3(conv2p))
        conv3p = self.pool(conv3)

        conv4s = self.relu(self.conv4s(conv3p))
        conv4 = self.relu(self.conv4(conv4s) + self.downsample4(conv3p))
        conv4p = self.pool(conv4)

        conv5s = self.relu(self.conv5s(conv4p))
        conv5 = self.relu(self.conv5(conv5s) + self.downsample5(conv4p))
        conv5p = self.pool(conv5)

        center = self.center(conv5p)

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)
