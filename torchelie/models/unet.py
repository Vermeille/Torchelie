from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchelie.transforms.differentiable import center_crop


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 2)

    def forward(self, x):
        y_sz = x.shape[2] * 2 + 1
        x_sz = x.shape[3] * 2 + 1
        return self.conv(F.interpolate(x, size=(y_sz, x_sz)))


class UNetBone(nn.Module):
    """
    Configurable UNet model.

    Note: Not all input sizes are valid. Make sure that the model can decode an
    image of the same size first.

    Args:
        arch (list): an architecture specification made of:
            - an int, for an kernel with specified output_channels
            - 'U' for upsampling+conv
            - 'D' for downsampling (maxpooling)
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, arch, in_ch=3, out_ch=1):
        super(UNetBone, self).__init__()
        self.arch = arch
        layers = []
        img_in_ch = in_ch
        for x in arch:
            if isinstance(x, int):
                layers.append(self.double_conv(in_ch, x))
                in_ch = x
            elif x == 'D':
                layers.append(nn.MaxPool2d(2, 2))
            elif x == 'U':
                layers.append(UpConv(in_ch, in_ch // 2))
            else:
                assert False, 'Invalid arch spec ' + str(x)
        layers.append(nn.Conv2d(arch[-1], out_ch, 1))
        self.convs = nn.ModuleList(layers)
        self.pad = 0
        pad = 512 - self.forward(torch.randn(1, img_in_ch, 512, 512)).shape[2]
        self.pad = pad // 2

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (tensor): input tensor, batch of images
        """
        x = F.pad(x, [self.pad] * 4, mode='reflect')
        maps = []
        for f in self.convs:
            if isinstance(f, nn.MaxPool2d):
                maps.append(x)
            x = f(x)
            print(x.shape)
            if isinstance(f, UpConv):
                prev = maps.pop()
                prev = center_crop(prev, (x.shape[2], x.shape[3]))
                x = torch.cat([x, prev], dim=1)
        return x

    def double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(in_ch, out_ch, 3)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(out_ch, out_ch, 3)),
                ('relu2', nn.ReLU(inplace=True)),
            ]))


def UNet(in_ch=3, out_ch=1):
    """
    Instantiate the UNet network specified in _U-Net: Convolutional Networks
    for Biomedical Image Segmentation_ (Ronneberger, 2015)

    Valid input sizes include : 572x572, 132x132

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels

    Returns:
        An instantiated UNet
    """
    return UNetBone([
        64, 'D', 128, 'D', 256, 'D', 512, 'D', 1024, 'U', 512, 'U', 256, 'U',
        128, 'U', 64
    ],
                    in_ch=in_ch,
                    out_ch=out_ch)


if __name__ == '__main__':
    unet = UNet()
    print(unet)
    print(unet(torch.randn(1, 3, 132, 132)).shape)
