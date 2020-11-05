from collections import OrderedDict

import torchelie as tch
import torchelie.utils as tu
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchelie.transforms.differentiable import center_crop


class UBlock(nn.Module):
    def __init__(self, in_ch, out_ch, inner=None):
        """
        Initialize k - layer.

        Args:
            self: (todo): write your description
            in_ch: (int): write your description
            out_ch: (str): write your description
            inner: (todo): write your description
        """
        super(UBlock, self).__init__()
        self.in_conv = nn.Sequential(
            OrderedDict([
                ('pad1', nn.ReflectionPad2d(1)),
                ('conv1', tu.kaiming(nn.Conv2d(in_ch, out_ch, 3))),
                ('relu1', nn.ReLU(inplace=True)),
                ('pad2', nn.ReflectionPad2d(1)),
                ('conv2', tu.kaiming(nn.Conv2d(out_ch, out_ch, 3))),
                ('relu2', nn.ReLU(inplace=True)),
            ]))

        self.inner = inner
        if inner is not None:
            self.inner = nn.Sequential(
                    nn.MaxPool2d(2, 2),
                    inner,
                    nn.UpsamplingNearest2d(scale_factor=2),
                    nn.ReflectionPad2d(1),
                    tu.kaiming(nn.Conv2d(out_ch, out_ch, 3)),
                )
            self.skip = nn.Sequential(
                    tu.kaiming(nn.Conv2d(out_ch, out_ch, 1)))

        inner_ch = out_ch * (1 if inner is None else 2)
        self.out_conv = nn.Sequential(
            OrderedDict([
                ('pad1', nn.ReflectionPad2d(1)),
                ('conv1', tu.kaiming(nn.Conv2d(inner_ch, out_ch, 3))),
                ('relu1', nn.ReLU(inplace=True)),
                ('pad2', nn.ReflectionPad2d(1)),
                ('conv2', tu.kaiming(nn.Conv2d(out_ch, in_ch, 3))),
                ('relu2', nn.ReLU(inplace=True)),
            ]))

    def forward(self, x_orig):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x_orig: (todo): write your description
        """
        x = self.in_conv(x_orig)
        if self.inner is not None:
            x2 = x
            x = self.inner(x)
            x = torch.cat([
                x,
                self.skip(center_crop(x2, x.shape[2:]))
            ], dim=1)

        return self.out_conv(x)



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
        """
        Initialize k - layer.

        Args:
            self: (todo): write your description
            arch: (todo): write your description
            in_ch: (int): write your description
            out_ch: (str): write your description
        """
        super(UNetBone, self).__init__()
        self.in_conv = nn.Sequential(
                tu.kaiming(nn.Conv2d(in_ch, arch[0], 5, padding=2)),
                nn.ReLU(True)
            )
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            tu.kaiming(nn.Conv2d(arch[-1], arch[-1], 3)),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            tu.kaiming(nn.Conv2d(arch[-1], arch[-1], 3)),
            nn.ReLU(True),
        )
        for x1, x2 in zip(reversed(arch[:-1]), reversed(arch[1:])):
            self.conv = UBlock(x1, x2, self.conv)
        self.out_conv = nn.Sequential(
                tu.kaiming(nn.Conv2d(arch[0], out_ch, 1)),
            )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (tensor): input tensor, batch of images
        """
        out = self.out_conv(self.conv(self.in_conv(x)))
        return out


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
    return UNetBone([32, 64, 128, 256, 512, 512], in_ch=in_ch, out_ch=out_ch)


if __name__ == '__main__':
    unet = UNet()
    print(unet)
    print(unet(torch.randn(1, 3, 132, 132)).shape)
