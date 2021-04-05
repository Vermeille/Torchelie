from collections import OrderedDict

import torchelie as tch
import torchelie.utils as tu
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchelie.transforms.differentiable import center_crop


class UBlock(nn.Module):
    def __init__(self, in_ch, out_ch, inner=None, skip=True, up_mode='bilinear'):
        super(UBlock, self).__init__()
        self.up_mode = up_mode
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
            if skip:
                self.skip = nn.Sequential(
                    tu.kaiming(nn.Conv2d(out_ch, out_ch, 1)),
                    nn.ReLU(True))
            else:
                self.skip = nn.Identity()

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
        x = self.in_conv(x_orig)
        if self.inner is not None:
            x2 = x
            x = F.interpolate(
                    self.inner(
                        F.avg_pool2d(x, 3, 2, 1)
                    ),
                mode=self.up_mode, size=x.shape[2:]
                )
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

    def __init__(self, arch, in_ch=3, out_ch=1, *, skip=True,
        up_mode='bilinear'):
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
            self.conv = UBlock(x1, x2, self.conv, skip=skip, up_mode=up_mode)
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
