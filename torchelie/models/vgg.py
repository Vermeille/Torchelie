import torch
import torch.nn as nn
import torchelie.nn as tnn
from torchelie.utils import kaiming
from typing import cast
from .classifier import ClassificationHead


class VGG(nn.Module):
    def __init__(self, arch: list, num_classes: int) -> None:
        super().__init__()
        self.arch = arch
        in_ch = 3
        self.in_channels = in_ch

        feats = tnn.CondSeq()
        block_num = 1
        conv_num = 1
        for layer in arch:
            if layer == 'M':
                feats.add_module(f'pool_{block_num}', nn.MaxPool2d(2, 2))
                block_num += 1
                conv_num = 1
            else:
                ch = cast(int, layer)
                feats.add_module(
                    f'conv_{block_num}_{conv_num}',
                    tnn.ConvBlock(in_ch, ch, 3).remove_batchnorm())
                in_ch = ch
                conv_num += 1
        self.out_channels = ch

        self.features = feats
        self.classifier = ClassificationHead(self.out_channels, num_classes)
        self.classifier.to_vgg_style(4096)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def add_bn(self) -> 'VGG':
        for m in self.features:
            if isinstance(m, tnn.ConvBlock):
                m.restore_bn()
        return self

    def set_input_specs(self, in_channels: int) -> 'VGG':
        c1 = self.features.conv_1_1
        assert isinstance(c1, tnn.ConvBlock)
        c1.conv = kaiming(tnn.Conv3x3(in_channels, c1.conv.out_channels, 3))
        return self


def vgg11(num_classes: int) -> 'VGG':
    return VGG(
        [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        num_classes)


def vgg13(num_classes: int) -> 'VGG':
    return VGG([
        64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'
    ], num_classes)


def vgg16(num_classes: int) -> 'VGG':
    return VGG([
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ], num_classes)


def vgg19(num_classes: int) -> 'VGG':
    return VGG([
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,
        512, 'M', 512, 512, 512, 512, 'M'
    ], num_classes)


def vgg11_bn(num_classes: int) -> 'VGG':
    return vgg11(num_classes).add_bn()


def vgg13_bn(num_classes: int) -> 'VGG':
    return vgg13(num_classes).add_bn()


def vgg16_bn(num_classes: int) -> 'VGG':
    return vgg16(num_classes).add_bn()


def vgg19_bn(num_classes: int) -> 'VGG':
    return vgg19(num_classes).add_bn()
