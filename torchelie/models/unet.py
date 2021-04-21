from collections import OrderedDict
from typing import Optional, List, Tuple, cast

import torchelie as tch
import torchelie.nn as tnn
import torchelie.utils as tu
import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, arch: List[int]) -> None:
        super().__init__()
        self.arch = arch
        self.in_channels = 3
        self.out_channels = arch[-1]

        feats = tnn.CondSeq()
        feats.input = tnn.Conv2dBNReLU(3, arch[0], 3)

        encdec: nn.Module = tnn.Conv2dBNReLU(arch[-1], arch[-1] * 2, 3)
        for outer, inner in zip(arch[-2::-1], arch[:0:-1]):
            encdec = tnn.UBlock(outer, inner, encdec)
        feats.encoder_decoder = encdec
        self.features = feats
        self.classifier = tnn.CondSeq()
        assert isinstance(encdec.out_channels, int)
        self.classifier.conv = tnn.Conv2dBNReLU(encdec.out_channels, 3,
                                                3).remove_bn()

    def forward(self, x):
        return self.classifier(self.features(x))

    def set_input_specs(self, in_channels: int) -> 'UNet':
        assert isinstance(self.features.input, tnn.Conv2dBNReLU)
        c = self.features.input.conv
        self.features.input.conv = tu.kaiming(
            nn.Conv2d(in_channels,
                      c.out_channels,
                      cast(Tuple[int, int], c.kernel_size),
                      bias=c.bias is not None,
                      padding=cast(Tuple[int, int], c.padding)))
        return self

    def remove_first_batchnorm(self) -> 'UNet':
        assert isinstance(self.features.input, tnn.Conv2dBNReLU)
        self.features.input.remove_bn()
        return self

    def remove_batchnorm(self) -> 'UNet':
        for m in self.modules():
            if isinstance(m, tnn.Conv2dBNReLU):
                m.remove_bn()
        return self


class Pix2PixGenerator(UNet):
    def __init__(self, arch: List[int]) -> None:
        super().__init__(arch)
        self.remove_first_batchnorm()

        encdec = cast(nn.Module, self.features.encoder_decoder)
        for m in encdec.modules():
            if isinstance(m, tnn.UBlock):
                m.to_bilinear_sampling()
                m.set_encoder_num_layers(1)

        tnn.utils.make_leaky(self)
        self.classifier.conv.relu = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.padding_mode = 'reflect'

        self._add_noise()


    def _add_noise(self):
        layers = [self.features.encoder_decoder]
        while hasattr(layers[-1].inner, 'inner'):
            layers.append(layers[-1].inner)
        tnn.utils.insert_after(layers[-1].inner, 'norm',
                                   tnn.Noise(1, True),
                                   'noise')

        for m in layers:
            tnn.utils.insert_after(m.out_conv.conv_0, 'norm',
                                   tnn.Noise(1, True),
                                   'noise')


def pix2pix_generator() -> Pix2PixGenerator:
    return Pix2PixGenerator([64, 128, 256, 512, 512, 512, 512])

def pix2pix_dev() -> Pix2PixGenerator:
    return Pix2PixGenerator([32, 64, 128, 128, 256, 256, 512])

