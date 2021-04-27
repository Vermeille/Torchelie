from collections import OrderedDict
from typing import Optional, List, Tuple, cast

import torchelie as tch
import torchelie.nn as tnn
import torchelie.utils as tu
import torch
import torch.nn as nn


class UNet(nn.Module):
    @tu.experimental
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
                                                3).remove_batchnorm()

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
        self.features.input.remove_batchnorm()
        return self

    def remove_batchnorm(self) -> 'UNet':
        for m in self.modules():
            if isinstance(m, tnn.Conv2dBNReLU):
                m.remove_batchnorm()
        return self


class Pix2PixGenerator(UNet):
    @tu.experimental
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
        tnn.utils.insert_after(layers[-1].inner, 'norm', tnn.Noise(1, True),
                               'noise')

        for m in layers:
            tnn.utils.insert_after(
                m.out_conv.conv_0, 'norm',
                tnn.Noise(m.out_conv.conv_0.out_channels, True), 'noise')


class Pix2PixResidualGenerator(tnn.CondSeq):
    @tu.experimental
    def __init__(self, arch: List[str])->None:
        super().__init__()
        self.arch = arch
        self.input = tnn.CondSeq(
            tnn.SinePositionEncoding2d(15),
            tnn.Conv2dBNReLU(33, int(arch[0]), 7),
        )
        ch, i = int(arch[0]), 1

        self.encode = tnn.CondSeq()
        while arch[i][0] == 'd':
            out_ch = int(arch[i][1:])
            self.encode.add_module(f'conv_{i}', tnn.Conv2dBNReLU(ch, out_ch, 3, stride=2))
            ch = out_ch
            i += 1

        self.transform = tnn.CondSeq()
        while arch[i][0] == 'R':
            out_ch = int(arch[i][1:])
            self.encode.add_module(f'transform_{i}', tnn.ResBlock(ch, out_ch))
            ch = out_ch
            i += 1

        self.decode = tnn.CondSeq()
        while i < len(arch) and arch[i][0] == 'u':
            out_ch = int(arch[i][1:])
            self.encode.add_module(f'out_conv_{i}', tnn.Conv2dBNReLU(ch, out_ch, 3,
                stride=1).add_upsampling())
            ch = out_ch
            i += 1
        self.to_rgb = tnn.Conv2dBNReLU(out_ch, 3, 7)
        self.to_rgb.relu = nn.Sigmoid()

        def to_instance_norm(m):
            if isinstance(m, nn.BatchNorm2d):
                return nn.InstanceNorm2d(m.num_features, affine=True)
            if isinstance(m, nn.Conv2d):
                m.padding_mode = 'reflect'
            return m
        tnn.utils.edit_model(self, to_instance_norm)


def pix2pix_256() -> Pix2PixGenerator:
    return Pix2PixGenerator([64, 128, 256, 512, 512, 512, 512])


def pix2pix_res_dev() -> Pix2PixResidualGenerator:
    return Pix2PixResidualGenerator([32, 'd128', 'd512', 'd512', 'R512', 'R512',
        'R512', 'R512', 'R512', 'u512', 'u512', 'u128'])

def pix2pix_dev() -> Pix2PixGenerator:
    return Pix2PixGenerator([32, 64, 128, 128, 256, 256, 512])
