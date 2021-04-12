import torch
import torch.nn as nn
import torchelie.nn as tnn
import torchelie.utils as tu
from typing import List, Union, Tuple, cast, Optional
from collections import OrderedDict
from .classifier import ClassificationHead, ProjectionDiscr


class ResidualDiscriminator(nn.Module):
    classifier: Union[ClassificationHead, ProjectionDiscr]

    def __init__(self, arch: List[Union[str, int]]) -> None:
        super().__init__()
        self.arch = arch
        self.in_channels = 3
        in_ch = arch[0]

        features = tnn.CondSeq()
        assert isinstance(in_ch, int)
        features.add_module('input', tnn.Conv3x3(3, in_ch))

        for i, (x, x2) in enumerate(zip(arch, arch[1:] + ['dummy'])):
            if x == 'D':
                continue

            downsample = x2 == 'D'
            assert isinstance(x, int)
            features.add_module(f'block_{i}',
                                tnn.ResidualDiscrBlock(in_ch, x, downsample))
            in_ch = x
        self.out_channels = in_ch
        features.add_module('final_relu', nn.LeakyReLU(0.2, True))
        assert isinstance(features.block_0, tnn.ResidualDiscrBlock)
        features.block_0.preact_skip()
        self.features = features

        self.classifier = ClassificationHead(self.out_channels, 1)

    def forward(self,
                x: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.features(x)
        if isinstance(self.classifier, ProjectionDiscr):
            return self.classifier(x, y)
        else:
            return self.classifier(x)

    def set_input_specs(self, in_channels: int) -> 'ResidualDiscriminator':
        self.input = tnn.Conv3x3(in_channels, self.in_channels)
        return self

    def to_spectral_norm(self) -> 'ResidualDiscriminator':
        for m in self.modules():
            if isinstance(m, tnn.ResidualDiscrBlock):
                m.to_spectral_norm()
            elif isinstance(m, nn.Linear):
                nn.utils.spectral_norm(m)

        assert isinstance(self.features.input, nn.Module)
        nn.utils.spectral_norm(self.features.input)
        if hasattr(self.features, 'mbconv'):
            assert isinstance(self.features.mbconv, tnn.Conv2dBNReLU)
            assert isinstance(self.features.mbconv.conv, nn.Module)
            nn.utils.spectral_norm(self.features.mbconv.conv)
        if isinstance(self.classifier, ProjectionDiscr):
            self.classifier.to_spectral_norm()
        return self

    def to_equal_lr(self) -> 'ResidualDiscriminator':
        for m in self.modules():
            if isinstance(m, tnn.ResidualDiscrBlock):
                m.to_equal_lr()
            elif isinstance(m, nn.Linear):
                tu.kaiming(m,
                           dynamic=True,
                           nonlinearity='leaky_relu',
                           a=0.2,
                           mode='fan_in')

        assert isinstance(self.features.input, nn.Module)
        tu.kaiming(self.features.input,
                   dynamic=True,
                   nonlinearity='leaky_relu',
                   a=0.2,
                   mode='fan_in')
        if hasattr(self.features, 'mbconv'):
            assert isinstance(self.features.mbconv, tnn.Conv2dBNReLU)
            assert isinstance(self.features.mbconv.conv, nn.Module)
            tu.kaiming(self.features.mbconv.conv,
                       dynamic=True,
                       nonlinearity='leaky_relu',
                       a=0.2,
                       mode='fan_in')
        if isinstance(self.classifier, ProjectionDiscr):
            self.classifier.to_equal_lr()
        return self

    def add_minibatch_stddev(self) -> 'ResidualDiscriminator':
        out_ch = self.out_channels
        self.features.add_module('mbstd', tnn.MinibatchStddev())
        self.features.add_module(
            'mbconv',
            tnn.Conv2dBNReLU(out_ch + 1, out_ch, 3).remove_bn().leaky())
        return self

    def to_projection_discr(self, num_classes: int) -> 'ResidualDiscriminator':
        self.classifier = ProjectionDiscr(self.out_channels, num_classes)
        return self


def res_discr_4l() -> ResidualDiscriminator:
    return ResidualDiscriminator([32, 'D', 64, 'D', 128, 'D', 256, 'D'])


def snres_discr_4l() -> ResidualDiscriminator:
    return res_discr_4l().to_spectral_norm()


def snres_projdiscr_4l(num_classes: int) -> ResidualDiscriminator:
    return res_discr_4l().to_projection_discr(num_classes).to_spectral_norm()


def res_discr_5l() -> ResidualDiscriminator:
    return ResidualDiscriminator(
        [32, 'D', 64, 'D', 128, 'D', 256, 'D', 512, 'D'])


def snres_discr_5l() -> ResidualDiscriminator:
    return res_discr_5l().to_spectral_norm()


def snres_projdiscr_5l(num_classes: int) -> ResidualDiscriminator:
    return res_discr_5l().to_projection_discr(num_classes).to_spectral_norm()


def res_discr_6l() -> ResidualDiscriminator:
    return ResidualDiscriminator(
        [32, 'D', 64, 'D', 128, 'D', 256, 'D', 512, 'D', 512, 'D'])


def snres_discr_6l() -> ResidualDiscriminator:
    return res_discr_6l().to_spectral_norm()


def snres_projdiscr_6l(num_classes: int) -> ResidualDiscriminator:
    return res_discr_6l().to_projection_discr(num_classes).to_spectral_norm()


def res_discr_7l() -> ResidualDiscriminator:
    return ResidualDiscriminator(
        [32, 'D', 64, 'D', 128, 'D', 256, 'D', 512, 'D', 512, 'D', 512, 'D'])


def snres_discr_7l() -> ResidualDiscriminator:
    return res_discr_7l().to_spectral_norm()


def snres_projdiscr_7l(num_classes: int) -> ResidualDiscriminator:
    return res_discr_7l().to_projection_discr(num_classes).to_spectral_norm()


def stylegan2_discr(input_sz,
                    max_ch: int = 512,
                    ch_mul: int = 1) -> ResidualDiscriminator:
    """
        Build the discriminator for StyleGAN2

        Args:
            input_sz (int): image size
            max_ch (int): maximum number of channels (default: 512)
            ch_mul (float): multiply the number of channels on each layer by this
                value (default, 1.)
            equal_lr (bool): equalize the learning rates with dynamic weight
                scaling
    """
    import math
    res = input_sz
    ch = int(512 / (2**(math.log2(res) - 6)) * ch_mul)
    layers: List[Union[str, int]] = [min(max_ch, ch)]

    while res > 4:
        res = res // 2
        layers.append(min(max_ch, ch * 2))
        layers.append('D')
        ch *= 2

    net = ResidualDiscriminator(layers)
    net.add_minibatch_stddev()
    net.classifier.set_pool_size(4)
    net.to_equal_lr()
    return net
