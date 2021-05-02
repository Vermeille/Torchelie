import torch
import torch.nn as nn
import torchelie.nn as tnn
import torchelie.utils as tu
from typing import List, Union, Optional
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

        ii = 0
        for i, (x, x2) in enumerate(zip(arch, arch[1:] + ['dummy'])):
            if x == 'D':
                continue

            downsample = x2 == 'D'
            assert isinstance(x, int)
            features.add_module(f'block_{ii}',
                                tnn.ResidualDiscrBlock(in_ch, x, downsample))
            in_ch = x
            ii += 1
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
        self.features.input = tnn.Conv3x3(in_channels,
                                          self.features.input.out_channels)
        return self

    @tu.experimental
    def to_spectral_norm(self) -> 'ResidualDiscriminator':
        for m in self.modules():
            if isinstance(m, tnn.ResidualDiscrBlock):
                m.to_spectral_norm()
            elif isinstance(m, nn.Linear):
                nn.utils.spectral_norm(m)

        assert isinstance(self.features.input, nn.Module)
        nn.utils.spectral_norm(self.features.input)
        if hasattr(self.features, 'mbconv'):
            assert isinstance(self.features.mbconv, tnn.ConvBlock)
            assert isinstance(self.features.mbconv.conv, nn.Module)
            nn.utils.spectral_norm(self.features.mbconv.conv)
        if isinstance(self.classifier, ProjectionDiscr):
            self.classifier.to_spectral_norm()
        return self

    @tu.experimental
    def to_equal_lr(self, leak=0.2) -> 'ResidualDiscriminator':
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                tu.kaiming(m, dynamic=True, a=leak)

        return self

    def add_minibatch_stddev(self) -> 'ResidualDiscriminator':
        out_ch = self.out_channels
        self.features.add_module('mbstd', tnn.MinibatchStddev())
        self.features.add_module(
            'mbconv',
            tnn.ConvBlock(out_ch + 1, out_ch, 3).remove_batchnorm().leaky())
        return self

    def to_projection_discr(self, num_classes: int) -> 'ResidualDiscriminator':
        self.classifier = ProjectionDiscr(self.out_channels, num_classes)
        return self


def residual_patch34():
    D = ResidualDiscriminator([32, 'D', 64, 'D', 128])
    D.classifier.to_convolutional().leaky()
    return D


def residual_patch70():
    D = ResidualDiscriminator([32, 'D', 64, 'D', 128, 'D', 256])
    D.classifier.to_convolutional().leaky()
    return D


def residual_patch142():
    D = ResidualDiscriminator([32, 'D', 64, 'D', 128, 'D', 256, 'D', 512])
    D.classifier.to_convolutional().leaky()
    return D


def residual_patch286():
    D = ResidualDiscriminator(
        [32, 'D', 64, 'D', 128, 'D', 256, 'D', 512, 'D', 512])
    D.classifier.to_convolutional().leaky()
    return D


def res_discr_3l() -> ResidualDiscriminator:
    return ResidualDiscriminator([32, 'D', 64, 'D', 128, 'D'])


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
