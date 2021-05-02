from collections import OrderedDict
import torch
import torch.nn as nn

from .condseq import CondSeq
from .conv import ConvBlock, Conv3x3
from .interpolate import InterpolateBilinear2d


class ConvDeconvBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 inner: nn.Module) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_channels = hidden_channels
        self.with_skip = False

        self.pre = CondSeq()

        self.downsample = CondSeq(
            OrderedDict([
                ('conv_0',
                 ConvBlock(self.in_channels,
                           self.hidden_channels,
                           4,
                           stride=2)),
            ]))
        self.inner = inner
        assert isinstance(inner.out_channels, int)
        self.upsample = CondSeq(
            OrderedDict([
                ('conv_0',
                 ConvBlock(inner.out_channels,
                           self.in_channels,
                           4,
                           stride=2).to_transposed_conv()),
            ]))

        self.post = CondSeq()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        out = self.upsample(self.inner(self.downsample(x)))
        if self.with_skip:
            out = torch.cat([out, x], dim=1)
        out = self.post(out)
        return out

    def add_skip(self) -> 'ConvDeconvBlock':
        self.with_skip = True
        if self.with_skip:
            self.out_channels = self.in_channels + self.in_channels
        return self

    def leaky(self) -> 'ConvDeconvBlock':
        for block in [self.pre, self.downsample, self.upsample, self.post]:
            for m in block:
                if isinstance(m, ConvBlock):
                    m.leaky()
        return self


class UBlock(nn.Module):
    downsample: nn.Module

    def __init__(self, in_channels: int, hidden_channels: int,
                 inner: nn.Module) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_channels = hidden_channels

        self.set_encoder_num_layers(2)

        self.downsample = CondSeq(nn.MaxPool2d(2))

        self.inner = inner
        assert isinstance(inner.in_channels, int)
        assert isinstance(inner.out_channels, int)

        self.upsample = CondSeq(
            nn.ConvTranspose2d(inner.out_channels,
                               inner.out_channels // 2,
                               kernel_size=2,
                               stride=2))

        self.set_decoder_num_layers(2)

    def to_bilinear_sampling(self) -> 'UBlock':
        from torchelie.transforms.differentiable import BinomialFilter2d
        self.downsample = BinomialFilter2d(2)
        assert isinstance(self.inner.in_channels, int)
        assert isinstance(self.inner.out_channels, int)
        self.upsample = CondSeq(
            InterpolateBilinear2d(scale_factor=2),
            Conv3x3(self.inner.out_channels, self.inner.out_channels // 2),
            nn.ReLU(True))
        return self

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        x_skip = self.in_conv(x_orig)

        x = self.upsample(self.inner(self.downsample(x_skip)))
        x_skip = torch.cat([x, x_skip], dim=1)

        return self.out_conv(x_skip)

    def set_encoder_num_layers(self, num_layers: int) -> 'UBlock':
        layers = CondSeq()
        for i in range(num_layers):
            layers.add_module(
                f'conv_{i}',
                ConvBlock(
                    self.in_channels if i == 0 else self.hidden_channels,
                    self.hidden_channels, 3))
        self.in_conv = layers
        return self

    def set_decoder_num_layers(self, num_layers: int) -> 'UBlock':
        assert isinstance(self.inner.in_channels, int)
        assert isinstance(self.inner.out_channels, int)

        layers = CondSeq()
        for i in range(num_layers):
            in_ch = (self.inner.out_channels // 2
                     + self.hidden_channels if i == 0 else self.hidden_channels)
            out_ch = (self.in_channels if i == (num_layers
                                                - 1) else self.hidden_channels)

            layers.add_module(f'conv_{i}', ConvBlock(in_ch, out_ch, 3))
        self.out_conv = layers
        return self

    def remove_batchnorm(self) -> 'UBlock':
        for m in self.modules():
            if isinstance(m, ConvBlock):
                m.remove_batchnorm()
        return self

    def leaky(self) -> 'UBlock':
        for m in self.modules():
            if isinstance(m, ConvBlock):
                m.leaky()
        return self

    def set_padding_mode(self, mode: str) -> 'UBlock':
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.padding_mode = mode
        return self
