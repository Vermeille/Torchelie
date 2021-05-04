from typing import List, Tuple, cast

import torchelie.nn as tnn
import torchelie.utils as tu
import torch.nn as nn


@tu.experimental
class UNet(tnn.CondSeq):
    """
    U-Net from `U-Net: Convolutional Networks for Biomedical Image Segmentation
    <https://arxiv.org/abs/1505.04597>`_. This net has architectural changes
    operations for further customization.

    Args:
        arch (List[int]): a list of channels from the outermost to innermost
            layers
        num_classes (int): number of output channels
    """

    def __init__(self, arch: List[int], num_classes: int) -> None:
        super().__init__()
        self.arch = arch
        self.in_channels = 3
        self.out_channels = arch[-1]

        feats = tnn.CondSeq()
        feats.input = tnn.ConvBlock(3, arch[0], 3)

        encdec: nn.Module = tnn.ConvBlock(arch[-1], arch[-1] * 2, 3)
        for outer, inner in zip(arch[-2::-1], arch[:0:-1]):
            encdec = tnn.UBlock(outer, inner, encdec)
        feats.encoder_decoder = encdec
        self.features = feats
        assert isinstance(encdec.out_channels, int)
        self.classifier = tnn.ConvBlock(encdec.out_channels, num_classes,
                                        3).remove_batchnorm().no_relu()

    def leaky(self, leak: float = 0.2) -> 'UNet':
        for m in self.modules():
            if isinstance(m, tnn.ConvBlock):
                m.leaky(leak)
        return self

    def set_padding_mode(self, mode: str) -> 'UNet':
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.padding_mode = mode
        return self

    def set_input_specs(self, in_channels: int) -> 'UNet':
        assert isinstance(self.features.input, tnn.ConvBlock)
        c = self.features.input.conv
        self.features.input.conv = tu.kaiming(
            nn.Conv2d(in_channels,
                      c.out_channels,
                      cast(Tuple[int, int], c.kernel_size),
                      bias=c.bias is not None,
                      padding=cast(Tuple[int, int], c.padding)))
        return self

    def set_encoder_num_layers(self, num: int) -> 'UNet':
        for m in self.modules():
            if isinstance(m, tnn.UBlock):
                m.set_encoder_num_layers(num)
        return self

    def set_decoder_num_layers(self, num: int) -> 'UNet':
        for m in self.modules():
            if isinstance(m, tnn.UBlock):
                m.set_decoder_num_layers(num)
        return self

    def to_bilinear_sampling(self) -> 'UNet':
        for m in self.modules():
            if isinstance(m, tnn.UBlock):
                m.to_bilinear_sampling()
        return self

    def remove_first_batchnorm(self) -> 'UNet':
        assert isinstance(self.features.input, tnn.ConvBlock)
        self.features.input.remove_batchnorm()
        return self

    def remove_batchnorm(self) -> 'UNet':
        for m in self.modules():
            if isinstance(m, tnn.ConvBlock):
                m.remove_batchnorm()
        return self
