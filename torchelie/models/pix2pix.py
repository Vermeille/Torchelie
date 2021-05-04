import torch
from typing import List, cast

import torchelie.nn as tnn
import torchelie.utils as tu
import torch.nn as nn
from .unet import UNet


class Pix2PixGenerator(UNet):
    """
    UNet generator from Pix2Pix. Dropout layers have been substitued with Noise
    injections from StyleGAN2.

    Args:
        arch (List[int]): the number of channel for each depth level of the
            UNet.
    """

    def __init__(self, arch: List[int]) -> None:
        super().__init__(arch, 3)
        self.remove_first_batchnorm()

        self.features.input = tnn.ConvBlock(3, int(arch[0]), 7)
        self.features.input.remove_batchnorm()

        encdec = cast(nn.Module, self.features.encoder_decoder)
        for m in encdec.modules():
            if isinstance(m, tnn.UBlock):
                m.to_bilinear_sampling()
                m.set_encoder_num_layers(1)
                m.set_decoder_num_layers(1)

        tnn.utils.make_leaky(self)
        self.classifier.relu = nn.Sigmoid()

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

    def to_equal_lr(self) -> 'Pix2PixGenerator':
        return tnn.utils.net_to_equal_lr(self)

    def set_padding_mode(self, mode: str) -> 'Pix2PixGenerator':
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.padding_mode = mode
        return self

    @torch.no_grad()
    def to_instance_norm(self, affine: bool = True) -> 'Pix2PixGenerator':
        """
        Pix2Pix sometimes uses batch size 1, similar to instance norm.
        """

        def to_instancenorm(m):
            if isinstance(m, nn.BatchNorm2d):
                return nn.InstanceNorm2d(m.num_features, affine=affine)
            return m

        tnn.utils.edit_model(self, to_instancenorm)

        return self


def pix2pix_256() -> Pix2PixGenerator:
    """
    The architecture used in `Pix2Pix <https://arxiv.org/abs/1611.07004>`_,
    able to train on 256x256 or 512x512 images.
    """
    return Pix2PixGenerator([64, 128, 256, 512, 512, 512, 512, 512])


def pix2pix_dev() -> Pix2PixGenerator:
    """
    A version of pix2pix_256 with less filter to use less memory and compute.
    """
    return Pix2PixGenerator([32, 64, 128, 128, 256, 256, 512, 512])
