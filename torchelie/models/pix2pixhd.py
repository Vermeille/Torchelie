from typing import List
import copy

import torch

import torchelie.nn as tnn
import torchelie.utils as tu
import torch.nn as nn


class Pix2PixHDGlobalGenerator(tnn.CondSeq):
    """
    Residual generator used in `Pix2PixHD <https://arxiv.org/abs/1711.11585>`_
    .

    :code:`arch` is a list of strings representing blocks.

    For example, this creates a first conv with 32 output channels, 3
    downsampling stride 2 convs that double the number of channels, 5 residual
    blocks, 3 upsampling convs halving the number of channels, and a final conv
    that converts back to RGB.

    :code:```
    Pix2PixHDGlobalGenerator(['32', 'd128', 'd512', 'd512', 'R512', 'R512',
        'R512', 'R512', 'R512', 'u512', 'u512', 'u128'])
    ```

    """

    def __init__(self, arch: List[str]) -> None:
        super().__init__()
        self.arch = arch
        self.to_standard_arch()

    def to_standard_arch(self):
        self._modules.clear()
        arch = self.arch
        self.input = tnn.ConvBlock(3, int(arch[0]), 3)
        ch, i = int(arch[0]), 1

        ii = 0
        self.encode = tnn.CondSeq()
        while arch[i][0] == 'd':
            out_ch = int(arch[i][1:])
            self.encode.add_module(f'conv_{ii}',
                                   tnn.ConvBlock(ch, out_ch, 3, stride=2))
            ch = out_ch
            i += 1
            ii += 1

        ii = 0
        self.transform = tnn.CondSeq()
        while arch[i][0] == 'R':
            out_ch = int(arch[i][1:])
            resblock = tnn.PreactResBlock(ch, out_ch)
            tnn.utils.insert_after(resblock.branch, 'bn1',
                                   tnn.Noise(1, inplace=False), 'noise')
            self.transform.add_module(f'transform_{ii}', resblock)
            ch = out_ch
            i += 1
            ii += 1
        self.transform.add_module('norm', nn.InstanceNorm2d(ch))
        self.transform.add_module('relu', nn.ReLU(True))
        self.transform[0].preact_skip()

        ii = 0
        self.decode = tnn.CondSeq()
        while i < len(arch) and arch[i][0] == 'u':
            out_ch = int(arch[i][1:])
            convblock = tnn.ConvBlock(ch, out_ch, 3, stride=1).add_upsampling()
            self.decode.add_module(f'out_conv_{ii}', convblock)
            tnn.utils.insert_after(convblock, 'norm',
                                   tnn.Noise(1, inplace=True), 'noise')
            ch = out_ch
            i += 1
            ii += 1
        self.to_rgb = tnn.ConvBlock(out_ch, 3, 3).remove_batchnorm()
        self.to_rgb.relu = nn.Sigmoid()

        def to_instance_norm(m):
            if isinstance(m, nn.BatchNorm2d):
                return nn.InstanceNorm2d(m.num_features, affine=True)
            if isinstance(m, nn.Conv2d):
                m.padding_mode = 'reflect'
            return m

        tnn.utils.edit_model(self, to_instance_norm)

    def leaky(self) -> 'Pix2PixHDGlobalGenerator':
        tnn.utils.make_leaky(self)
        return self

    def to_unet(self) -> 'Pix2PixHDGlobalGenerator':
        self._modules.clear()
        arch = self.arch

        ch = int(self.arch[0])

        self.input = tnn.ConvBlock(33, int(arch[0]), 7)

        def _build(i, prev_ch):
            ch = int(arch[i][1:])

            if arch[i][0] == 'R':
                transforms = tnn.CondSeq()
                transforms.in_channels = prev_ch
                ii = 0
                while arch[i][0] == 'R':
                    ch = int(arch[i][1:])
                    transforms.add_module(f'transform_{ii}',
                                          tnn.PreactResBlock(prev_ch, ch))
                    prev_ch = ch
                    i += 1
                    ii += 1
                transforms.out_channels = ch

                transform.add_module('norm', nn.InstanceNorm2d(ch))
                transform.add_module('relu', nn.ReLU(True))
                transform[0].preact_skip()
                return transforms
            if arch[i][0] == 'd':
                u = tnn.encdec.UBlock(prev_ch, ch, _build(i + 1, ch))
                u.to_bilinear_sampling()
                u.set_decoder_num_layers(1).set_encoder_num_layers(1)
                return u

        self.encdec = _build(1, ch)
        self.to_rgb = tnn.ConvBlock(ch, 3, 3).remove_batchnorm()
        self.to_rgb.relu = nn.Sigmoid()

        def to_instance_norm(m):
            if isinstance(m, nn.BatchNorm2d):
                return nn.InstanceNorm2d(m.num_features, affine=True)
            if isinstance(m, nn.Conv2d):
                m.padding_mode = 'reflect'
            return m

        tnn.utils.edit_model(self, to_instance_norm)

        return self

    def to_equal_lr(self, leak=0.) -> 'Pix2PixHDGlobalGenerator':
        tnn.utils.net_to_equal_lr(self, leak=leak)
        for m in self.modules():
            if isinstance(m, tnn.PreactResBlock):
                pass
        return self


@tu.experimental
def pix2pixhd() -> Pix2PixHDGlobalGenerator:
    return Pix2PixHDGlobalGenerator(['64', 'd128', 'd256', 'd512', 'd1024'] +
                                    ['R1024'] * 10 +
                                    ['u1024', 'u512', 'u256', 'u128'])


def pix2pixhd_dev() -> Pix2PixHDGlobalGenerator:
    return Pix2PixHDGlobalGenerator(['64', 'd128', 'd256', 'd512'] +
                                    ['R512'] * 10 + ['u512', 'u256', 'u128'])


@tu.experimental
def pix2pixhd_res_dev() -> Pix2PixHDGlobalGenerator:
    return Pix2PixHDGlobalGenerator(['8', 'd16', 'd32', 'd64', 'd128', 'd256'] +
                                    ['R256'] * 10 +
                                    ['u256', 'u128', 'u64', 'u32', 'u16'])


class MultiScaleDiscriminator(nn.Module):

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.scale_1 = base_model
        self.scale_2 = copy.deepcopy(base_model)
        self.scale_4 = copy.deepcopy(base_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        return torch.cat([
            self.scale_1(x).view(N, -1),
            self.scale_2(
                nn.functional.interpolate(x, scale_factor=0.5,
                                          mode='bilinear')).view(N, -1),
            self.scale_4(
                nn.functional.interpolate(x, scale_factor=0.25,
                                          mode='bilinear')).view(N, -1),
        ],
                         dim=1)
