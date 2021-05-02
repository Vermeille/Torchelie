import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from .batchnorm import Spade2d
from .condseq import CondSeq
from .maskedconv import MaskedConv2d
from torchelie.utils import kaiming, xavier
from torchelie.utils import experimental
from .layers import ModulatedConv
from .noise import Noise
from .resblock import PreactResBlock
from torchelie.nn.graph import ModuleGraph
from typing import List, Tuple
from .utils import edit_model, make_leaky
from .utils import remove_weight_scale
from .interpolate import InterpolateBilinear2d
from .encdec import ConvDeconvBlock
from .conv import ConvBlock, Conv2d, Conv3x3, Conv1x1


@experimental
def MConvNormReLU(in_ch: int,
                  out_ch: int,
                  ks: int,
                  norm,
                  center: bool = True) -> CondSeq:
    """
    A packed block with Masked Conv-Norm-ReLU

    Args:
        in_ch (int): input channels
        out_ch (int): output channels
        ks (int): kernel size
        norm (ctor): A normalization layer constructor in the form
            :code:`(num_layers) -> norm layer` or None
        center (bool): whether the masked conv has access to the central pixel
            or not

    Returns:
        A packed block with MaskedConv-Norm-ReLU as a CondSeq
    """
    layers: List[Tuple[str, nn.Module]] = []
    layers.append(('conv', MaskedConv2d(in_ch, out_ch, ks, center=center)))

    if norm is not None:
        layers.append(('norm', norm(out_ch)))

    layers.append(('relu', nn.ReLU(True)))
    return CondSeq(collections.OrderedDict(layers))


@experimental
def MConvBNReLU(in_ch: int, out_ch: int, ks: int, center=True) -> CondSeq:
    """
    A packed block with Masked Conv-BN-ReLU

    Args:
        in_ch (int): input channels
        out_ch (int): output channels
        ks (int): kernel size
        center (bool): whether the masked conv has access to the central pixel
            or not

    Returns:
        A packed block with MaskedConv-BN-ReLU as a CondSeq
    """
    return MConvNormReLU(in_ch, out_ch, ks, nn.BatchNorm2d, center=center)


class SpadeResBlock(PreactResBlock):
    """
    A Spade ResBlock from `Semantic Image Synthesis with Spatially-Adaptive
    Normalization`

    https://arxiv.org/abs/1903.07291
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 cond_channels: int,
                 hidden: int,
                 stride: int = 1) -> None:
        super(SpadeResBlock, self).__init__(in_channels=in_channels,
                                            out_channels=out_channels,
                                            stride=stride)

        def to_spade(m: nn.Module) -> nn.Module:
            if isinstance(m, nn.BatchNorm2d):
                return Spade2d(m.num_features, cond_channels, hidden)
            return m

        edit_model(self, to_spade)


class AutoGANGenBlock(nn.Module):
    """
    A block of the generator discovered by AutoGAN.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        skips_ch (list of int): a list with one element per incoming skip
            connection. Each element is the number of channels of the incoming
            skip connection.
        ks (int): kernel size of the convolutions
        mode (str): usampling mode, 'nearest' or 'bilinear'
    """
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 skips_ch: List[int],
                 ks: int = 3,
                 mode: str = 'nearest') -> None:
        super(AutoGANGenBlock, self).__init__()
        assert mode in ['nearest', 'bilinear']
        self.mode = mode

        self.preact = CondSeq()

        self.conv1 = ConvBlock(in_ch, out_ch, ks)
        self.conv1.to_preact().remove_batchnorm()
        self.conv1.leaky()

        self.conv2 = ConvBlock(out_ch, out_ch, ks)
        self.conv2.to_preact().remove_batchnorm()
        self.conv2.leaky()

        self.shortcut = None
        if in_ch != out_ch:
            self.shortcut = kaiming(Conv1x1(in_ch, out_ch, 1))
            self.preact.add_module('relu', self.conv1.relu)
            del self.conv1.relu

        self.skip_convs = nn.ModuleList(
            [kaiming(Conv1x1(ch, out_ch)) for ch in skips_ch])

    def forward(
            self,
            x: torch.Tensor,
            skips: List[torch.Tensor] = []
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x (tensor): input tensor
            skips (list of tensor): a tensor per incoming skip connection

        Return:
            output tensor, intermediate value to use in the next block's skip
                connections
        """
        x = F.interpolate(x, scale_factor=2., mode=self.mode)
        x = self.preact(x)
        x_skip = x

        if self.shortcut is not None:
            x_skip = self.shortcut(x_skip)

        x_mid = self.conv1(x)

        x_w_skips = x_mid.clone()
        for conv, skip in zip(self.skip_convs, skips):
            x_w_skips.add_(
                conv(F.interpolate(skip, size=x_mid.shape[-2:],
                                   mode=self.mode)))

        x = self.conv2(x_w_skips)
        return x + x_skip, x_mid


class ResidualDiscrBlock(PreactResBlock):
    """
    A preactivated resblock suited for discriminators: it features leaky relus,
    no batchnorm, and an optional downsampling operator.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 downsample: bool = False) -> None:
        super().__init__(in_channels, out_channels)
        if downsample:
            self.post.add_module('downsample', nn.AvgPool2d(2))
        self.remove_batchnorm()
        make_leaky(self)

    @experimental
    def to_equal_lr(self) -> 'ResidualDiscrBlock':
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                try:
                    nn.utils.remove_spectral_norm(m)
                except ValueError:
                    pass
                kaiming(m, dynamic=True, a=0.2)

        with torch.no_grad():
            self.branch.conv2.weight_g.zero_()

        return self

    @experimental
    def to_spectral_norm(self) -> 'ResidualDiscrBlock':
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                try:
                    remove_weight_scale(m)
                except ValueError:
                    pass
                nn.utils.spectral_norm(m)

        assert isinstance(self.branch.conv2, nn.Conv2d)
        xavier(self.branch.conv2, a=0.2)
        return self


@experimental
class StyleGAN2Block(nn.Module):
    """
    A Upsample-(ModulatedConv-Noise-LeakyReLU)* block from StyleGAN2

    Args:
        in_ch (int): input channels
        out_ch (int): output channels
        noise_size (int): channels of the conditioning style / noise vector
        upsample (bool): whether to upsample the input or not (default: False)
        n_layers (int): number of conv-noise-relu blocks (default: 2)
        equal_lr (bool): whether to equalize parameters' lr with dynamic weight
            init scaling (weight_scale)

    """
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 noise_size: int,
                 upsample: bool = False,
                 n_layers: int = 2,
                 equal_lr: bool = True):
        super().__init__()
        self.upsample_mode = 'bilinear'
        self.equal_lr = equal_lr
        dyn = equal_lr
        self.upsample = upsample
        inside = ModuleGraph(outputs=f'in_{n_layers}')

        for i in range(n_layers):
            conv = ModulatedConv(in_ch,
                                 noise_size,
                                 out_ch,
                                 kernel_size=3,
                                 padding=1,
                                 bias=True)
            kaiming(conv, dynamic=dyn, a=0.2)
            inside.add_operation(inputs=[f'in_{i}', 'w'],
                                 outputs=[f'conv_{i}'],
                                 name=f'conv_{i}',
                                 operation=conv)

            noise = Noise(out_ch, inplace=True, bias=False)
            inside.add_operation(inputs=[f'conv_{i}', f'noise_{i}'],
                                 operation=noise,
                                 name=f'plus_noise_{i}',
                                 outputs=[f'plus_noise_{i}'])

            inside.add_operation(inputs=[f'plus_noise_{i}'],
                                 operation=nn.LeakyReLU(0.2, True),
                                 outputs=[f'in_{i+1}'],
                                 name=f'relu_{i+1}')

            in_ch = out_ch

        if dyn:
            for m in inside:
                if isinstance(m, ModulatedConv):
                    xavier(m.make_s, dynamic=True)

        self.inside = inside
        self.to_rgb = xavier(ModulatedConv(out_ch,
                                           noise_size,
                                           3,
                                           kernel_size=1,
                                           padding=0,
                                           bias=True,
                                           demodulate=False),
                             dynamic=dyn,
                             a=0.2)

    def n_noise(self) -> int:
        return len(self.inside) // 3

    def extra_repr(self):
        return (f"upsample={self.upsample} n_layers={len(self.inside) // 3} "
                f"equal_lr={self.equal_lr}")

    def forward(self, maps, w, rgb, noises=None):
        if rgb is None:
            rgb = torch.zeros(maps.shape[0],
                              3,
                              *maps.shape[2:],
                              device=maps.device)

        if self.upsample:
            rgb = nn.functional.interpolate(rgb,
                                            scale_factor=2,
                                            mode=self.upsample_mode,
                                            align_corners=False,
                                            recompute_scale_factor=True)
            maps = nn.functional.interpolate(maps,
                                             scale_factor=2,
                                             mode=self.upsample_mode,
                                             align_corners=False,
                                             recompute_scale_factor=True)
        maps = self.inside(
            in_0=maps,
            w=w,
            **({f'noise_{i}': None
                for i in range(len(self.inside) // 3)}))
        return rgb + self.to_rgb(maps, w), maps
