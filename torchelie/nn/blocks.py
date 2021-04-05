import collections
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Conv2d, Conv3x3, Conv1x1
from .debug import Dummy
from .batchnorm import ConditionalBN2d, Spade2d
from .condseq import CondSeq
from .maskedconv import MaskedConv2d
from torchelie.utils import kaiming, xavier, normal_init, constant_init
from .layers import ModulatedConv
from .noise import Noise
from torchelie.nn.graph import ModuleGraph


def Conv2dNormReLU(in_ch, out_ch, ks, norm, stride=1, leak=0, inplace=False):
    """
    A packed block with Conv-Norm-ReLU

    Args:
        in_ch (int): input channels
        out_ch (int): output channels
        ks (int): kernel size
        norm (ctor): A normalization layer constructor in the form
            :code:`(num_layers) -> norm layer` or None
        stride (int): stride of the conv
        leak (float): negative slope of the LeakyReLU or 0 for ReLU

    Returns:
        A packed block with Conv-Norm-ReLU as a CondSeq
    """
    layer = [('conv',
              kaiming(Conv2d(in_ch,
                             out_ch,
                             ks,
                             stride=stride,
                             bias=norm is None),
                      a=leak))]

    inplace = inplace or norm is not None
    if norm is not None:
        layer.append(('norm', norm(out_ch)))

    if leak != 0:
        layer.append(('relu', nn.LeakyReLU(leak, inplace=inplace)))
    else:
        layer.append(('relu', nn.ReLU(inplace=inplace)))

    return CondSeq(collections.OrderedDict(layer))


class SEBlock(nn.Module):
    """
    A Squeeze-And-Excite block

    Args:
        in_ch (int): input channels
        reduction (int): channels reduction factor for the hidden number of
            channels
    """
    def __init__(self, in_ch, reduction=16):
        super(SEBlock, self).__init__()
        reduc = in_ch // reduction
        self.proj = nn.Sequential(
            collections.OrderedDict([
                ('pool', nn.AdaptiveAvgPool2d(1)),
                ('squeeze', kaiming(nn.Conv2d(in_ch, reduc, 1))),
                ('relu', nn.ReLU(True)),
                ('excite', constant_init(nn.Conv2d(reduc, in_ch, 1), 0)),
                ('attn', nn.Sigmoid())
            ]))

    def forward(self, x):
        return x * self.proj(x)


def MConvNormReLU(in_ch, out_ch, ks, norm, center=True):
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
    layers = [('conv', MaskedConv2d(in_ch, out_ch, ks, center=center))]

    if norm is not None:
        layers.append(('norm', norm(out_ch)))

    layers.append(('relu', nn.ReLU(True)))
    return CondSeq(collections.OrderedDict(layers))


def MConvBNrelu(in_ch, out_ch, ks, center=True):
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


def Conv2dBNReLU(in_ch, out_ch, ks, stride=1, leak=0, inplace=False):
    """
    A packed block with Conv-BN-ReLU

    Args:
        in_ch (int): input channels
        out_ch (int): output channels
        ks (int): kernel size
        stride (int): stride of the conv
        leak (float): negative slope of the LeakyReLU or 0 for ReLU
        inplace (bool): if relu should be inplace

    Returns:
        A packed block with Conv-BN-ReLU as a CondSeq
    """
    return Conv2dNormReLU(in_ch,
                          out_ch,
                          ks,
                          nn.BatchNorm2d,
                          leak=leak,
                          stride=stride,
                          inplace=inplace)


class Conv2dCondBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, cond_ch, ks, leak=0):
        super(Conv2dCondBNReLU, self).__init__()
        self.conv = kaiming(Conv2d(in_ch, out_ch, ks, bias=False), a=leak)
        self.cbn = ConditionalBN2d(out_ch, cond_ch)
        self.leak = leak

    def condition(self, z):
        self.cbn.condition(z)

    def forward(self, x, z=None):
        x = self.conv(x)
        x = self.cbn(x, z)
        if self.leak == 0:
            x = F.relu(x)
        else:
            x = F.leaky_relu(x, self.leak)
        return x


def make_resnet_shortcut(in_ch, out_ch, stride, norm=nn.BatchNorm2d):
    if in_ch == out_ch and stride == 1:
        return CondSeq()

    sc = []
    if stride != 1:
        sc.append(('pool', nn.AvgPool2d(3, stride, 1)))

    sc += [('conv', kaiming(Conv1x1(in_ch, out_ch, bias=norm is None)))]

    if norm:
        sc += [('norm', norm(out_ch))]

    return CondSeq(collections.OrderedDict(sc))


def make_preact_resnet_shortcut(in_ch, out_ch, stride):
    if in_ch == out_ch and stride == 1:
        return CondSeq()

    sc = []
    if stride != 1:
        sc.append(('pool', nn.AvgPool2d(3, stride, 1)))
    sc.append(('conv', kaiming(Conv1x1(in_ch, out_ch))))

    return CondSeq(collections.OrderedDict(sc))


class ResBlock(nn.Module):
    """
    A Residual Block. Skip connection will be added if the number of input and
    output channels don't match or stride > 1 is used.

    Args:
        in_ch (int): input channels
        out_ch (int): output channels
        stride (int): stride
        norm (callable or None): a norm layer constructor in the form
            :code:`(num channels) -> norm layer` or None.
        use_se (bool): whether to use Squeeze-And-Excite after the convolutions
            for a SE-ResBlock
        bottleneck (bool): whether to use the standard block with two 3x3 convs
            or the bottleneck block with 1x1 -> 3x3 -> 1x1 convs.
    """
    def __init__(self,
                 in_ch,
                 out_ch,
                 stride=1,
                 norm=nn.BatchNorm2d,
                 use_se=False,
                 bottleneck=False):
        super(ResBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.use_se = use_se
        self.bottleneck = bottleneck
        bias = norm is None

        if bottleneck:
            mid = out_ch // 4
            self.branch = CondSeq(
                collections.OrderedDict([
                    ('conv1', kaiming(Conv1x1(in_ch, mid, bias=bias))),
                    ('bn1', constant_init(norm(mid), 1) if norm else Dummy()),
                    ('relu', nn.ReLU(True)),
                    ('conv2',
                     kaiming(Conv3x3(mid, mid, stride=stride, bias=bias))),
                    ('bn2', constant_init(norm(mid), 1) if norm else Dummy()),
                    ('relu2', nn.ReLU(True)),
                    ('conv3', constant_init(Conv1x1(mid, out_ch), 0)),
                ]))
        else:
            self.branch = CondSeq(
                collections.OrderedDict([
                    ('conv1',
                     kaiming(Conv3x3(in_ch, out_ch, stride=stride,
                                     bias=bias))),
                    ('bn1',
                     constant_init(norm(out_ch), 1) if norm else Dummy()),
                    ('relu', nn.ReLU(True)),
                    ('conv2', kaiming(Conv3x3(out_ch, out_ch, bias=bias))),
                    ('bn2',
                     constant_init(norm(out_ch), 0) if norm else Dummy())
                ]))

        if use_se:
            self.branch.add_module('se', SEBlock(out_ch))

        self.relu = nn.ReLU(True)

        self.shortcut = make_resnet_shortcut(in_ch, out_ch, stride, norm=None)

    def __repr__(self):
        return "{}({}, {}, stride={}, norm={})".format(
            ("SE-" if self.use_se else "") +
            ("Bottleneck" if self.bottleneck else "ResBlock"), self.in_ch,
            self.out_ch, self.stride, self.branch.bn1.__class__.__name__)

    def condition(self, z):
        self.branch.condition(z)
        self.shortcut.condition(z)

    def forward(self, x, z=None):
        if z is not None:
            self.condition(z)

        return self.relu(self.branch(x).add_(self.shortcut(x)))


class HardSigmoid(nn.Module):
    """
    Hard Sigmoid
    """
    def forward(self, x):
        return x.add_(0.5).clamp_(min=0, max=1)


class HardSwish(nn.Module):
    """
    Hard Swish
    """
    def forward(self, x):
        return x.add(0.5).clamp_(min=0, max=1).mul_(x)


class PreactResBlock(nn.Module):
    """
    A Preactivated Residual Block. Skip connection will be added if the number
    of input and output channels don't match or stride > 1 is used.

    Args:
        in_ch (int): input channels
        out_ch (int): output channels
        stride (int): stride
        norm (callable or None): a norm layer constructor in the form
            :code:`(num channels) -> norm layer` or None.
        use_se (bool): whether to use Squeeze-And-Excite after the convolutions
            for a SE-ResBlock
        bottleneck (bool): whether to use the standard block with two 3x3 convs
            or the bottleneck block with 1x1 -> 3x3 -> 1x1 convs.
    """
    def __init__(self,
                 in_ch,
                 out_ch,
                 stride=1,
                 norm=nn.BatchNorm2d,
                 dropout=0.,
                 use_se=False,
                 bottleneck=False,
                 first_layer=False):
        super(PreactResBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.use_se = use_se
        self.bottleneck = bottleneck
        self.first_layer = first_layer
        bias = norm is None

        if bottleneck:
            mid = out_ch // 4
            self.branch = CondSeq(
                collections.OrderedDict([
                    ('bn1',
                     constant_init(norm(in_ch), 1) if norm else Dummy()),
                    ('relu', nn.ReLU(not first_layer and norm is not None)),
                    ('conv1', kaiming(Conv1x1(in_ch, mid, bias=bias))),
                    ('bn2', constant_init(norm(mid), 1) if norm else Dummy()),
                    ('relu2', nn.ReLU(True)),
                    ('conv2',
                     kaiming(Conv3x3(mid, mid, stride=stride, bias=bias))),
                    ('bn3', constant_init(norm(mid), 1) if norm else Dummy()),
                    ('relu3', nn.ReLU(True)),
                    ('conv3', constant_init(Conv1x1(mid, out_ch), 0))
                ]))
        else:
            self.branch = CondSeq(
                collections.OrderedDict([
                    ('bn1',
                     constant_init(norm(in_ch), 1) if norm else Dummy()),
                    ('relu', nn.ReLU(not first_layer and norm is not None)),
                    ('conv1',
                     kaiming(Conv3x3(in_ch, out_ch, stride=stride,
                                     bias=bias))),
                    ('dropout1', nn.Dropout2d(dropout, inplace=True)),
                    ('bn2',
                     constant_init(norm(out_ch), 1) if norm else Dummy()),
                    ('relu2', nn.ReLU(True)),
                    ('conv2', constant_init(Conv3x3(out_ch, out_ch), 0)),
                    ('dropout2', nn.Dropout2d(dropout, inplace=True)),
                ]))

        if use_se:
            self.branch.add_module('se', SEBlock(out_ch))

        self.shortcut = make_preact_resnet_shortcut(in_ch, out_ch, stride)

    def condition(self, z):
        self.branch.condition(z)
        self.shortcut.condition(z)

    def forward(self, x, z=None):
        if z is not None:
            self.condition(z)

        if len(self.shortcut) > 0:
            out = self.branch.relu(self.branch.bn1(x))
            return self.shortcut(out).add_(self.branch[2:](out))
        elif self.first_layer:
            out = self.branch.relu(self.branch.bn1(x))
            return self.shortcut(out) + self.branch[2:](out)
        else:
            return x + self.branch(x)


class SpadeResBlock(PreactResBlock):
    """
    A Spade ResBlock from `Semantic Image Synthesis with Spatially-Adaptive
    Normalization`

    https://arxiv.org/abs/1903.07291
    """
    def __init__(self, in_ch, out_ch, cond_channels, hidden, stride=1):
        norm = functools.partial(Spade2d,
                                 cond_channels=cond_channels,
                                 hidden=hidden)
        super(SpadeResBlock, self).__init__(in_ch=in_ch,
                                            out_ch=out_ch,
                                            stride=stride,
                                            norm=norm)


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
    def __init__(self, in_ch, out_ch, skips_ch, ks=3, mode='nearest'):
        super(AutoGANGenBlock, self).__init__()
        assert mode in ['nearest', 'bilinear']
        self.mode = mode

        self.conv1 = kaiming(nn.Conv2d(in_ch, out_ch, ks, padding=ks // 2))
        self.conv2 = kaiming(nn.Conv2d(out_ch, out_ch, ks, padding=ks // 2))

        self.shortcut = None
        if in_ch != out_ch:
            self.shortcut = kaiming(Conv1x1(in_ch, out_ch, 1))

        self.skip_convs = nn.ModuleList(
            [kaiming(Conv1x1(ch, out_ch)) for ch in skips_ch])

    def forward(self, x, skips=[]):
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
        if self.shortcut is not None:
            x = F.leaky_relu(x, 0.2)
            x_skip = x
            x_skip = self.shortcut(x_skip)
        else:
            x_skip = x
            x = F.leaky_relu(x, 0.2)

        x_mid = self.conv1(x)

        x_w_skips = x_mid
        for conv, skip in zip(self.skip_convs, skips):
            x_w_skips += conv(
                F.interpolate(skip, size=x_mid.shape[-2:], mode=self.mode))

        x = self.conv2(F.leaky_relu(x_w_skips, 0.2))
        return x + x_skip, F.leaky_relu(x_mid, 0.2)


class ResidualDiscrBlock(torch.nn.Module):
    """
    A residual block with downsampling

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        downsample (bool): whether to downsample
        equal_lr (bool): whether to use dynamic weight scaling for equal lr
            such as in StyleGAN2
    """
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 downsample: bool = False,
                 equal_lr: bool = False,
                 force_shortcut: bool = False):
        super(ResidualDiscrBlock, self).__init__()
        self.equal_lr = equal_lr
        self.branch = nn.Sequential(*[
            nn.LeakyReLU(0.2),
            kaiming(nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    dynamic=equal_lr,
                    mode='fan_in'),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(3, 2, 1) if downsample else Dummy(),
            xavier(nn.Conv2d(out_ch, out_ch, 3, padding=1),
                   dynamic=equal_lr,
                   nonlinearity='linear',
                   mode='fan_in')
        ])

        with torch.no_grad():
            if equal_lr:
                self.branch[-1].weight_g.data.zero_()
            else:
                self.branch[-1].weight.data.zero_()

        self.downsample = downsample
        self.sc = None
        if in_ch != out_ch or force_shortcut:
            self.sc = nn.Sequential(
                kaiming(nn.Conv2d(in_ch, out_ch, 3, padding=1),
                       dynamic=equal_lr,
                       nonlinearity='linear',
                       mode='fan_in'),
                nn.AvgPool2d(3, 2, 1) if downsample else Dummy(),
                )

    def extra_repr(self):
        return f"equal_lr={self.equal_lr} downsample={self.downsample}"

    def forward(self, x):
        """
        Forward pass

        Args:
            x (tensor): input tensor

        Returns:
            output tensor
        """
        if self.sc is not None:
            # FIXME: we can share the relu and have it inplace
            x = self.branch[0](x)
            res = self.branch[1:](x)
            x = self.sc(x)
        else:
            res = self.branch(x)
            if self.downsample:
                x = nn.functional.avg_pool2d(x, 3, 2, 1)
        return x + res


class SNResidualDiscrBlock(ResidualDiscrBlock):
    """
    A residual block with downsampling and spectral normalization.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        downsample (bool): whether to downsample
    """
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = False,
            equal_lr: bool = False):
        super().__init__(in_ch, out_ch, downsample, equal_lr=False)
        xavier(self.branch[-1], nonlinearity='linear')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.utils.spectral_norm(m)


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
        inside = []

        for i in range(n_layers):
            conv = ModulatedConv(in_ch,
                                 noise_size,
                                 out_ch,
                                 kernel_size=3,
                                 padding=1,
                                 bias=True)
            conv = kaiming(conv, dynamic=dyn, a=0.2)

            noise = Noise(out_ch, inplace=True, bias=False)
            inside += [((f'in_{i}', 'w'), conv, f'conv_{i}'),
                       ((f'conv_{i}', f'noise_{i}'), noise, f'plus_noise_{i}'),
                       (f'plus_noise_{i}', nn.LeakyReLU(0.2,
                                                        True), f'in_{i+1}')]
            in_ch = out_ch
        inside += [(f'in_{n_layers}', nn.Identity(), 'out')]

        if dyn:
            for m in inside:
                if isinstance(m[1], ModulatedConv):
                    xavier(m[1].make_s, dynamic=True)

        self.inside = ModuleGraph(inside, outputs='out')
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
                                            mode=self.upsample_mode)
            maps = nn.functional.interpolate(maps,
                                             scale_factor=2,
                                             mode=self.upsample_mode)
        maps = self.inside(
            in_0=maps,
            w=w,
            **({f'noise_{i}': None
                for i in range(len(self.inside) // 3)}))['out']
        return rgb + self.to_rgb(maps, w), maps
