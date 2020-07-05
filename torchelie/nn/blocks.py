import collections
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Conv2d, Conv3x3, Conv1x1
from .batchnorm import ConditionalBN2d, Spade2d
from .condseq import CondSeq
from .maskedconv import MaskedConv2d
from torchelie.utils import kaiming, xavier, normal_init, constant_init


def Conv2dNormReLU(in_ch, out_ch, ks, norm, stride=1, leak=0):
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

    if norm is not None:
        layer.append(('norm', norm(out_ch)))

    if leak != 0:
        layer.append(('relu', nn.LeakyReLU(leak)))
    else:
        layer.append(('relu', nn.ReLU()))

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
        self.proj = nn.Sequential(
            collections.OrderedDict([
                ('pool', nn.AdaptiveAvgPool2d(1)),
                ('squeeze', kaiming(nn.Conv2d(in_ch, in_ch // reduction, 1))),
                ('relu', nn.ReLU(True)),
                ('excite', xavier(nn.Conv2d(in_ch // reduction, in_ch, 1))),
                ('attn', HardSigmoid())
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


def Conv2dBNReLU(in_ch, out_ch, ks, stride=1, leak=0):
    """
    A packed block with Conv-BN-ReLU

    Args:
        in_ch (int): input channels
        out_ch (int): output channels
        ks (int): kernel size
        stride (int): stride of the conv
        leak (float): negative slope of the LeakyReLU or 0 for ReLU

    Returns:
        A packed block with Conv-BN-ReLU as a CondSeq
    """
    return Conv2dNormReLU(in_ch,
                          out_ch,
                          ks,
                          nn.BatchNorm2d,
                          leak=leak,
                          stride=stride)


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

        if bottleneck:
            mid = out_ch // 4
            self.branch = CondSeq(
                collections.OrderedDict([
                    ('conv1', kaiming(Conv1x1(in_ch, mid, bias=False))),
                    ('bn1', norm(mid)),
                    ('relu', nn.ReLU(True)),
                    ('conv2',
                     kaiming(Conv3x3(mid, mid, stride=stride, bias=False))),
                    ('bn2', norm(mid)),
                    ('relu2', nn.ReLU(True)),
                    ('conv3', kaiming(Conv1x1(mid, out_ch, bias=False))),
                ]))
        else:
            self.branch = CondSeq(
                collections.OrderedDict([
                    ('conv1', (Conv3x3(in_ch, out_ch, stride=stride, bias=False))),
                    ('bn1', norm(out_ch)),
                    ('relu', nn.ReLU(True)),
                    ('conv2', (Conv3x3(out_ch, out_ch, bias=False))),
                    ('bn2', norm(out_ch))
                ]))

        if use_se:
            self.branch.add_module('se', SEBlock(out_ch))

        self.relu = nn.ReLU(True)

        self.shortcut = make_resnet_shortcut(in_ch, out_ch, stride, norm)

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
                 use_se=False,
                 bottleneck=False):
        super(PreactResBlock, self).__init__()
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
                    ('bn1', norm(in_ch)),
                    ('relu', nn.ReLU(True)),
                    ('conv1', kaiming(Conv1x1(in_ch, mid, bias=bias))),
                    ('bn2', norm(mid)),
                    ('relu2', nn.ReLU(True)),
                    ('conv2',
                     kaiming(Conv3x3(mid, mid, stride=stride, bias=bias))),
                    ('bn3', norm(mid)),
                    ('relu3', nn.ReLU(True)),
                    ('conv3', normal_init(Conv1x1(mid, out_ch), std=0)),
                ]))
        else:
            self.branch = CondSeq(
                collections.OrderedDict([
                    ('bn1', constant_init(norm(in_ch), 1)),
                    ('relu', nn.ReLU(True)),
                    ('conv1',
                     kaiming(Conv3x3(in_ch, out_ch, stride=stride,
                                     bias=bias))),
                    ('bn2', constant_init(norm(out_ch), 1)),
                    ('relu2', nn.ReLU(True)),
                    ('conv2', normal_init(Conv3x3(out_ch, out_ch), std=0))
                ]))

        if use_se:
            self.branch.add_module('se', SEBlock(out_ch))

        self.shortcut = make_resnet_shortcut(in_ch, out_ch, stride, norm=None)

    def __repr__(self):
        return "{}({}, {}, stride={}, norm={})".format(
            ("SE-" if self.use_se else "") +
            ("PreactBottleneck" if self.bottleneck else "PreactResBlock"),
            self.in_ch, self.out_ch, self.stride,
            self.branch.bn1.__class__.__name__)

    def condition(self, z):
        self.branch.condition(z)
        self.shortcut.condition(z)

    def forward(self, x, z=None):
        if z is not None:
            self.condition(z)

        if len(self.shortcut) == 0:
            return x + self.branch(x)
        else:
            out = self.branch.relu(self.branch.bn1(x))
            return self.shortcut(out).add_(self.branch[2:](out))


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


class SNResidualDiscrBlock(torch.nn.Module):
    """
    A residual block with downsampling and spectral normalization.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        downsample (bool): whether to downsample
    """

    def __init__(self, in_ch, out_ch, downsample=False):
        super(SNResidualDiscrBlock, self).__init__()
        self.branch = nn.Sequential(*[
            nn.ReLU(),
            kaiming(nn.Conv2d(in_ch, out_ch, 3, padding=1)),
            nn.ReLU(True),
            kaiming(nn.Conv2d(out_ch, out_ch, 3, padding=1)),
        ])
        nn.utils.spectral_norm(self.branch[1])
        nn.utils.spectral_norm(self.branch[3])

        self.downsample = downsample
        self.sc = None
        if in_ch != out_ch:
            self.sc = kaiming(nn.Conv2d(in_ch, out_ch, 1))
            nn.utils.spectral_norm(self.sc)

    def forward(self, x):
        """
        Forward pass

        Args:
            x (tensor): input tensor

        Returns:
            output tensor
        """
        res = self.branch(x)
        if self.sc is not None:
            # FIXME: we can share the relu and have it inplace
            x = self.sc(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2, 2, 0)
            res = F.avg_pool2d(res, 2, 2, 0)
        return x + res

