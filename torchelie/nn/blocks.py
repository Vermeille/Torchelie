import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Conv2d, Conv3x3, Conv1x1
from .batchnorm import ConditionalBN2d, Spade2d
from .condseq import CondSeq
from .maskedconv import MaskedConv2d
from torchelie.utils import kaiming, xavier


def Conv2dNormReLU(in_ch, out_ch, ks, norm, stride=1, leak=0):
    layer = [kaiming(Conv2d(in_ch, out_ch, ks, stride=stride), a=leak)]

    if norm is not None:
        layer.append(norm(out_ch))

    if leak != 0:
        layer.append(nn.LeakyReLU(leak))
    else:
        layer.append(nn.ReLU())

    return CondSeq(*layer)


def MConvNormReLU(in_ch, out_ch, ks, norm, center=True):
    return CondSeq(
            MaskedConv2d(in_ch, out_ch, ks, center=center),
            *([norm(out_ch)] if norm is not None else []),
            nn.ReLU(inplace=True)
        )


def MConvBNrelu(in_ch, out_ch, ks, center=True):
    return MConvNormReLU(in_ch, out_ch, ks, nn.BatchNorm2d, center=center)


def Conv2dBNReLU(in_ch, out_ch, ks, stride=1, leak=0):
    return Conv2dNormReLU(in_ch,
                          out_ch,
                          ks,
                          nn.BatchNorm2d,
                          leak=leak,
                          stride=stride)


class Conv2dCondBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, cond_ch, ks, leak=0):
        super(Conv2dCondBNReLU, self).__init__()
        self.conv = kaiming(Conv2d(in_ch, out_ch, ks), a=leak)
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


class OriginalResBlockFn:
    @staticmethod
    def __call__(m, x):
        out = m.conv1(x)
        out = m.bn1(out)
        out = m.relu(out)

        out = m.conv2(out)
        out = m.bn2(out)
        out += getattr(m, 'shortcut', lambda x: x)(x)
        return m.relu(out)


class PreactResBlockFn:
    @staticmethod
    def __call__(m, x):
        out = m.bn1(x)
        out = m.relu(out)
        x = m.shortcut(out) if hasattr(m, 'shortcut') else x
        out = m.conv1(out)

        out = m.bn2(out)
        out = m.relu(out)
        out = m.conv2(out)
        return out + x


class ConditionalResBlockFn:
    @staticmethod
    def condition(m, z):
        m.bn1.condition(z)
        m.bn2.condition(z)

    @staticmethod
    def __call__(m, x, z=None):
        out = m.conv1(x)
        out = m.bn1(out, z)
        out = m.relu(out)

        out = m.conv2(out)
        out = m.bn2(out, z)
        out += getattr(m, 'shortcut', lambda x: x)(x)
        return m.relu(out)


class PreactCondResBlock:
    @staticmethod
    def condition(m, z):
        m.bn1.condition(z)
        m.bn2.condition(z)

    @staticmethod
    def __call__(m, x, z=None):
        out = m.bn1(x, z)
        out = m.relu(out)
        x = m.shortcut(out) if hasattr(m, 'shortcut') else x
        out = m.conv1(out)

        out = m.bn2(out, z)
        out = m.relu(out)
        out = m.conv2(out)
        return out + x


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super(ResBlock, self).__init__()
        self.fn = OriginalResBlockFn()

        self.conv1 = kaiming(Conv3x3(in_ch, in_ch, stride=stride))
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU()

        self.conv2 = kaiming(Conv3x3(in_ch, out_ch))
        self.bn2 = nn.BatchNorm2d(out_ch)

        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                kaiming(Conv1x1(in_ch, out_ch, stride)),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        return self.fn(self, x)


class PreactResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super(PreactResBlock, self).__init__()
        self.fn = PreactResBlockFn()

        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU()
        self.conv1 = kaiming(Conv3x3(in_ch, in_ch, stride=stride))

        self.bn2 = nn.BatchNorm2d(in_ch)
        self.conv2 = kaiming(Conv3x3(in_ch, out_ch))

        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                kaiming(Conv1x1(in_ch, out_ch, stride)),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        return self.fn(self, x)


class ConditionalResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, hidden, stride):
        super(ConditionalResBlock, self).__init__()
        self.fn = ConditionalResBlockFn()
        self.conv1 = kaiming(Conv3x3(in_ch, in_ch, stride=stride))
        self.bn1 = ConditionalBN2d(in_ch, hidden)
        self.relu = nn.ReLU()
        self.conv2 = kaiming(Conv3x3(in_ch, out_ch))
        self.bn2 = ConditionalBN2d(out_ch, hidden)

        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                kaiming(Conv1x1(in_ch, out_ch, stride)),
                nn.BatchNorm2d(out_ch))

    def condition(self, z):
        self.fn.condition(self, z)

    def forward(self, x, z=None):
        return self.fn(self, x, z)


class SpadeResBlock(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 cond_channels,
                 hidden,
                 stride,
                 blktype=PreactCondResBlock):
        super(SpadeResBlock, self).__init__()
        self.fn = blktype()
        self.conv1 = kaiming(Conv3x3(in_ch, in_ch, stride=stride))
        self.bn1 = Spade2d(in_ch, cond_channels, hidden)
        self.relu = nn.ReLU()
        self.conv2 = kaiming(Conv3x3(in_ch, out_ch))
        self.bn2 = Spade2d(out_ch, cond_channels, hidden)

        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                kaiming(Conv1x1(in_ch, out_ch, stride=stride)),
                Spade2d(out_ch, cond_channels, hidden))

    def condition(self, z):
        self.fn.condition(self, z)

    def forward(self, x, z=None):
        return self.fn(self, x, z)
