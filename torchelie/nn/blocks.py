import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Conv2d
from .batchnorm import ConditionalBN2d, Spade2d


def Conv2dNormReLU(in_ch, out_ch, ks, norm, leak=0):
    layer = [
            Conv2d(in_ch, out_ch, ks),
            norm,
    ]

    if leak != 0:
        layer.append(nn.LeakyReLU(leak))
    else:
        layer.append(nn.ReLU())

    nn.init.kaiming_normal_(layer[0].weight, a=leak)
    nn.init.constant_(layer[0].bias, 0)

    return nn.Sequential(*layer)


def Conv2dBNReLU(in_ch, out_ch, ks, leak=0):
    return Conv2dNormReLU(in_ch, out_ch, ks, nn.BatchNorm2d(out_ch), leak=leak)


class Conv2dCondBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, cond_ch, ks, leak=0):
        super(Conv2dCondBNReLU, self).__init__()
        self.conv = Conv2d(in_ch, out_ch, ks)
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


class ResBlockBase_(nn.Module):
    def __init__(self):
        super(ResBlockBase_, self).__init__()
        # inherit me and give me my members

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += getattr(self, 'shortcut', lambda x: x)(x)
        return self.relu(out)


class CondResBlockBase_(nn.Module):
    def __init__(self):
        super(ResBlockBase_, self).__init__()
        # inherit me and give me my members

    def condition(self, z):
        self.bn1.condition(z)
        self.bn2.condition(z)

    def forward(self, x, z=None):
        out = self.conv1(x)
        out = self.bn1(out, z)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, z)
        out += x
        return self.relu(out)


class ResBlock(ResBlockBase_):
    def __init__(self, in_ch, out_ch, stride):
        super(ResBlock, self).__init__()
        self.conv1 = Conv3x3(in_ch, in_ch)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU()
        self.conv2 = Conv3x3(in_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                    Conv1x1(in_ch, out_ch, stride),
                    nn.BatchNorm2d(out_ch)
            )

class ConditionalResBlock(CondResBlockBase_):
    def __init__(self, in_ch, out_ch, hidden, stride):
        super(ResBlock, self).__init__()
        self.conv1 = Conv3x3(in_ch, in_ch)
        self.bn1 = ConditionalBN2d(in_ch, hidden)
        self.relu = nn.ReLU()
        self.conv2 = Conv3x3(in_ch, out_ch)
        self.bn2 = ConditionalBN2d(out_ch, hidden)

        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                    Conv1x1(in_ch, out_ch, stride),
                    ConditionalBN2d(out_ch)
            )

class SpadeResBlock(CondResBlockBase_):
    def __init__(self, in_ch, out_ch, hidden, stride):
        super(ResBlock, self).__init__()
        self.conv1 = Conv3x3(in_ch, in_ch, stride=stride)
        self.bn1 = Spade2d(in_ch, hidden)
        self.relu = nn.ReLU()
        self.conv2 = Conv3x3(in_ch, out_ch)
        self.bn2 = Spade2d(out_ch, hidden)

        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                    Conv1x1(in_ch, out_ch, stride=stride),
                    Spade2d(out_ch)
            )

