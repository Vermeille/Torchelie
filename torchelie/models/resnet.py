import torch
import torch.nn as nn
import torchelie.nn as tnn
import torchelie.utils as tu

from typing import List, Optional
from .classifier import ClassificationHead

PREACT_BLOCKS = (tnn.PreactResBlock, tnn.PreactResBlockBottleneck)
STD_BLOCKS = (tnn.ResBlock, tnn.ResBlockBottleneck)
BLOCKS = PREACT_BLOCKS + STD_BLOCKS
BOTTLENECKS_BLOCKS = (tnn.PreactResBlockBottleneck, tnn.ResBlockBottleneck)


class ResNetInput(nn.Module):
    relu: Optional[nn.ReLU]
    pool: Optional[nn.Module]

    def __init__(self, in_channels: int = 3, out_channels: int = 64) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = tu.kaiming(
            tnn.Conv2d(in_channels, out_channels, 7, stride=2))
        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool2d(3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.pool is not None:
            x = self.pool(x)
        return x

    def set_input_specs(self, input_size: int, in_channels=3) -> 'ResNetInput':
        self.in_channels = in_channels
        in_ch = self.in_channels
        out_ch = self.out_channels

        if input_size <= 64:
            self.conv = tu.kaiming(tnn.Conv2d(in_ch, out_ch, ks=3))
            del self.pool
            self.pool = None
        elif input_size <= 128:
            self.conv = tu.kaiming(tnn.Conv2d(in_ch, out_ch, 5, stride=2))
            del self.pool
            self.pool = None
        else:
            self.conv = tu.kaiming(tnn.Conv2d(in_ch, out_ch, 7, stride=2))
            if self.pool is None:
                self.pool = nn.MaxPool2d(3, 2, 1)

        return self


class ResNet(nn.Module):
    def __init__(self, arch: List[str], num_classes: int) -> None:
        super().__init__()

        def parse(layer: str) -> List[int]:
            return [int(x) for x in layer.split(':')]

        self.arch = list(map(parse, arch))

        self.features = tnn.CondSeq()
        self.features.add_module('input', ResNetInput(3, self.arch[0][0]))

        self._change_block_type('basic')
        self.classifier = ClassificationHead(self.arch[-1][0], num_classes)

    def _make_block(self, block_type: str, in_ch: int, out_ch: int,
                    stride: int) -> nn.Module:
        if block_type == 'basic':
            return tnn.ResBlock(in_ch, out_ch, stride)
        if block_type == 'bottleneck':
            return tnn.ResBlockBottleneck(in_ch, out_ch, stride)
        if block_type == 'preact_basic':
            return tnn.PreactResBlock(in_ch, out_ch, stride)
        if block_type == 'preact_bottleneck':
            return tnn.PreactResBlockBottleneck(in_ch, out_ch, stride)
        if block_type == 'resnext':
            return tnn.ResBlockBottleneck(in_ch, out_ch, stride).resnext()
        if block_type == 'preact_resnext':
            return tnn.PreactResBlockBottleneck(in_ch, out_ch,
                                                stride).resnext()
        if block_type == 'wide':
            return tnn.ResBlockBottleneck(in_ch, out_ch, stride).wide()
        if block_type == 'preact_wide':
            return tnn.PreactResBlockBottleneck(in_ch, out_ch, stride).wide()
        assert False

    def _change_block_type(self, ty: str) -> None:
        arch = self.arch

        feats = tnn.CondSeq()
        assert isinstance(self.features.input, ResNetInput)
        feats.add_module('input', self.features.input)
        in_ch = arch[0][0]
        self.in_channels = in_ch

        for i, (ch, s) in enumerate(arch):
            feats.add_module(f'block_{i}',
                             self._make_block(ty, in_ch, ch, stride=s))
            in_ch = ch
        self.out_channels = ch

        if 'preact' in ty:
            assert isinstance(feats.block_0, PREACT_BLOCKS)
            feats.block_0.no_preact()
            feats.add_module('final_bn', nn.BatchNorm2d(self.out_channels))
            feats.add_module('final_relu', nn.ReLU(True))
        self.features = feats

    def to_bottleneck(self) -> 'ResNet':
        self._change_block_type('bottleneck')
        return self

    def to_preact_bottleneck(self) -> 'ResNet':
        self._change_block_type('preact_bottleneck')
        return self

    def to_preact(self) -> 'ResNet':
        self._change_block_type('preact_basic')
        return self

    def to_resnext(self) -> 'ResNet':
        self._change_block_type('resnext')
        return self

    def to_preact_resnext(self) -> 'ResNet':
        self._change_block_type('preact_resnext')
        return self

    def to_wide(self) -> 'ResNet':
        self._change_block_type('wide')
        return self

    def to_preact_wide(self) -> 'ResNet':
        self._change_block_type('preact_wide')
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def add_se(self) -> 'ResNet':
        for m in self.features:
            if isinstance(m, BLOCKS):
                m.use_se()
        return self

    def set_input_specs(self,
                        input_size: int = 224,
                        in_channels: int = 3) -> 'ResNet':
        assert isinstance(self.features.input, ResNetInput)
        self.features.input.set_input_specs(input_size=input_size,
                                            in_channels=in_channels)
        return self


#
# CIFAR
#


def resnet20_cifar() -> ResNet:
    return ResNet([
        '16:1', '16:1', '16:1', '32:2', '32:1', '32:1', '64:2', '64:1', '64:1'
    ], 10).set_input_specs(input_size=32)


def preact_resnet20_cifar() -> ResNet:
    return resnet20_cifar().to_preact()


#
# ResNets
#


def resnet18(num_classes: int) -> ResNet:
    return ResNet(
        ['64:1', '64:1', '128:2', '128:1', '256:2', '256:1', '512:2', '512:1'],
        num_classes)


def resnet34(num_classes: int) -> ResNet:
    return ResNet(['64:1'] * 3 + ['128:2'] + ['128:1'] * 3 + ['256:2']
                  + ['256:1'] * 5 + ['512:2', '512:1', '512:1'], num_classes)


def resnet50(num_classes: int) -> ResNet:
    net = ResNet(['256:1'] * 3 + ['512:2'] + ['512:1'] * 3 + ['1024:2']
                 + ['1024:1'] * 5 + ['2048:2', '2048:1', '2048:1'], num_classes)
    net.to_bottleneck()
    return net


def resnet101(num_classes: int) -> ResNet:
    net = ResNet(['256:1'] * 3 + ['512:2'] + ['512:1'] * 3 + ['1024:2']
                 + ['1024:1'] * 22 + ['2048:2', '2048:1', '2048:1'], num_classes)
    net.to_bottleneck()
    return net


def resnet152(num_classes: int) -> ResNet:
    net = ResNet(['256:1'] * 3 + ['512:2'] + ['512:1'] * 7 + ['1024:2']
                 + ['1024:1'] * 35 + ['2048:2', '2048:1', '2048:1'], num_classes)
    net.to_bottleneck()
    return net


#
# Preact ResNet
#


def preact_resnet18(num_classes: int) -> ResNet:
    return resnet18(num_classes).to_preact()


def preact_resnet34(num_classes: int) -> ResNet:
    return resnet34(num_classes).to_preact()


def preact_resnet50(num_classes: int) -> ResNet:
    return resnet50(num_classes).to_preact_bottleneck()


def preact_resnet101(num_classes: int) -> ResNet:
    return resnet101(num_classes).to_preact_bottleneck()


def preact_resnet152(num_classes: int) -> ResNet:
    return resnet152(num_classes).to_preact_bottleneck()


#
# ResNeXt
#


def resnext50_32x4d(num_classes: int) -> ResNet:
    return resnet50(num_classes).to_resnext()


def resnext101_32x4d(num_classes: int) -> ResNet:
    return resnet101(num_classes).to_resnext()


def resnext152_32x4d(num_classes: int) -> ResNet:
    return resnet152(num_classes).to_resnext()


#
# Preact ResNeXt
#


def preact_resnext50_32x4d(num_classes: int) -> ResNet:
    return resnet50(num_classes).to_preact_resnext()


def preact_resnext101_32x4d(num_classes: int) -> ResNet:
    return resnet101(num_classes).to_preact_resnext()


def preact_resnext152_32x4d(num_classes: int) -> ResNet:
    return resnet152(num_classes).to_preact_resnext()


#
# Wide
#

def preact_wide_resnet50(num_classes: int) -> ResNet:
    return resnet50(num_classes).to_preact_wide()


def preact_wide_resnet101(num_classes: int) -> ResNet:
    return resnet101(num_classes).to_preact_wide()

#
# Wide
#


def wide_resnet50(num_classes: int) -> ResNet:
    return resnet50(num_classes).to_wide()


def wide_resnet101(num_classes: int) -> ResNet:
    return resnet101(num_classes).to_wide()
