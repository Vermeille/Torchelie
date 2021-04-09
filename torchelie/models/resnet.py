import torch
import functools
import torch.nn as nn
import torchelie.nn as tnn
import torchelie.utils as tu

from typing import List, Callable, Optional
from typing_extensions import Literal

from .classifier import Classifier2, Classifier1

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

    def input_specs(self, input_size: int, in_channels=3) -> 'ResNetInput':
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
        self.restype = 'basic'

        def parse(l: str) -> List[int]:
            return [int(x) for x in l.split(':')]

        in_ch = parse(arch[0])[0]
        self.in_channels = in_ch

        feats = tnn.CondSeq()
        feats.add_module('input', ResNetInput(3, in_ch))

        for i, layer in enumerate(arch):
            ch, s = parse(layer)
            feats.add_module(f'block_{i}', tnn.ResBlock(in_ch, ch, stride=s))
            in_ch = ch

        self.features = feats
        self.out_channels = ch

        self.classifier = Classifier1(None, ch, num_classes, 0).head

    def _change_block_type(self, ty: str) -> None:
        Type = ({
            'basic': tnn.ResBlock,
            'bottleneck': tnn.ResBlockBottleneck,
            'preact_basic': tnn.PreactResBlock,
            'preact_bottleneck': tnn.PreactResBlockBottleneck,
            'resnext': tnn.ResBlockBottleneck,
            'preact_resnext': tnn.PreactResBlockBottleneck,
        })[ty]

        for nm, m in self.features.named_children():
            if not isinstance(m, BLOCKS):
                continue
            new_block = Type(m.in_channels, m.out_channels, m.stride)
            if 'resnext' in ty:
                assert isinstance(new_block, BOTTLENECKS_BLOCKS)
                new_block.resnext()
            setattr(self.features, nm, new_block)

    def bottleneck(self) -> 'ResNet':
        if 'bottleneck' in self.restype:
            return self

        if 'preact' in self.restype:
            self.restype = 'preact_bottleneck'
        else:
            self.restype = 'bottleneck'

        self._change_block_type(self.restype)
        if 'preact' in self.restype:
            assert isinstance(self.features.block_0, PREACT_BLOCKS)
            self.features.block_0.no_preact()
        return self

    def preact(self) -> 'ResNet':
        if 'preact' in self.restype:
            return self

        self.restype = 'preact_' + self.restype
        self._change_block_type(self.restype)
        assert isinstance(self.features.block_0,
                          (tnn.PreactResBlock, tnn.PreactResBlockBottleneck))
        self.features.block_0.no_preact()
        self.features.add_module('final_bn', nn.BatchNorm2d(self.out_channels))
        self.features.add_module('final_relu', nn.ReLU(True))
        return self

    def resnext(self) -> 'ResNet':
        if 'resnext' in self.restype:
            return self

        if 'preact' in self.restype:
            self.restype = 'preact_resnext'
        else:
            self.restype = 'resnext'

        self._change_block_type(self.restype)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def input_specs(self,
                    input_size: int = 224,
                    in_channels: int = 3) -> 'ResNet':
        assert isinstance(self.features.input, ResNetInput)
        self.features.input.input_specs(input_size=input_size,
                                        in_channels=in_channels)
        return self


###
### CIFAR
###


def resnet20_cifar() -> ResNet:
    return ResNet([
        '16:1', '16:1', '16:1', '32:2', '32:1', '32:1', '64:2', '64:1', '64:1'
    ], 10).input_specs(input_size=32)


def preact_resnet20_cifar() -> ResNet:
    return resnet20_cifar().preact()


###
### ResNets
###


def resnet18(num_classes: int) -> ResNet:
    return ResNet(
        ['64:1', '64:1', '128:2', '128:1', '256:2', '256:1', '512:2', '512:1'],
        num_classes)


def resnet34(num_classes: int) -> ResNet:
    return ResNet(['64:1'] * 3 + ['128:2'] + ['128:1'] * 3 + ['256:2'] +
                  ['256:1'] * 5 + ['512:2', '512:1', '512:1'], num_classes)


def resnet50(num_classes: int) -> ResNet:
    net = ResNet(['256:1'] * 3 + ['512:2'] + ['512:1'] * 3 + ['1024:2'] +
                 ['1024:1'] * 5 + ['2048:2', '2048:1', '2048:1'], num_classes)
    net.bottleneck()
    return net


def resnet101(num_classes: int) -> ResNet:
    net = ResNet(['256:1'] * 3 + ['512:2'] + ['512:1'] * 3 + ['1024:2'] +
                 ['1024:1'] * 22 + ['2048:2', '2048:1', '2048:1'], num_classes)
    net.bottleneck()
    return net


def resnet152(num_classes: int) -> ResNet:
    net = ResNet(['256:1'] * 3 + ['512:2'] + ['512:1'] * 7 + ['1024:2'] +
                 ['1024:1'] * 35 + ['2048:2', '2048:1', '2048:1'], num_classes)
    net.bottleneck()
    return net


###
### Preact ResNet
###


def preact_resnet18(num_classes: int) -> ResNet:
    return resnet18(num_classes).preact()


def preact_resnet34(num_classes: int) -> ResNet:
    return resnet34(num_classes).preact()


def preact_resnet50(num_classes: int) -> ResNet:
    return resnet50(num_classes).preact()


def preact_resnet101(num_classes: int) -> ResNet:
    return resnet101(num_classes).preact()


def preact_resnet152(num_classes: int) -> ResNet:
    return resnet152(num_classes).preact()


###
### ResNeXt
###


def resnext50_32x4d(num_classes: int) -> ResNet:
    return resnet50(num_classes).resnext()


def resnext101_32x4d(num_classes: int) -> ResNet:
    return resnet101(num_classes).resnext()


def resnext152_32x4d(num_classes: int) -> ResNet:
    return resnet152(num_classes).resnext()


###
### Preact ResNeXt
###


def preact_resnext50_32x4d(num_classes: int) -> ResNet:
    return preact_resnet50(num_classes).resnext()


def preact_resnext101_32x4d(num_classes: int) -> ResNet:
    return preact_resnet101(num_classes).resnext()


def preact_resnext152_32x4d(num_classes: int) -> ResNet:
    return preact_resnet152(num_classes).resnext()
