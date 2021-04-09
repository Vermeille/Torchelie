import torch
import functools
import torch.nn as nn
import torchelie.nn as tnn
import torchelie.utils as tu

from typing import List, Callable
from typing_extensions import Literal

from .classifier import Classifier2, Classifier1


def _preact_head(in_ch: int, out_ch: int, input_size: int = 224) -> nn.Module:
    if input_size <= 64:
        return tu.kaiming(tnn.Conv2d(in_ch, out_ch, ks=3))
    elif input_size <= 128:
        return tu.kaiming(tnn.Conv2d(in_ch, out_ch, ks=5, stride=2))
    else:
        return tnn.CondSeq(
            tu.kaiming(tnn.Conv2d(in_ch, out_ch, ks=7, stride=2)),
            nn.MaxPool2d(3, 2, 1))


def _head(in_ch: int, out_ch: int, input_size: int = 224) -> nn.Module:
    if input_size <= 64:
        return tnn.Conv2dBNReLU(in_ch, out_ch, 3).remove_bn()
    elif input_size <= 128:
        return tnn.Conv2dBNReLU(in_ch, out_ch, 5, stride=2).remove_bn()
    else:
        h = tnn.Conv2dBNReLU(in_ch, out_ch, 7, stride=2).remove_bn()
        h.add_module('pool', nn.MaxPool2d(3, 2, 1))
        return h


def ResNet(arch: List[str],
           block: Literal['basic', 'bottleneck', 'preact_basic',
                          'preact_bottleneck'], input_size: int,
           in_channels: int, num_classes: int) -> nn.Module:
    """
    A resnet

    How to specify an architecture:

    It's a list of block specifications. Each element is a string of the form
    "output channels:stride". For instance "64:2" is a block with input stride
    2 and 64 output channels.

    Args:
        arch (list): the architecture specification
        block (fn): the residual block to use ctor

    Returns:
        A Resnet instance
    """
    def parse(l: str) -> List[int]:
        return [int(x) for x in l.split(':')]

    assert block in [
        'basic', 'bottleneck', 'preact_basic', 'preact_bottleneck'
    ]

    b = ({
        'basic': tnn.ResBlock,
        'bottleneck': tnn.ResBlockBottleneck,
        'preact_basic': tnn.PreactResBlock,
        'preact_bottleneck': tnn.PreactResBlockBottleneck
    })[block]
    layers = []

    if 'preact' in block:
        layers.append(_preact_head(in_channels, parse(arch[0])[0], input_size))
    else:
        layers.append(_head(in_channels, parse(arch[0])[0], input_size))

    in_ch = parse(arch[0])[0]
    for i, layer in enumerate(arch):
        if layer == 'U':
            layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
        else:
            ch, s = parse(layer)
            layers.append(b(in_ch, ch, stride=s))
            in_ch = ch

    if 'preact' in block:
        assert isinstance(layers[1],
                          (tnn.PreactResBlock, tnn.PreactResBlockBottleneck))
        layers[1].preact_skip()
        layers.append(nn.BatchNorm2d(ch))
        layers.append(nn.ReLU(True))

    return Classifier1(tnn.CondSeq(*layers), ch, num_classes)


def resnet20_cifar(num_classes, in_channels=3, input_size=224) -> nn.Module:
    return ResNet([
        '16:1', '16:1', '16:1', '32:2', '32:1', '32:1', '64:2', '64:1', '64:1'
    ],
                  'basic',
                  input_size=input_size,
                  in_channels=in_channels,
                  num_classes=num_classes)


def preact_resnet20_cifar(num_classes: int,
                          in_channels: int = 3,
                          input_size: int = 224) -> nn.Module:
    return ResNet([
        '16:1', '16:1', '16:1', '32:2', '32:1', '32:1', '64:2', '64:1', '64:1'
    ],
                  'preact_basic',
                  input_size=input_size,
                  in_channels=in_channels,
                  num_classes=num_classes)


def resnet18(num_classes: int,
             in_channels: int = 3,
             input_size: int = 224) -> nn.Module:
    return ResNet(
        ['64:1', '64:1', '128:2', '128:1', '256:2', '256:1', '512:2', '512:1'],
        'basic',
        input_size=input_size,
        in_channels=in_channels,
        num_classes=num_classes)


def resnet50(num_classes: int,
             in_channels: int = 3,
             input_size: int = 224) -> nn.Module:
    return ResNet(['64:1'] * 3 + ['128:2'] + ['128:1'] * 3 + ['256:2'] +
                  ['256:1'] * 5 + ['512:2', '512:1', '512:1'],
                  'bottleneck',
                  input_size=input_size,
                  in_channels=in_channels,
                  num_classes=num_classes)

def resnext50(num_classes:int,
        in_channels: int=3,
        input_size:int=224)->nn.Module:
    m = resnet50(num_classes, in_channels, input_size)
    for block in m.modules():
        if isinstance(block, tnn.ResBlockBottleneck):
            block.to_resnext()
    return m

def preact_resnext50(num_classes:int,
        in_channels: int=3,
        input_size:int=224)->nn.Module:
    m = preact_resnet50(num_classes, in_channels, input_size)
    for block in m.modules():
        if isinstance(block, tnn.PreactResBlockBottleneck):
            block.to_resnext()
    return m


def preact_resnet50(num_classes: int,
                    in_channels: int = 3,
                    input_size: int = 224) -> nn.Module:
    return ResNet(['64:1'] * 3 + ['128:2'] + ['128:1'] * 3 + ['256:2'] +
                  ['256:1'] * 5 + ['512:2', '512:1', '512:1'],
                  'preact_bottleneck',
                  input_size=input_size,
                  in_channels=in_channels,
                  num_classes=num_classes)


def preact_resnet18(num_classes: int,
                    in_channels: int = 3,
                    input_size: int = 224) -> nn.Module:
    return ResNet(
        ['64:1', '64:1', '128:2', '128:1', '256:2', '256:1', '512:2', '512:1'],
        'preact_basic',
        input_size=input_size,
        in_channels=in_channels,
        num_classes=num_classes)


def preact_resnet34(num_classes: int,
                    in_channels: int = 3,
                    input_size: int = 224) -> nn.Module:
    return ResNet(['64:1'] * 3 + ['128:2'] + ['128:1'] * 3 + ['256:2'] +
                  ['256:1'] * 5 + ['512:2', '512:1', '512:1'],
                  'preact_basic',
                  in_channels=in_channels,
                  input_size=input_size,
                  num_classes=num_classes)
