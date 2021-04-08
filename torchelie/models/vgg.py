from functools import partial
import torch.nn as nn
import torchelie.nn as tnn
from torchelie.utils import kaiming, xavier
from typing import List, Union, cast

from .classifier import Classifier2, ConcatPoolClassifier1


def VggBNBone(arch, in_ch: int = 3, debug: bool = False) -> tnn.CondSeq:
    """
    Construct a VGG net

    How to specify a VGG architecture:

    It's a list of blocks specifications. Blocks are either:

    - 'M' for maxpool of kernel size 2 and stride 2
    - 'A' for average pool of kernel size 2 and stride 2
    - 'U' for bilinear upsampling (scale factor 2)
    - an integer `ch` for a block with `ch` output channels

    Args:
        arch (list): architecture specification
        in_ch (int): number of input channels

    Returns:
        A VGG instance
    """
    layers: List[nn.Module] = []

    if debug:
        layers.append(tnn.Debug('Input'))

    for i, layer in enumerate(arch):
        if layer == 'M':
            layers.append(nn.MaxPool2d(2, 2))
        elif layer == 'A':
            layers.append(nn.AvgPool2d(2, 2))
        elif layer == 'U':
            layers.append(tnn.InterpolateBilinear2d(scale_factor=2))
        else:
            layers.append(tnn.Conv2dBNReLU(in_ch, layer, kernel_size=3))
            in_ch = layer
        if debug:
            layer_name = 'layer_{}_{}'.format(layers[-1].__class__.__name__, i)
            layers.append(tnn.Debug(layer_name))
    return tnn.CondSeq(*layers)


def VggDebugFullyConv(num_classes: int,
                      in_ch: int = 3,
                      input_size: int = 32) -> nn.Module:
    layers: List[Union[str, int]] = [32, 'A', 64, 'A', 128, 'A', 256]

    input_size = input_size // 32

    while input_size != 1:
        layers += ['A', min(1024, layers[-1] * 2)]
        input_size = input_size // 2

    return ConcatPoolClassifier1(VggBNBone(layers, in_ch=in_ch),
                                 cast(int, layers[-1]), num_classes)


def VggDebug(num_classes: int, in_ch=1, debug: bool = False) -> nn.Module:
    """
    A not so small Vgg net classifier for testing purposes

    Args:
        num_classes (int): number of output classes
        in_ch (int): number of input channels, 3 for RGB images
        debug (bool): whether to add debug layers

    Returns:
        a VGG instance
    """
    layers = [64, 'M', 128, 'M', 128, 'M', 256]
    return Classifier2(VggBNBone(layers, in_ch=in_ch, debug=debug),
                       cast(int, layers[-1]), num_classes)
