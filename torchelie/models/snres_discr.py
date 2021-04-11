import torch
import torch.nn as nn
import torchelie.nn as tnn
from typing import List, Union, Tuple, cast
from collections import OrderedDict
from .classifier import ClassificationHead, ProjectionDiscr


def _parse_snres(arch: List[Union[str, int]],
                 in_ch: int) -> Tuple[nn.Module, int]:
    assert isinstance(arch[0], int)
    blocks: List[nn.Module] = [
        nn.utils.spectral_norm(tnn.Conv3x3(in_ch, arch[0]))
    ]
    in_ch = arch[0]

    for x, x2 in zip(arch, arch[1:] + ['dummy']):
        if x == 'D':
            continue

        downsample = x2 == 'D'
        assert isinstance(x, int)
        blocks.append(tnn.SNResidualDiscrBlock(in_ch, x, downsample))
        in_ch = x
    blocks[1].preact_skip()
    return tnn.CondSeq(*blocks), in_ch


def snres_discr(num_classes: int,
                in_ch: int = 3,
                input_sz: int = 32,
                max_channels: int = 1024,
                base_ch: int = 32) -> nn.Module:
    ch = base_ch
    layers: List[Union[int, str]] = [
        ch, 'D',
        min(max_channels, ch * 2), 'D',
        min(max_channels, ch * 4), 'D'
    ]
    input_sz = input_sz // 32
    while input_sz != 1:
        layers += [min(max_channels, cast(int, layers[-2]) * 2), 'D']
        input_sz = input_sz // 2

    return snres_discr_ctor(layers, in_ch=in_ch, out_ch=num_classes)


def snres_discr_ctor(arch: List[Union[int, str]],
                     in_ch: int = 3,
                     out_ch: int = 1) -> nn.Module:
    """
    Make a resnet discriminator with spectral norm, using
    `SNResidualDiscrBlock`.

    Args:
        arch (list): a list of ints to specify output channels of the blocks,
            and 'D' to downsample. Example: `[32, 'D', 64, 'D']`
        in_ch (int): number of input channels
        out_ch (int): number of output channels

    Returns:
        an instance
    """
    bone, in_ch = _parse_snres(arch, in_ch)
    bone.add_module('final_relu', nn.LeakyReLU(0.2, True))
    clf = ClassificationHead(in_ch, out_ch)

    for m in clf.modules():
        if isinstance(m, nn.Linear):
            nn.utils.spectral_norm(m)

    return tnn.CondSeq(OrderedDict([('bone', bone), ('head', clf)]))


def snres_projdiscr(arch: List[Union[int, str]],
                    num_classes: int,
                    in_ch: int = 3) -> nn.Module:
    """
    Make a resnet discriminator with spectral norm and projection, using
    `SNResidualDiscrBlock`.

    Args:
        arch (list): a list of ints to specify output channels of the blocks,
            and 'D' to downsample. Example: `[32, 'D', 64, 'D']`
        in_ch (int): number of input channels
        num_classes (int): number of classes in the dataset

    Returns:
        an instance
    """
    bone, in_ch = _parse_snres(arch, in_ch)
    clf = ProjectionDiscr(bone, in_ch, num_classes=num_classes)

    for m in clf.head.modules():
        if isinstance(m, nn.Linear):
            nn.utils.spectral_norm(m)
    return clf


def snres_discr_4l(in_ch: int = 3, out_ch: int = 1) -> nn.Module:
    """
    Make a 4 layers resnet discriminator with spectral norm, using
    `SNResidualDiscrBlock`.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels

    Returns:
        an instance
    """
    return snres_discr_ctor(arch=[32, 'D', 64, 'D', 128, 'D', 256, 'D'],
                            in_ch=in_ch,
                            out_ch=out_ch)


def snres_projdiscr_4l(num_classes: int, in_ch: int = 3) -> nn.Module:
    """
    Make a 4 layers resnet discriminator with spectral norm and projection,
    using `SNResidualDiscrBlock`.

    Args:
        in_ch (int): number of input channels
        num_classes (int): number of classes in the dataset

    Returns:
        an instance
    """
    return snres_projdiscr(arch=[32, 'D', 64, 'D', 128, 'D', 256, 'D'],
                           in_ch=in_ch,
                           num_classes=num_classes)


def snres_discr_5l(in_ch: int = 3, out_ch: int = 1) -> nn.Module:
    """
    Make a 5 layers resnet discriminator with spectral norm, using
    `SNResidualDiscrBlock`.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels

    Returns:
        an instance
    """
    return snres_discr_ctor(
        arch=[32, 'D', 64, 'D', 128, 'D', 256, 'D', 512, 'D'],
        in_ch=in_ch,
        out_ch=out_ch)


def snres_discr_6l(in_ch: int = 3, out_ch: int = 1) -> nn.Module:
    """
    Make a 6 layers resnet discriminator with spectral norm, using
    `SNResidualDiscrBlock`.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels

    Returns:
        an instance
    """
    return snres_discr_ctor(
        arch=[32, 'D', 64, 'D', 128, 'D', 256, 'D', 512, 'D', 512, 'D'],
        in_ch=in_ch,
        out_ch=out_ch)


def snres_discr_7l(in_ch: int = 3, out_ch: int = 1) -> nn.Module:
    """
    Make a 7 layers resnet discriminator with spectral norm, using
    `SNResidualDiscrBlock`.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels

    Returns:
        an instance
    """
    return snres_discr_ctor(arch=[
        32, 'D', 64, 'D', 128, 'D', 256, 'D', 512, 'D', 512, 'D', 1024, 'D'
    ],
                            in_ch=in_ch,
                            out_ch=out_ch)


def snres_projdiscr_5l(num_classes: int, in_ch: int = 3) -> nn.Module:
    """
    Make a 5 layers resnet discriminator with spectral norm and projection,
    using `SNResidualDiscrBlock`.

    Args:
        in_ch (int): number of input channels
        num_classes (int): number of classes in the dataset

    Returns:
        an instance
    """
    return snres_projdiscr(
        arch=[32, 'D', 64, 'D', 128, 'D', 256, 'D', 512, 'D'],
        in_ch=in_ch,
        num_classes=num_classes)
