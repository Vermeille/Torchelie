import torch
import torch.nn as nn
import torchelie.nn as tnn
from .classifier import Classifier2, ProjectionDiscr, ConcatPoolClassifier1


def _parse_snres(arch, in_ch):
    blocks = [nn.utils.spectral_norm(tnn.Conv3x3(3, in_ch))]
    for x, x2 in zip(arch, arch[1:] + ['dummy']):
        if x == 'D':
            continue

        downsample = x2 == 'D'
        blocks.append(tnn.SNResidualDiscrBlock(in_ch, x, downsample))
        in_ch = x
    return tnn.CondSeq(*blocks), in_ch


def snres_discr(num_classes, in_ch=3, input_sz=32, max_channels=1024):
    layers = [32, 'D', 64, 'D', 128, 'D']
    input_sz = input_sz // 32
    while input_sz != 1:
        layers += [min(max_channels, layers[-2] * 2), 'D']
        input_sz = input_sz // 2

    return snres_discr_ctor(layers, in_ch=in_ch, out_ch=num_classes)


def snres_discr_ctor(arch, in_ch=3, out_ch=1):
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
    clf = ConcatPoolClassifier1(bone, in_ch, out_ch, dropout=0.)

    for m in clf.head.modules():
        if isinstance(m, nn.Linear):
            nn.utils.spectral_norm(m)

    return clf


def snres_projdiscr(arch, num_classes, in_ch=3):
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


def snres_discr_4l(in_ch=3, out_ch=1):
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


def snres_projdiscr_4l(num_classes, in_ch=3):
    """
    Make a 4 layers resnet discriminator with spectral norm and projection,
    using `SNResidualDiscrBlock`.

    Args:
        in_ch (int): number of input channels
        num_classes (int): number of classes in the dataset

    Returns:
        an instance
    """
    return snres_discr_ctor(arch=[32, 'D', 64, 'D', 128, 'D', 256, 'D'],
                       in_ch=in_ch,
                       num_classes=num_classes)


def snres_discr_5l(in_ch=3, out_ch=1):
    """
    Make a 5 layers resnet discriminator with spectral norm, using
    `SNResidualDiscrBlock`.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels

    Returns:
        an instance
    """
    return snres_discr_ctor(arch=[32, 'D', 64, 'D', 128, 'D', 256, 'D', 512, 'D'],
                       in_ch=in_ch,
                       out_ch=out_ch)


def snres_discr_6l(in_ch=3, out_ch=1):
    """
    Make a 6 layers resnet discriminator with spectral norm, using
    `SNResidualDiscrBlock`.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels

    Returns:
        an instance
    """
    return snres_discr_ctor(arch=[32, 'D', 64, 'D', 128, 'D', 256, 'D', 512, 'D',
        512, 'D'],
                       in_ch=in_ch,
                       out_ch=out_ch)

def snres_discr_7l(in_ch=3, out_ch=1):
    """
    Make a 7 layers resnet discriminator with spectral norm, using
    `SNResidualDiscrBlock`.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels

    Returns:
        an instance
    """
    return snres_discr_ctor(arch=[32, 'D', 64, 'D', 128, 'D', 256, 'D', 512, 'D',
        512, 'D', 1024, 'D'],
                       in_ch=in_ch,
                       out_ch=out_ch)


def snres_projdiscr_5l(num_classes, in_ch=3):
    """
    Make a 5 layers resnet discriminator with spectral norm and projection,
    using `SNResidualDiscrBlock`.

    Args:
        in_ch (int): number of input channels
        num_classes (int): number of classes in the dataset

    Returns:
        an instance
    """
    return snres_discr_ctor(arch=[32, 'D', 64, 'D', 128, 'D', 256, 'D', 512, 'D'],
                       in_ch=in_ch,
                       num_classes=num_classes)
