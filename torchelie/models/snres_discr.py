import torch
import torch.nn as nn
import torchelie.nn as tnn
from .classifier import Classifier, ProjectionDiscr


def _parse_snres(arch, in_ch):
    blocks = []
    for x, x2 in zip(arch, arch[1:] + ['dummy']):
        if x == 'D':
            continue

        downsample = x2 == 'D'
        blocks.append(tnn.SNResidualDiscrBlock(in_ch, x, downsample))
        in_ch = x
    return tnn.CondSeq(*blocks), in_ch


def snres_discr(arch, in_ch=3, out_ch=1):
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
    clf = Classifier(bone, in_ch, out_ch)

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
    return snres_discr(arch=[32, 'D', 64, 'D', 128, 'D', 256, 'D'],
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
    return snres_discr(arch=[32, 'D', 64, 'D', 128, 'D', 256, 'D'],
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
    return snres_discr(arch=[32, 'D', 64, 'D', 128, 'D', 256, 'D', 512, 'D'],
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
    return snres_discr(arch=[32, 'D', 64, 'D', 128, 'D', 256, 'D', 512, 'D'],
                       in_ch=in_ch,
                       num_classes=num_classes)
