import torch.nn as nn
import torchelie.nn as tnn
import torchelie.utils as tu
from torchelie.utils import kaiming, xavier
from .classifier import Classifier2, ProjectionDiscr
from typing import Callable, Optional, List


def base_patch_discr(arch: List[int],
                     in_ch: int = 3,
                     out_ch: int = 1) -> tnn.CondSeq:

    layers = [
        tnn.Conv2dBNReLU(in_ch, arch[0], kernel_size=4,
                         stride=2).remove_bn().leaky()
    ]
    layers[0].norm = nn.Identity()

    in_ch = arch[0]
    for next_ch in arch[1:]:
        layers.append(
            tnn.Conv2dBNReLU(in_ch, next_ch, kernel_size=4, stride=2).leaky())
        in_ch = next_ch
    layers.append(tnn.Conv1x1(in_ch, out_ch))

    return tnn.CondSeq(*layers)


def patch_discr(arch: List[int], in_ch: int = 3, out_ch: int = 1) -> nn.Module:
    """
    Construct a PatchGAN discriminator

    Args:
        arch (list of ints): a list of number of filters. For instance `[64,
            128, 256]` generates a PatchGAN with 3 conv layers, with respective
            number of kernels 64, 128 and 256.
        in_ch (int): number of input channels, 3 for RGB images
        out_ch (int): number of output channels, 1 for fake / real
            discriminator

    Returns:
        the specified patchGAN as CondSeq
    """
    return base_patch_discr(arch, in_ch=in_ch, out_ch=out_ch)


def Patch286(in_ch: int = 3, out_ch: int = 1) -> nn.Module:
    """
    Patch Discriminator from pix2pix

    Args:
        in_ch (int): input channels, 3 for pictures
        out_ch (int): output channels, 1 for binary real / fake classification
    """
    return patch_discr([64, 128, 256, 512, 512, 512],
                       in_ch=in_ch,
                       out_ch=out_ch)


def Patch70(in_ch: int = 3, out_ch: int = 1) -> nn.Module:
    """
    Patch Discriminator from pix2pix

    Args:
        in_ch (int): input channels, 3 for pictures
        out_ch (int): output channels, 1 for binary real / fake classification
    """
    return patch_discr([64, 128, 256, 512], in_ch=in_ch, out_ch=out_ch)


# Not sure about the receptive field but ok
def Patch32(in_ch: int = 3, out_ch: int = 1) -> nn.Module:
    """
    Patch Discriminator from pix2pix

    Args:
        in_ch (int): input channels, 3 for pictures
        out_ch (int): output channels, 1 for binary real / fake classification
        norm (function): the normalization layer to use
    """
    return patch_discr([64, 128, 256], in_ch=in_ch, out_ch=out_ch)


def Patch16(in_ch: int = 3, out_ch: int = 1) -> nn.Module:
    """
    Patch Discriminator from pix2pix

    Args:
        in_ch (int): input channels, 3 for pictures
        out_ch (int): output channels, 1 for binary real / fake classification
        norm (function): the normalization layer to use
    """
    return patch_discr([64, 128], in_ch=in_ch, out_ch=out_ch)
