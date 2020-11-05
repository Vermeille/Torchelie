import torch.nn as nn

import torchelie.nn as tnn
from torchelie.utils import kaiming, xavier
from .classifier import Classifier2, ProjectionDiscr


def base_patch_discr(arch, in_ch=3, out_ch=1, norm=None):
    """
    Patch a layer norm.

    Args:
        arch: (todo): write your description
        in_ch: (int): write your description
        out_ch: (todo): write your description
        norm: (todo): write your description
    """
    def block(in_ch, out_ch, norm):
        """
        Block block.

        Args:
            in_ch: (int): write your description
            out_ch: (todo): write your description
            norm: (todo): write your description
        """
        if norm is None:
            return [
                kaiming(nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
                        a=0.2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        else:
            return [
                kaiming(nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
                        a=0.2),
                norm(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ]

    layers = block(in_ch, arch[0], None)

    in_ch = arch[0]
    for out_ch in arch[1:]:
        layers += block(in_ch, out_ch, norm)
        in_ch = out_ch

    return tnn.CondSeq(*layers)


def patch_discr(arch, in_ch=3, out_ch=1, norm=None):
    """
    Construct a PatchGAN discriminator

    Args:
        arch (list of ints): a list of number of filters. For instance `[64,
            128, 256]` generates a PatchGAN with 3 conv layers, with respective
            number of kernels 64, 128 and 256.
        in_ch (int): number of input channels, 3 for RGB images
        out_ch (int): number of output channels, 1 for fake / real
            discriminator
        norm (fn): a normalization layer ctor

    Returns:
        the specified patchGAN as CondSeq
    """
    return Classifier2(base_patch_discr(arch, in_ch=in_ch, out_ch=out_ch, norm=norm),
                      arch[-1], out_ch)


def proj_patch_discr(arch, num_classes, in_ch=3, out_ch=1, norm=None):
    """
    Construct a PatchGAN discriminator with projection

    Args:
        arch (list of ints): a list of number of filters. For instance `[64,
            128, 256]` generates a PatchGAN with 3 conv layers, with respective
            number of kernels 64, 128 and 256.
        num_classes (int): number of classes to discriminate
        in_ch (int): number of input channels, 3 for RGB images
        out_ch (int): number of output channels, 1 for fake / real
            discriminator
        norm (fn): a normalization layer ctor

    Returns:
        the specified patchGAN as CondSeq
    """
    return ProjectionDiscr(base_patch_discr(arch, in_ch=in_ch, out_ch=out_ch, norm=norm),
                      arch[-1], num_classes=num_classes)


def Patch286(in_ch=3, out_ch=1, norm=nn.BatchNorm2d):
    """
    Patch Discriminator from pix2pix

    Args:
        in_ch (int): input channels, 3 for pictures
        out_ch (int): output channels, 1 for binary real / fake classification
        norm (function): the normalization layer to use
    """
    return patch_discr([64, 128, 256, 512, 512, 512],
                       in_ch=in_ch,
                       out_ch=out_ch,
                       norm=norm)


def Patch70(in_ch=3, out_ch=1, norm=nn.BatchNorm2d):
    """
    Patch Discriminator from pix2pix

    Args:
        in_ch (int): input channels, 3 for pictures
        out_ch (int): output channels, 1 for binary real / fake classification
        norm (function): the normalization layer to use
    """
    return patch_discr([64, 128, 256, 512],
                       in_ch=in_ch,
                       out_ch=out_ch,
                       norm=norm)


# Not sure about the receptive field but ok
def Patch32(in_ch=3, out_ch=1, norm=nn.BatchNorm2d):
    """
    Patch Discriminator from pix2pix

    Args:
        in_ch (int): input channels, 3 for pictures
        out_ch (int): output channels, 1 for binary real / fake classification
        norm (function): the normalization layer to use
    """
    return patch_discr([64, 128, 256], in_ch=in_ch, out_ch=out_ch, norm=norm)


# Not sure about the receptive field but ok
def ProjPatch32(in_ch=3, out_ch=1, norm=nn.BatchNorm2d, num_classes=10):
    """
    Patch Discriminator from pix2pix, with projection for conditional GANs

    Args:
        in_ch (int): input channels, 3 for pictures
        out_ch (int): output channels, 1 for binary real / fake classification
        norm (function): the normalization layer to use
        num_classes (int): how many classes to discriminate
    """
    return proj_patch_discr([64, 128, 256], num_classes=num_classes,
                            in_ch=in_ch,
                            out_ch=out_ch,
                            norm=norm)


def Patch16(in_ch=3, out_ch=1, norm=nn.BatchNorm2d):
    """
    Patch Discriminator from pix2pix

    Args:
        in_ch (int): input channels, 3 for pictures
        out_ch (int): output channels, 1 for binary real / fake classification
        norm (function): the normalization layer to use
    """
    return patch_discr([64, 128], in_ch=in_ch, out_ch=out_ch, norm=norm)
