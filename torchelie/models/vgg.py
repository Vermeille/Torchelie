import torch.nn as nn
import torchelie.nn as tnn
from torchelie.utils import kaiming, xavier

from .classifier import Classifier


def VggBNBone(arch, in_ch=3, leak=0, block=tnn.Conv2dBNReLU, debug=False):
    """
    Construct a VGG net

    How to specify a VGG architecture:

    It's a list of blocks specifications. Blocks are either:

    - 'M' for maxpool of kernel size 2 and stride 2
    - 'A' for average pool of kernel size 2 and stride 2
    - 'U' for nearest neighbors upsampling (scale factor 2)
    - an integer `ch` for a block with `ch` output channels

    Args:
        arch (list): architecture specification
        in_ch (int): number of input channels
        leak (float): leak in relus
        block (fn): block ctor

    Returns:
        A VGG instance
    """
    layers = []

    if debug:
        layers.append(tnn.Debug('Input'))

    for i, layer in enumerate(arch):
        if layer == 'M':
            layers.append(nn.MaxPool2d(2, 2))
        elif layer == 'A':
            layers.append(nn.AvgPool2d(2, 2))
        elif layer == 'U':
            layers.append(nn.UpsamplingNearest2d(scale_factor=2))
        else:
            layers.append(block(in_ch, layer, ks=3, leak=leak))
            in_ch = layer
        if debug:
            layer_name = 'layer_{}_{}'.format(layers[-1].__class__.__name__, i)
            layers.append(tnn.Debug(layer_name))
    return tnn.CondSeq(*layers)


def VggDebug(num_classes, in_ch=1, debug=False):
    """
    A not so small Vgg net classifier for testing purposes

    Args:
        num_classes (int): number of output classes
        in_ch (int): number of input channels, 3 for RGB images
        debug (bool): whether to add debug layers

    Returns:
        a VGG instance
    """
    return Classifier(
        VggBNBone([64, 64, 'M', 128, 'M', 128, 'M', 256, 256],
                  in_ch=in_ch,
                  debug=debug), 256, num_classes)


def VggGeneratorDebug(in_noise=32, out_ch=3):
    """
    A not so small Vgg net image GAN generator for testing purposes

    Args:
        in_noise (int): dimension of the input noise
        out_ch (int): number of output channels (3 for RGB images)

    Returns:
        a VGG instance
    """
    return nn.Sequential(
        kaiming(nn.Linear(in_noise, 128 * 16)), tnn.Reshape(128, 4, 4),
        nn.LeakyReLU(0.2, inplace=True),
        VggBNBone([128, 'U', 64, 'U', 32, 'U', 16], in_ch=128),
        xavier(tnn.Conv1x1(16, 1)), nn.Sigmoid())


class VggImg2ImgGeneratorDebug(nn.Module):
    """
    A vgg based image decoder that decodes a latent / noise vector into an
    image, conditioned on another image, like Pix2Pix or GauGAN. This
    architecture is really close to GauGAN as it's not an encoder-decoder
    architecture and uses SPADE

    Args:
        in_noise (int): dimension of the input latent
        out_ch (int): number of channels of the output image, 3 for RGB image
        side_ch (int): number of channels of the input image, 3 for RGB image
    """
    def __init__(self, in_noise, out_ch, side_ch=1):
        super(VggImg2ImgGeneratorDebug, self).__init__()

        def make_block(in_ch, out_ch, **kwargs):
            return tnn.Conv2dNormReLU(in_ch,
                                      out_ch,
                                      norm=lambda out: tnn.Spade2d(out, side_ch, 64),
                                      **kwargs)

        self.net = tnn.CondSeq(
            kaiming(nn.Linear(in_noise, 128 * 16)), tnn.Reshape(128, 4, 4),
            nn.LeakyReLU(0.2, inplace=True),
            VggBNBone([128, 'U', 64, 'U', 32, 'U', 16],
                      in_ch=128,
                      block=make_block), xavier(tnn.Conv1x1(16, out_ch)),
            nn.Sigmoid())

    def forward(self, x, y):
        """
        Generate an image

        Args:
            x (2D tensor): input latent vectors
            y (4D tensor): input images

        Returns:
            the generated images as a 4D tensor
        """
        return self.net(x, y)


class VggClassCondGeneratorDebug(nn.Module):
    """
    A vgg based image decoder that decodes a latent / noise vector into an
    image, conditioned on a class label (through conditional batchnorm).

    Args:
        in_noise (int): dimension of the input latent
        out_ch (int): number of channels of the output image, 3 for RGB image
        side_ch (int): number of channels of the input image, 3 for RGB image
    """
    def __init__(self, in_noise, out_ch, num_classes):
        super(VggClassCondGeneratorDebug, self).__init__()

        def make_block(in_ch, out_ch, **kwargs):
            return tnn.Conv2dNormReLU(in_ch,
                                      out_ch,
                                      norm=lambda out: tnn.ConditionalBN2d(out, 64),
                                      **kwargs)

        self.emb = nn.Embedding(num_classes, 64)
        self.net = tnn.CondSeq(
            kaiming(nn.Linear(in_noise, 128 * 16)), tnn.Reshape(128, 4, 4),
            nn.LeakyReLU(0.2, inplace=True),
            VggBNBone([128, 'U', 64, 'U', 32, 'U', 16],
                      in_ch=128,
                      block=make_block), xavier(tnn.Conv1x1(16, out_ch)),
            nn.Sigmoid())

    def forward(self, x, y):
        """
        Generate images

        Args:
            x (2D tensor): latent vectors
            y (1D tensor): class labels

        Returns:
            generated images as a 4D tensor
        """
        y_emb = self.emb(y)
        return self.net(x, y_emb)
