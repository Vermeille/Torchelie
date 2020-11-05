from functools import partial
import torch.nn as nn
import torchelie.nn as tnn
from torchelie.utils import kaiming, xavier

from .classifier import Classifier2, ConcatPoolClassifier1


def VggBNBone(arch, in_ch=3, leak=0, block=tnn.Conv2dBNReLU, debug=False):
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
            layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
        else:
            layers.append(block(in_ch, layer, ks=3, leak=leak))
            in_ch = layer
        if debug:
            layer_name = 'layer_{}_{}'.format(layers[-1].__class__.__name__, i)
            layers.append(tnn.Debug(layer_name))
    return tnn.CondSeq(*layers)

def VggDebugFullyConv(num_classes, in_ch=3, input_size=32):
    """
    A stack of vggnet layers.

    Args:
        num_classes: (int): write your description
        in_ch: (int): write your description
        input_size: (int): write your description
    """
    layers = [32, 'A', 64, 'A', 128, 'A', 256]

    input_size = input_size // 32

    while input_size != 1:
        layers += ['A', min(1024, layers[-1] * 2)]
        input_size = input_size // 2

    return ConcatPoolClassifier1(
        VggBNBone(layers, in_ch=in_ch), layers[-1], num_classes)

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
    layers = [64, 'M', 128, 'M', 128, 'M', 256]
    return Classifier2(
        VggBNBone(layers,
                  in_ch=in_ch,
                  debug=debug,
                  block=partial(tnn.Conv2dBNReLU,inplace=True)),
        layers[-1], num_classes)


def VggGeneratorDebug(in_noise=32, out_ch=3, out_sz=32):
    """
    A not so small Vgg net image GAN generator for testing purposes

    Args:
        in_noise (int): dimension of the input noise
        out_ch (int): number of output channels (3 for RGB images)

    Returns:
        a VGG instance
    """
    layers = [256, 256, 'U', 128, 128, 'U', 64, 64, 'U', 32, 32]
    out_sz = out_sz // 32
    while out_sz != 1:
        ch = min(1024, 2*layers[0])
        layers = [ch, ch, 'U'] + layers
        out_sz = out_sz // 2
    return nn.Sequential(
        kaiming(nn.Linear(in_noise, layers[0] * 4 * 4)), tnn.Reshape(-1, 4, 4),
        nn.LeakyReLU(0.2, inplace=True),
        VggBNBone(layers, in_ch=layers[0]),
        xavier(tnn.Conv1x1(layers[-1], out_ch)), nn.Sigmoid())


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
        """
        Initialize the convolution layer.

        Args:
            self: (todo): write your description
            in_noise: (int): write your description
            out_ch: (str): write your description
            side_ch: (int): write your description
        """
        super(VggImg2ImgGeneratorDebug, self).__init__()

        def make_block(in_ch, out_ch, **kwargs):
            """
            Build a block of_chade2d.

            Args:
                in_ch: (int): write your description
                out_ch: (todo): write your description
            """
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
        """
        Initialize kwargs

        Args:
            self: (todo): write your description
            in_noise: (int): write your description
            out_ch: (str): write your description
            num_classes: (int): write your description
        """
        super(VggClassCondGeneratorDebug, self).__init__()

        def make_block(in_ch, out_ch, **kwargs):
            """
            Parameters ---------- block

            Args:
                in_ch: (int): write your description
                out_ch: (todo): write your description
            """
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
