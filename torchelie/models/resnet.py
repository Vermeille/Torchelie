import functools
import torch.nn as nn
import torchelie.nn as tnn

from .classifier import Classifier


def VectorCondResNetBone(arch, head, hidden, in_ch=3, debug=False):
    """
    A resnet with vector side condition.

    Args:
        arch (list): the architecture specification
        head (fn): the module ctor to build for the first conv
        hidden (int): the hidden size of condition projection
        in_ch (int): number of input channels, 3 for RGB images
        debug (bool): should insert debug layers between each layer

    Returns:
        A Resnet instance
    """
    block_ctor = functools.partial(tnn.ConditionalResBlock, hidden=hidden)
    return ResNetBone(arch, head, block_ctor, in_ch, debug)


def VectorCondResNetDebug(vector_size, in_ch=3, debug=False):
    """
    A not so big predefined resnet classifier for debugging purposes.

    Args:
        vector_size (int): size of the conditioning vector
        in_ch (int): number of input channels, 3 for RGB images
        debug (bool): whereas to print additional debug info

    Returns:
        a resnet instance
    """
    return VectorCondResNetBone(
            ['64:2', '64:1', '64:1', '128:2', '128:1', '256:2', '256:1'],
            tnn.Conv2d,
            vector_size,
            in_ch=in_ch,
            debug=debug)


class ClassCondResNetBone(nn.Module):
    """
    A resnet with class side condition.

    Args:
        arch (list): the architecture specification
        head (fn): the module ctor to build for the first conv
        hidden (int): the hidden size of the side label embedding
        num_classes (int): the number of possible labels in the side condition
        in_ch (int): number of input channels, 3 for RGB images
        debug (bool): should insert debug layers between each layer

    Returns:
        A Resnet instance
    """
    def __init__(self, arch, head, hidden, num_classes, in_ch=3, debug=False):
        super(ClassCondResNetBone, self).__init__()
        block_ctor = functools.partial(tnn.ConditionalResBlock, hidden=hidden)
        self.bone = ResNetBone(arch, tnn.Conv2d, block_ctor, in_ch, debug)
        self.emb = nn.Embedding(num_classes, hidden)

    def forward(self, x, y):
        y_emb = self.emb(y)
        return self.bone(x, y_emb)


def ClassCondResNetDebug(num_classes, num_cond_classes, in_ch=3, debug=False):
    """
    A not so big predefined resnet classifier for debugging purposes.

    Args:
        num_cond_classes (int): the number of possible labels in the side condition
        num_classes (int): the number of output classes
        in_ch (int): number of input channels, 3 for RGB images
        debug (bool): whereas to print additional debug info

    Returns:
        a resnet instance
    """
    return Classifier(
        ClassCondResNetBone(
            ['64:2', '64:1', '64:1', '128:2', '128:1', '256:2', '256:1'],
            tnn.Conv2dBNReLU,
            64,
            num_cond_classes,
            in_ch=in_ch,
            debug=debug), 256, num_classes)


def ResNetBone(arch, head, block, in_ch=3, debug=False):
    """
    A resnet

    Args:
        arch (list): the architecture specification
        head (fn): the module ctor to build for the first conv
        block (fn): the residual block to use ctor
        in_ch (int): number of input channels, 3 for RGB images
        debug (bool): should insert debug layers between each layer

    Returns:
        A Resnet instance
    """
    def parse(l):
        return [int(x) for x in l.split(':')]

    layers = []

    if debug:
        layers.append(tnn.Debug('Input'))

    ch, s = parse(arch[0])
    layers.append(head(in_ch, ch, 3, stride=s))
    if debug:
        layers.append(tnn.Debug('Head'))
    in_ch = ch
    for i, (ch, s) in enumerate(map(parse, arch[1:])):
        layers.append(block(in_ch, ch, stride=s))
        in_ch = ch
        if debug:
            layer_name = 'layer_{}_{}'.format(layers[-1].__class__.__name__, i)
            layers.append(tnn.Debug(layer_name))
    return tnn.CondSeq(*layers)


def ResNetDebug(num_classes, in_ch=3, debug=False):
    """
    A not so big predefined resnet classifier for debugging purposes.

    Args:
        num_classes (int): the number of output classes
        in_ch (int): number of input channels, 3 for RGB images
        debug (bool): whereas to print additional debug info

    Returns:
        a resnet instance
    """
    return Classifier(
            ResNetBone(
                ['64:2', '64:1', '64:1', '128:2', '128:1', '256:2', '256:1'],
                tnn.Conv2dBNReLU,
                tnn.ResBlock,
                in_ch=in_ch,
                debug=debug), 256, num_classes)


def PreactResNetDebug(num_classes, in_ch=3, debug=False):
    """
    A not so big predefined preactivation resnet classifier for debugging purposes.

    Args:
        num_classes (int): the number of output classes
        in_ch (int): number of input channels, 3 for RGB images
        debug (bool): whereas to print additional debug info

    Returns:
        a resnet instance
    """
    return Classifier(
            ResNetBone(
                ['64:2', '64:1', '64:1', '128:2', '128:1', '256:2', '256:1'],
                tnn.Conv2d,
                tnn.PreactResBlock,
                in_ch=in_ch,
                debug=debug), 256, num_classes)
