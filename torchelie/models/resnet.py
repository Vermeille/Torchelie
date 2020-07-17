import functools
import torch.nn as nn
import torchelie.nn as tnn
import torchelie.utils as tu

from .classifier import Classifier, Classifier1


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
    norm_ctor = functools.partial(tnn.ConditionalBN2d, cond_channels=hidden)
    block_ctor = functools.partial(tnn.ResBlock, norm=norm_ctor)
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
            ['64:1', '64:1', '128:2', '128:1', '256:2', '256:1'],
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
        norm_ctor = functools.partial(tnn.ConditionalBN2d, cond_channels=hidden)
        block_ctor = functools.partial(tnn.ResBlock, norm=norm_ctor)
        self.bone = ResNetBone(arch, head, block_ctor, in_ch, debug)
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
            ['64:1', '64:1', '128:2', '128:1', '256:2', '256:1'],
            tnn.Conv2dBNReLU,
            64,
            num_cond_classes,
            in_ch=in_ch,
            debug=debug), 256, num_classes)


def ResNetBone(head, head_ch, arch, block, debug=False):
    """
    A resnet

    How to specify an architecture:

    It's a list of block specifications. Each element is a string of the form
    "output channels:stride". For instance "64:2" is a block with input stride
    2 and 64 output channels.

    Args:
        head (Module): head module at the start of the net
        head_ch (int): number of output channels of the head
        arch (list): the architecture specification
        block (fn): the residual block to use ctor
        debug (bool): should insert debug layers between each layer

    Returns:
        A Resnet instance
    """
    def parse(l):
        return [int(x) for x in l.split(':')]

    layers = [head]

    if debug:
        layers.append(tnn.Debug('Head'))
    in_ch = head_ch
    for i, (ch, s) in enumerate(map(parse, arch)):
        layers.append(block(in_ch, ch, stride=s))
        in_ch = ch
        if debug:
            layer_name = 'layer_{}_{}'.format(layers[-1].__class__.__name__, i)
            layers.append(tnn.Debug(layer_name))

    return tnn.CondSeq(*layers)


def PreactResNetBone(head, head_ch, arch, block, debug=False):
    """
    A resnet

    How to specify an architecture:

    It's a list of block specifications. Each element is a string of the form
    "output channels:stride". For instance "64:2" is a block with input stride
    2 and 64 output channels.

    Args:
        head (Module): head module at the start of the net
        head_ch (int): number of output channels of the head
        arch (list): the architecture specification
        block (fn): the residual block to use ctor
        debug (bool): should insert debug layers between each layer

    Returns:
        A Resnet instance
    """
    def parse(l):
        return [int(x) for x in l.split(':')]

    layers = [head]

    if debug:
        layers.append(tnn.Debug('Head'))
    in_ch = head_ch
    for i, (ch, s) in enumerate(map(parse, arch)):
        layers.append(block(in_ch, ch, stride=s, first_layer=(i == 0)))
        in_ch = ch
        if debug:
            layer_name = 'layer_{}_{}'.format(layers[-1].__class__.__name__, i)
            layers.append(tnn.Debug(layer_name))

    layers.append(nn.BatchNorm2d(ch))
    layers.append(nn.ReLU(True))
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
                tnn.Conv2dBNReLU(in_ch, 64, ks=7, stride=2), 64
                ['64:1', '64:1', '128:2', '128:1', '256:2', '256:1'],
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
                ['64:1', '64:1', '128:2', '128:1', '256:2', '256:1'],
                functools.partial(tnn.Conv2dBNReLU, ks=3, stride=1),
                tnn.PreactResBlock,
                in_ch=in_ch,
                debug=debug), 256, num_classes)


def resnet20_cifar(num_classes, in_ch=3, debug=False):
    return Classifier1(
            ResNetBone(
                tnn.Conv2dBNReLU(in_ch, 16, ks=3, stride=1),
                16,
                ['16:1', '16:1', '16:1',
                    '32:2', '32:1', '32:1',
                    '64:2', '64:1', '64:1'],
                tnn.ResBlock,
                debug=debug), 64, num_classes, dropout=0)

def _preact_head(in_ch, out_ch, input_size=128):
    if input_size <= 64:
        return tu.kaiming(tnn.Conv2d(in_ch, out_ch, ks=3))
    elif input_size <= 128:
        return tu.kaiming(tnn.Conv2d(in_ch, out_ch, ks=5, stride=2))
    else:
        return tnn.CondSeq(
                tu.kaiming(tnn.Conv2d(in_ch, out_ch, ks=7, stride=2)),
                nn.MaxPool2d(3, 2, 1)
            )

def preact_resnet20_cifar(num_classes, in_ch=3, debug=False, **kwargs):
    return Classifier1(
            PreactResNetBone(tnn.Conv2d(in_ch, 16,  ks=3), 16,
                ['16:1', '16:1', '16:1',
                    '32:2', '32:1', '32:1',
                    '64:2', '64:1', '64:1'],
                functools.partial(tnn.PreactResBlock, **kwargs),
                debug=debug,
                ), 64, num_classes, dropout=0)


def resnet18(num_classes, in_ch=3, debug=False):
    return Classifier1(
            ResNetBone(
                tnn.Conv2dBNReLU(3, 64, ks=7, stride=2),
                64,
                ['64:1', '64:1', '128:2', '128:1', '256:2', '256:1', '512:2',
                    '512:1'],
                tnn.ResBlock,
                debug=debug)
            , 512, num_classes, dropout=0)


def preact_resnet18(num_classes, in_ch=3, input_size=224, debug=False):
    head = _preact_head(in_ch, 64, input_size)
    return Classifier1(
            PreactResNetBone(head, 64,
                ['64:1', '64:1', '128:2', '128:1', '256:2', '256:1', '512:2',
                    '512:1'],
                tnn.PreactResBlock,
                debug=debug)
            , 512, num_classes, dropout=0)


def preact_resnet26_reduce(num_classes, in_ch=3, input_size=224, debug=False):
    head = _preact_head(in_ch, 64, input_size)
    return Classifier1(
            PreactResNetBone(head, 64,
                ['64:1', '64:1', '128:2', '128:1', '256:2', '256:1', '512:2',
                    '512:1', '512:2', '512:1', '512:2', '512:1'],
                tnn.PreactResBlock,
                debug=debug)
            , 512, num_classes, dropout=0)


def preact_resnet18_exp(num_classes, in_ch=3, debug=False):
    head = tu.kaiming(tnn.Conv2d(3, 64, ks=5, stride=2))
    return Classifier1(
            PreactResNetBone(head, 64,
                ['128:1', '128:1', '256:2', '256:1', '512:2', '512:1', '1024:2',
                    '1024:1'],
                tnn.PreactResBlock,
                debug=debug)
            , 1024, num_classes, dropout=0)
