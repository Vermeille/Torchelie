import torch.nn as nn
import torchelie.nn as tnn

from .classifier import Classifier


def ResNetBone(arch, head, block, in_ch=3, debug=False):
    def parse(l):
        return [int(x) for x in l.split(':')]

    layers = []

    if debug:
        layers.append(tnn.Debug('Input'))

    ch, s = parse(arch[0])
    layers.append(tnn.Conv2dBNReLU(in_ch, ch, 3, stride=s))
    if debug:
        layers.append(tnn.Debug('Head'))
    in_ch = ch
    for i, (ch, s) in enumerate(map(parse, arch[1:])):
        layers.append(block(in_ch, ch, stride=s))
        in_ch = ch
        if debug:
            layer_name = 'layer_{}_{}'.format(layers[-1].__class__.__name__, i)
            layers.append(tnn.Debug(layer_name))
    return nn.Sequential(*layers)


def ResNetDebug(num_classes, in_ch=3, debug=False):
    return Classifier(
        ResNetBone(
            ['64:2', '64:1', '64:1', '128:2', '128:1', '256:2', '256:1'],
            tnn.Conv2dBNReLU,
            tnn.ResBlock,
            in_ch=in_ch,
            debug=debug), 256, num_classes)


def PreactResNetDebug(num_classes, in_ch=3, debug=False):
    return Classifier(
        ResNetBone(
            ['64:2', '64:1', '64:1', '128:2', '128:1', '256:2', '256:1'],
            tnn.Conv2d,
            tnn.PreactResBlock,
            in_ch=in_ch,
            debug=debug), 256, num_classes)
