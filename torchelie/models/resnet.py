import functools
import torch.nn as nn
import torchelie.nn as tnn

from .classifier import Classifier


class ConditionalSequential(nn.Sequential):
    def forward(self, x, z=None):
        for m in self._modules.values():
            if hasattr(m, 'condition'):
                x = m(x, z)
            else:
                x = m(x)
        return x


class ClassCondResNetBone(nn.Module):
    def __init__(self, arch, head, hidden, num_classes, in_ch=3, debug=False):
        super(ClassCondResNetBone, self).__init__()
        block_ctor = functools.partial(tnn.ConditionalResBlock, hidden=hidden)
        self.bone = ConditionalSequential(
            *ResNetBone(arch, tnn.Conv2d, block_ctor, in_ch, debug))
        self.emb = nn.Embedding(num_classes, hidden)

    def forward(self, x, y):
        y_emb = self.emb(y)
        return self.bone(x, y_emb)


def ClassCondResNetDebug(num_classes, num_cond_classes, in_ch=3, debug=False):
    return Classifier(
        ClassCondResNetBone(
            ['64:2', '64:1', '64:1', '128:2', '128:1', '256:2', '256:1'],
            tnn.Conv2dBNReLU,
            64,
            num_cond_classes,
            in_ch=in_ch,
            debug=debug), 256, num_classes)


def ResNetBone(arch, head, block, in_ch=3, debug=False):
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
    return layers


def ResNetDebug(num_classes, in_ch=3, debug=False):
    return Classifier(
        nn.Sequential(
            *ResNetBone(
                ['64:2', '64:1', '64:1', '128:2', '128:1', '256:2', '256:1'],
                tnn.Conv2dBNReLU,
                tnn.ResBlock,
                in_ch=in_ch,
                debug=debug)), 256, num_classes)


def PreactResNetDebug(num_classes, in_ch=3, debug=False):
    return Classifier(
        nn.Sequential(
            *ResNetBone(
                ['64:2', '64:1', '64:1', '128:2', '128:1', '256:2', '256:1'],
                tnn.Conv2d,
                tnn.PreactResBlock,
                in_ch=in_ch,
                debug=debug)), 256, num_classes)
