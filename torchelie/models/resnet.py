import torch.nn as nn
import torchelie.nn as tnn

from .classifier import Classifier

def ResNetBone(arch, head, block, in_ch=3):
    def parse(l):
        return [int(x) for x in l.split(':')]

    ch, s = parse(arch[0])
    layers = [
        tnn.Conv2dBNReLU(in_ch, ch, 3, stride=s)
    ]
    in_ch = ch
    for ch, s in map(parse, arch[1:]):
        layers.append(block(in_ch, ch, stride=s))
        in_ch = ch
    return nn.Sequential(*layers)

def ResNetDebug(in_ch=3):
    return Classifier(
        ResNetBone(
            ['64:2', '64:1', '64:1', '128:2', '128:1', '256:2', '256:1'],
            tnn.Conv2dBNReLU,
            tnn.ResBlock,
            in_ch=in_ch),
        256, 10
    )

def PreactResNetDebug(in_ch=3):
    return Classifier(
        ResNetBone(
            ['64:2', '64:1', '64:1', '128:2', '128:1', '256:2', '256:1'],
            tnn.Conv2d,
            tnn.PreactResBlock,
            in_ch=in_ch),
        256, 10
    )
