import torch.nn as nn
import torchelie.nn as tnn

from .classifier import Classifier


def VggBNBone(arch, in_ch=3, leak=0):
    layers = []
    for layer in arch:
        if layer == 'M':
            layers.append(nn.MaxPool2d(2, 2))
        elif layer == 'A':
            layers.append(nn.AvgPool2d(2, 2))
        else:
            layers.append(tnn.Conv2dBNReLU(in_ch, layer, ks=3, leak=leak))
            in_ch = layer
    return nn.Sequential(*layers)


def VggDebug(in_ch=1):
    return Classifier(
        VggBNBone([64, 64, 'M', 128, 'M', 128, 'M', 256, 256], in_ch=in_ch),
        256, 10)
