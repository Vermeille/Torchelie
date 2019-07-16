import torch.nn as nn
import torchelie.nn as tnn

from .classifier import Classifier


def VggBNBone(arch, in_ch=3, leak=0, debug=False):
    layers = []

    if debug:
        layers.append(tnn.Debug('Input'))

    for i, layer in enumerate(arch):
        if layer == 'M':
            layers.append(nn.MaxPool2d(2, 2))
        elif layer == 'A':
            layers.append(nn.AvgPool2d(2, 2))
        else:
            layers.append(tnn.Conv2dBNReLU(in_ch, layer, ks=3, leak=leak))
            in_ch = layer
        if debug:
            layer_name = 'layer_{}_{}'.format(layers[-1].__class__.__name__, i)
            layers.append(tnn.Debug(layer_name))
    return nn.Sequential(*layers)


def VggDebug(num_classes, in_ch=1, debug=False):
    return Classifier(
        VggBNBone([64, 64, 'M', 128, 'M', 128, 'M', 256, 256],
                  in_ch=in_ch,
                  debug=debug), 256, num_classes)
