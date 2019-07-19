import torch.nn as nn
import torchelie.nn as tnn
from torchelie.utils import kaiming, xavier

from .classifier import Classifier


def VggBNBone(arch, in_ch=3, leak=0, block=tnn.Conv2dBNReLU, debug=False):
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
    return Classifier(
        VggBNBone([64, 64, 'M', 128, 'M', 128, 'M', 256, 256],
                  in_ch=in_ch,
                  debug=debug), 256, num_classes)


def VggGeneratorDebug(in_noise=32, out_ch=3):
    return nn.Sequential(
        kaiming(nn.Linear(in_noise, 128 * 16)), tnn.Reshape(128, 4, 4),
        nn.LeakyReLU(0.2, inplace=True),
        VggBNBone([128, 'U', 64, 'U', 32, 'U', 16], in_ch=128),
        xavier(tnn.Conv1x1(16, 1)), nn.Sigmoid())


class VggImg2ImgGeneratorDebug(nn.Module):
    def __init__(self, in_noise, out_ch, side_ch=1):
        super(VggImg2ImgGeneratorDebug, self).__init__()

        def make_block(in_ch, out_ch, **kwargs):
            return tnn.Conv2dNormReLU(in_ch,
                                      out_ch,
                                      norm=tnn.Spade2d(out_ch, side_ch, 64),
                                      **kwargs)

        self.net = tnn.CondSeq(
            kaiming(nn.Linear(in_noise, 128 * 16)), tnn.Reshape(128, 4, 4),
            nn.LeakyReLU(0.2, inplace=True),
            VggBNBone([128, 'U', 64, 'U', 32, 'U', 16],
                      in_ch=128,
                      block=make_block), xavier(tnn.Conv1x1(16, out_ch)),
            nn.Sigmoid())

    def forward(self, x, y):
        return self.net(x, y)


class VggClassCondGeneratorDebug(nn.Module):
    def __init__(self, in_noise, out_ch, num_classes):
        super(VggClassCondGeneratorDebug, self).__init__()

        def make_block(in_ch, out_ch, **kwargs):
            return tnn.Conv2dNormReLU(in_ch,
                                      out_ch,
                                      norm=tnn.ConditionalBN2d(out_ch, 64),
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
        y_emb = self.emb(y)
        return self.net(x, y_emb)
