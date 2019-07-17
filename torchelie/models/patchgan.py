import torch.nn as nn

import torchelie.nn as tnn
from torchelie.utils import kaiming, n002, xavier


def patch_discr(arch, in_ch=3, out_ch=1, norm=None):
    def block(in_ch, out_ch, norm):
        if norm is None:
            return [
                kaiming(nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1), a=0.2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        else:
            return [
                kaiming(nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1), a=0.2),
                norm(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ]

    layers = block(in_ch, arch[0], None)

    in_ch = arch[0]
    for out_ch in arch[1:]:
        layers += block(in_ch, out_ch, norm)
        in_ch = out_ch

    layers += [
        n002(nn.Conv2d(in_ch, out_ch, 1)),
        nn.AdaptiveAvgPool2d(1),
        tnn.Reshape(out_ch)
    ]

    return nn.Sequential(*layers)


def Patch286(in_ch=3, out_ch=1, norm=nn.BatchNorm2d):
    return patch_discr([64, 128, 256, 512, 512, 512],
            in_ch=in_ch, out_ch=out_ch, norm=norm)


def Patch70(in_ch=3, out_ch=1, norm=nn.BatchNorm2d):
    return patch_discr([64, 128, 256, 512], in_ch=in_ch, out_ch=out_ch,
            norm=norm)


# Not sure about the receptive field but ok
def Patch32(in_ch=3, out_ch=1, norm=nn.BatchNorm2d):
    return patch_discr([64, 128, 256], in_ch=in_ch, out_ch=out_ch,
            norm=norm)


def Patch16(in_ch=3, out_ch=1, norm=nn.BatchNorm2d):
    return patch_discr([64, 128], in_ch=in_ch, out_ch=out_ch,
            norm=norm)
