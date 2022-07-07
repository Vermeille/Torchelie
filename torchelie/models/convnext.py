import torch
import torch.nn as nn
import torchelie.nn as tnn
import torchelie.utils as tu
from .registry import register
from .classifier import ClassificationHead


class LayerScale(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.scales = nn.Parameter(torch.empty(num_features))
        nn.init.normal_(self.scales, 0, 1e-5)

    def forward(self, w):
        s = self.scales.view(-1, *([1] * (w.ndim - 1)))
        return s * w


class ConvNeXtBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        e = 4
        self.branch = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=7, padding=3, groups=ch, bias=False),
            nn.GroupNorm(1, ch), tu.kaiming(tnn.Conv1x1(ch, ch * e)),
            nn.SiLU(True), tu.constant_init(tnn.Conv1x1(ch * e, ch), 0))
        nn.utils.parametrize.register_parametrization(self.branch[-1],
                                                      'weight', LayerScale(ch))

    def forward(self, x):
        return self.branch(x).add_(x)


class ConvNeXt(nn.Sequential):
    def __init__(self, num_classes, arch):
        super().__init__()
        self.add_module('input', tu.xavier(nn.Conv2d(3, arch[0], 4, 1, 0)))

        prev_ch = arch[0]
        ch = arch[0]
        self.add_module(f'norm0', nn.GroupNorm(1, ch))
        self.add_module(f'act0', nn.SiLU(True))
        for i in range(len(arch)):
            if isinstance(arch[i], int):
                ch = arch[i]
                self.add_module(f'layer{i}', ConvNeXtBlock(ch))
                prev_ch = ch
            else:
                assert arch[i] == 'D'
                self.add_module(f'act{i}', nn.SiLU(True))
                self.add_module(f'norm{i}', nn.GroupNorm(1, ch))
                self.add_module(
                    f'layer{i}', tu.kaiming(nn.Conv2d(ch, arch[i + 1], 2, 2,
                                                      0)))
        self.add_module(f'norm{i}', nn.GroupNorm(1, ch))

        self.add_module('classifier',
                        ClassificationHead(arch[-1], num_classes))


@register
def convnext_xxt(num_classes):
    return ConvNeXt(num_classes, [64, 'D', 128, 'D', 256, 256, 256, 'D', 512])


@register
def convnext_xt(num_classes):
    return ConvNeXt(num_classes, [96, 'D', 192, 'D', 384, 384, 384, 'D', 768])


@register
def convnext_t(num_classes):
    return ConvNeXt(num_classes, [96] * 3 + ['D'] + [192] * 3 + ['D'] +
                    [384] * 9 + ['D'] + [768] * 3)
