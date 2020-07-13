import torch
import torch.nn as nn


def Conv2d(in_ch, out_ch, ks, stride=1, bias=True):
    """
    A Conv2d with 'same' padding
    """
    return nn.Conv2d(in_ch, out_ch, ks, padding=ks // 2, stride=stride,
            bias=bias)


def Conv3x3(in_ch, out_ch, stride=1, bias=True):
    """
    A 3x3 Conv2d with 'same' padding
    """
    return Conv2d(in_ch, out_ch, 3, stride=stride, bias=bias)


def Conv1x1(in_ch, out_ch, stride=1, bias=True):
    """
    A 1x1 Conv2d
    """
    return Conv2d(in_ch, out_ch, 1, stride=stride, bias=bias)


class AdaptiveConcatPool2d(nn.Module):
    """
    Pools with AdaptiveMaxPool2d AND AdaptiveAvgPool2d and concatenates both
    results.

    Args:
        target_size: the target output size (single integer or
            double-integer tuple)
    """
    def __init__(self, target_size):
        super(AdaptiveConcatPool2d, self).__init__()
        self.target_size = target_size

    def forward(self, x):
        return torch.cat([
            nn.functional.adaptive_avg_pool2d(x, self.target_size),
            nn.functional.adaptive_max_pool2d(x, self.target_size),
        ], dim=1)
