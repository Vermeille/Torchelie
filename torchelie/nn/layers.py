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
