import torch.nn as nn

def Conv2d(in_ch, out_ch, ks, stride=1):
    return nn.Conv2d(in_ch, out_ch, ks, padding=ks // 2, stride=stride)


def Conv3x3(in_ch, out_ch, stride=1):
    return Conv2d(in_ch, out_ch, 3, stride=stride)


def Conv1x1(in_ch, out_ch, stride=1):
    return Conv2d(in_ch, out_ch, 1, stride=stride)

