import torch.nn as nn

def Conv2d(in_ch, out_ch, ks):
    return nn.Conv2d(in_ch, out_ch, ks, padding=ks // 2)


def Conv3x3(in_ch, out_ch):
    return Conv(in_ch, out_ch, 3)


def Conv1x1(in_ch, out_ch):
    return Conv(in_ch, out_ch, 1)

