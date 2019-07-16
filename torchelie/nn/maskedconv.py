import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_chan, out_chan, ks, center):
        super(MaskedConv2d, self).__init__(in_chan,
                                           out_chan, (ks // 2 + 1, ks),
                                           padding=0)
        self.register_buffer('mask', torch.ones(ks // 2 + 1, ks))
        self.mask[-1, ks // 2 + (1 if center else 0):] = 0

    def forward(self, x):
        self.weight_orig = self.weight
        del self.weight
        self.weight = self.weight_orig * self.mask
        ks = self.weight.shape[-1]

        x = F.pad(x, (ks // 2, ks // 2, ks // 2, 0))
        res = super(MaskedConv2d, self).forward(x)

        self.weight = self.weight_orig
        return res
