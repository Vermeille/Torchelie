import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_chan, out_chan, ks, center, stride=1, bias=(1, 1)):
        super(MaskedConv2d, self).__init__(in_chan,
                                           out_chan, (ks // 2 + 1, ks),
                                           padding=0,
                                           stride=stride,
                                           bias=False)
        self.register_buffer('mask', torch.ones(ks // 2 + 1, ks))
        self.mask[-1, ks // 2 + (1 if center else 0):] = 0

        self.spatial_bias = None
        if bias is not None:
            self.spatial_bias = nn.Parameter(torch.zeros(out_chan, *bias))

        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x):
        self.weight_orig = self.weight
        del self.weight
        self.weight = self.weight_orig * self.mask
        ks = self.weight.shape[-1]

        x = F.pad(x, (ks // 2, ks // 2, ks // 2, 0))
        res = super(MaskedConv2d, self).forward(x)

        self.weight = self.weight_orig
        del self.weight_orig
        if self.spatial_bias is not None:
            return res + self.spatial_bias
        else:
            return res
