import torch
import torch.nn as nn


class Noise(nn.Module):
    """
    Add gaussian noise to the input, with a per channel or global learnable std.

    Args:
        ch (int): number of input channels for a different std on each channel,
            or 1
    """
    def __init__(self, ch, init_val=0, inplace=False):
        super(Noise, self).__init__()
        self.a = nn.Parameter(torch.ones(ch, 1, 1) *  init_val)
        self.inplace = inplace

    def forward(self, x):
        N, C, H, W = x.shape
        z = torch.randn(N, 1, H, W, device=x.device, dtype=x.dtype)
        if self.inplace:
            return x.add_(z * self.a)
        else:
            return x + z * self.a
