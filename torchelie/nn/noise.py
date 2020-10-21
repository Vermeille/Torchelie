import torch
import torch.nn as nn


class Noise(nn.Module):
    """
    Add gaussian noise to the input, with a per channel or global learnable std.

    Args:
        ch (int): number of input channels for a different std on each channel,
            or 1
    """
    def __init__(self, ch, inplace=False, bias=None):
        super(Noise, self).__init__()
        self.a = nn.Parameter(torch.zeros(ch, 1, 1))
        self.inplace = inplace
        self.bias = nn.Parameter(torch.zeros_like(self.a)) if bias else None

    def forward(self, x):
        N, C, H, W = x.shape
        z = torch.randn(N, 1, H, W, device=x.device, dtype=x.dtype)
        z = z * self.a
        if self.bias is not None:
            z.add_(self.bias)

        if self.inplace:
            return x.add_(z)
        else:
            return x + z
