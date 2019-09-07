import torch
import torch.nn as nn


class Noise(nn.Module):
    """
    Add gaussian noise to the input, with a per channel or global learnable std.

    Args:
        ch (int): number of input channels for a different std on each channel,
            or 1
    """
    def __init__(self, ch):
        super(Noise, self).__init__()
        self.a = nn.Parameter(torch.zeros(ch, 1, 1))

    def forward(self, x):
        return x + self.a * torch.randn_like(x)
