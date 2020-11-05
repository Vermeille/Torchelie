import torch
import torch.nn as nn


class Noise(nn.Module):
    """
    Add gaussian noise to the input, with a per channel or global learnable std.

    Args:
        ch (int): number of input channels for a different std on each channel,
            or 1
    """
    def __init__(self, ch, inplace=False):
        """
        Initialize a chipy.

        Args:
            self: (todo): write your description
            ch: (todo): write your description
            inplace: (todo): write your description
        """
        super(Noise, self).__init__()
        self.a = nn.Parameter(torch.zeros(ch, 1, 1))
        self.inplace = inplace

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        N, C, H, W = x.shape
        z = torch.randn(N, 1, H, W, device=x.device, dtype=x.dtype)
        if self.inplace:
            return x.add_(z * self.a)
        else:
            return x + z * self.a
