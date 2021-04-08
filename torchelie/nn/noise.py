import torch
import torch.nn as nn
from typing import Optional


class Noise(nn.Module):
    """
    Add gaussian noise to the input, with a per channel or global learnable std.

    Args:
        ch (int): number of input channels for a different std on each channel,
            or 1
    """
    def __init__(self, ch: int, inplace: bool = False, bias: bool = False):
        super(Noise, self).__init__()
        self.a = nn.Parameter(torch.zeros(ch, 1, 1))
        self.inplace = inplace
        self.bias = nn.Parameter(torch.zeros_like(self.a)) if bias else None

    def forward(self,
                x: torch.Tensor,
                z: Optional[torch.Tensor] = None) -> torch.Tensor:
        N, C, H, W = x.shape
        if z is None:
            z = torch.randn(N, 1, H, W, device=x.device, dtype=x.dtype)
        else:
            assert z.shape == [N, 1, H, W]
        z = z * self.a
        if self.bias is not None:
            z.add_(self.bias)

        if self.inplace:
            return x.add_(z)
        else:
            return x + z
