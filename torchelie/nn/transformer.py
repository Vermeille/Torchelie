import torch
import torch.nn.functional as F
import torch.nn as nn
import torchelie.utils as tu
from .conv import Conv1x1
from .functional.transformer import local_attention_2d

from typing import Optional


class LocalSelfAttention2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_heads: int,
                 kernel_size: int,
                 hidden_channels: Optional[int] = None):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        self.hidden_channels = hidden_channels
        self.proj = tu.kaiming(Conv1x1(in_channels, hidden_channels * 3))
        self.position = nn.Parameter(
            torch.zeros(hidden_channels, kernel_size, kernel_size))
        self.out = tu.kaiming(Conv1x1(hidden_channels, in_channels))
        self.kernel_size = kernel_size
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W, P = x.shape[2], x.shape[3], self.kernel_size
        pad = (P - W % P, P - H % P)
        if pad != (0, 0):
            x = F.pad(x, (pad[0] // 2, pad[0] - pad[0] // 2, pad[1] // 2,
                          pad[1] - pad[1] // 2))
        x = local_attention_2d(x, self.proj, self.position, self.num_heads,
                               self.kernel_size)
        x = self.out(x)
        if pad == (0, 0):
            return x
        return x[:, :, pad[1] // 2:H + pad[1] // 2,
                 pad[0] // 2:W + pad[0] // 2]
