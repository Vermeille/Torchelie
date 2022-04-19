import torch
import torch.nn.functional as F
import torch.nn as nn
import torchelie.utils as tu
from .conv import Conv1x1
from .functional.transformer import local_attention_2d

from typing import Optional


class LocalSelfAttentionHook(nn.Module):
    def forward(self, x, attn, pad):
        return x, attn, pad


class LocalSelfAttention2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_heads: int,
                 kernel_size: int,
                 hidden_channels: Optional[int] = None,
                 dropout=0):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        self.hidden_channels = hidden_channels
        self.proj = tu.kaiming(
            Conv1x1(in_channels, hidden_channels * 3, bias=False))
        self.position = nn.Parameter(
            torch.zeros(num_heads, 2 * kernel_size, 2 * kernel_size))
        self.out = tu.kaiming(Conv1x1(hidden_channels, in_channels))
        self.kernel_size = kernel_size
        self.num_heads = num_heads

    def unfolded_posenc(self):
        w = self.kernel_size
        pos = torch.tensor([[x, y] for x in range(w) for y in range(w)])
        pos = pos[None, :, :] - pos[:, None, :]
        pos += w
        return self.position[:, pos[:, :, 0], pos[:, :, 1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W, P = x.shape[2], x.shape[3], self.kernel_size
        pad = (P - W % P, P - H % P)
        if pad != (0, 0):
            x = F.pad(x, (pad[0] // 2, pad[0] - pad[0] // 2, pad[1] // 2,
                          pad[1] - pad[1] // 2))
        x = local_attention_2d(x, self.proj, self.unfolded_posenc(),
                               self.num_heads, self.kernel_size)
        x = self.out(x)
        if pad == (0, 0):
            return x
        return x[:, :, pad[1] // 2:H + pad[1] // 2,
                 pad[0] // 2:W + pad[0] // 2]
