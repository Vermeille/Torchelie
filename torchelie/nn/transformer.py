import torch
import torch.nn.functional as F
import torch.nn as nn
import torchelie.utils as tu
from .conv import Conv1x1
from ..nn.condseq import CondSeq
from ..nn.llm import SelfAttention
from .functional.transformer import local_attention_2d

from typing import Optional
from typing_extensions import Literal


class LocalSelfAttentionHook(nn.Module):

    def forward(self, x, attn, pad):
        return x, attn, pad


class LocalSelfAttention2d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 num_heads: int,
                 kernel_size: int,
                 hidden_channels: Optional[int] = None,
                 padding_mode: Literal['none', 'auto'] = 'none'):
        """
        Args:
            in_channels (int): number of input channels
            num_heads (int): how many self attention heads
            kernel_size (int): the self attention window size. Must divide
                input size if padding_mode is 'none'.
            hidden_channels (int): how many channels *per head*.
            padding_mode (str): if 'none', no padding is used and kernel_size
                must divide the input spatial size. If 'auto', zero padding
                will be used to center the input feature map, and make it
                a multiple of kernel_size
        """
        super().__init__()
        hidden_channels = hidden_channels or (in_channels // num_heads)
        self.hidden_channels = hidden_channels
        self.proj = tu.kaiming(
            Conv1x1(in_channels, hidden_channels * num_heads * 3, bias=False))
        self.position = nn.Parameter(
            torch.zeros(num_heads, 2 * kernel_size, 2 * kernel_size))
        self.out = tu.kaiming(Conv1x1(hidden_channels * num_heads, in_channels))
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.attn_hook = LocalSelfAttentionHook()
        self.padding_mode = padding_mode

    def unfolded_posenc(self):
        w = self.kernel_size
        pos = torch.tensor([[x, y] for x in range(w) for y in range(w)])
        pos = pos[None, :, :] - pos[:, None, :]
        pos += w
        return self.position[:, pos[:, :, 0], pos[:, :, 1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W, P = x.shape[2], x.shape[3], self.kernel_size
        pad = (-(W % -P), -(H % -P))
        assert self.padding_mode == 'auto' or pad == (0, 0), (
            f'kernel_size {self.kernel_size} does not divide size {W}x{H} '
            'and padding_mode="none" specified')
        pad = (pad[0] // 2, pad[0] - pad[0] // 2, pad[1] // 2,
               pad[1] - pad[1] // 2)
        if pad != (0, 0, 0, 0):
            x = F.pad(x, pad)

        x, attn = local_attention_2d(x, self.proj, self.unfolded_posenc(),
                                     self.num_heads, self.kernel_size)
        self.attn_hook(x, attn, pad)

        if pad != (0, 0, 0, 0):
            x = x[:, :, pad[2]:H + pad[2], pad[0]:W + pad[0]]

        x = self.out(x)
        return x


class ViTBlock(nn.Module):
    """
    Vision Transformer (ViT) block consisting of a self-attention layer and a feed-forward MLP,
    each followed by RMS normalization and gated residual connections.

    Args:
        d_model (int): Dimension of the model.
        num_heads (int): Number of attention heads.

    Forward Args:
        x (Tensor): Input tensor of shape [B, L, d_model].
        z (Any): Optional conditioning input for CondSeq modules.

    Returns:
        Tensor: Output tensor of shape [B, L, d_model].
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.sa = CondSeq(
            nn.RMSNorm(d_model),
            SelfAttention(
                d_model,
                num_heads,
                head_size=d_model // num_heads,
                causal=False,
                rotary=True,
            ),
        )
        self.mlp = CondSeq(
            nn.RMSNorm(d_model),
            tu.kaiming(nn.Linear(d_model, 4 * d_model)),
            nn.GELU(),
            tu.kaiming(nn.Linear(4 * d_model, d_model)),
        )
        self.g1 = tu.kaiming(nn.Linear(d_model, d_model))
        self.g2 = tu.kaiming(nn.Linear(d_model, d_model))

    def forward(self, x):
        """
        Forward pass for the ViTBlock.

        Args:
            x (Tensor): Input tensor of shape [B, L, d_model].

        Returns:
            Tensor: Output tensor of shape [B, L, d_model].
        """
        x = self.sa(x) * torch.tanh(self.g1(x)) + x
        x = self.mlp(x) * torch.tanh(self.g2(x)) + x
        return x
