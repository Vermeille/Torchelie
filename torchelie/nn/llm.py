from typing import Tuple

import math
import torch
import torch.nn as nn
import torchelie.utils as tu


class Rotary(torch.nn.Module):
    _cache = {}

    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.inv_freq = nn.Buffer(inv_freq)

    def _cache_key(self, device):
        return (self.dim, self.base, device)

    def forward(self, q, k, v, seq_dim=-2):
        seq_len = q.shape[seq_dim]
        device = q.device
        key = self._cache_key(device)
        cos_cached, sin_cached, cached_len = self._cache.get(key, (None, None, 0))
        needed_len = seq_len
        if cached_len < needed_len or cos_cached is None:
            t = torch.arange(needed_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(device)
            cos_cached = emb.cos()
            sin_cached = emb.sin()
            cached_len = needed_len
            self._cache[key] = (cos_cached, sin_cached, cached_len)
        cos, sin = cos_cached[:seq_len], sin_cached[:seq_len]
        ndim = q.ndim
        seq_dim = seq_dim % ndim
        # default layout already matches (seq_len, dim)
        if seq_dim != ndim - 2:
            shape = [1] * ndim
            shape[seq_dim] = seq_len
            shape[-1] = cos.shape[-1]
            cos = cos.reshape(shape)
            sin = sin.reshape(shape)

        return self.apply_rotary_pos_emb(q, k, v, cos, sin)

    # rotary pos emb helpers:

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat(
            (-x2, x1), dim=x1.ndim - 1
        )  # dim=-1 triggers a bug in torch < 1.8.0

    def apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            (q * cos) + (self.rotate_half(q) * sin),
            (k * cos) + (self.rotate_half(k) * sin),
            v,
        )


class SelfAttention(nn.Module):
    """
    Self-attention layer
    Assumes input of shape (b, l, hidden_size). Uses scaled dot-product
    attention and rotary positional embeddings.

    Args:
        hidden_size (int): size of the hidden dimension
        num_heads (int): number of heads
        head_size (int): size of each head
        causal (bool, optional): whether to apply causal masking. Defaults to True.
        rotary (bool, optional): whether to apply RoPE. Defaults to True.
    """

    def __init__(self, hidden_size, num_heads, head_size, causal=True, rotary=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.qkv = tu.kaiming(
            nn.Linear(hidden_size, head_size * num_heads * 3, bias=False)
        )
        self.g = tu.kaiming(nn.Linear(hidden_size, num_heads))
        self.fc = tu.xavier(nn.Linear(head_size * num_heads, hidden_size, bias=False))
        self.rotary = Rotary(head_size) if rotary else None
        self.causal = causal

    def forward(self, x):
        b, l, h, d = x.shape[0], x.shape[1], self.num_heads, self.head_size
        # bld -> (q/k/v)bhld
        qkv = self.qkv(x).reshape(b, l, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rotary is not None:
            q, k, v = self.rotary(q, k, v)

        g = self.g(x).permute(0, 2, 1).view(b, h, l, 1)  # blh -> bhl1
        att = nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=self.causal
        ) * torch.sigmoid(g)
        # bhld -> blhd
        att = att.permute(0, 2, 1, 3).contiguous().reshape(b, l, h * d)
        return self.fc(att)

    def extra_repr(self):
        return f"hidden_size={self.qkv.in_features}, num_heads={self.num_heads}, head_size={self.head_size}, causal={self.causal}"
