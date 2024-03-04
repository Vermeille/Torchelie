import math
import torch
import torch.nn as nn
import torchelie.utils as tu


class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        """
        Rotary Positional Embedding
        Assumes input of shape (..., seq_len, dim)
        Args:
            dim (int): dimension of the input
            base (int, optional): base of the sinusoidal function. Defaults to 10000.
        """
        super().__init__()
        inv_freq = 1.0 / (base**(torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, q, k, v, seq_dim=-2):
        seq_len = q.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(q.shape[seq_dim],
                             device=q.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(q.device)
            self.cos_cached = emb.cos()[:, :]
            self.sin_cached = emb.sin()[:, :]
        return self.apply_rotary_pos_emb(q, k, v, self.cos_cached,
                                         self.sin_cached)

    # rotary pos emb helpers:

    def rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat(
            (-x2, x1),
            dim=x1.ndim - 1)  # dim=-1 triggers a bug in torch < 1.8.0

    def apply_rotary_pos_emb(self, q, k, v, cos, sin):
        return (q * cos) + (self.rotate_half(q) *
                            sin), (k * cos) + (self.rotate_half(k) * sin), v


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
    """

    def __init__(self, hidden_size, num_heads, head_size, causal=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.qkv = tu.normal_init(
            nn.Linear(hidden_size, head_size * num_heads * 3, bias=False),
            math.sqrt(2 / (5 * hidden_size)))
        self.fc = tu.xavier(
            nn.Linear(head_size * num_heads, hidden_size, bias=False))
        self.rotary = Rotary(head_size)
        self.causal = causal

    def forward(self, x, kv_cache=None):
        b, l, h, d = x.shape[0], x.shape[1], self.num_heads, self.head_size
        # bld -> (q/k/v)bhld
        qkv = self.qkv(x).reshape(b, l, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if kv_cache is not None:
            # update k, v
            k, v = (torch.cat([kv_cache[0], k],
                              dim=2), torch.cat([kv_cache[1], v], dim=2))
            # update cache
            kv_cache[:] = [k, v]
        q, k, v = self.rotary(q, k, v)
        att = nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=kv_cache is None or self.causal)
        # bhld -> blhd
        att = att.permute(0, 2, 1, 3).contiguous().reshape(b, l, h * d)
        return self.fc(att)

    def extra_repr(self):
        return f"hidden_size={self.qkv.in_features}, num_heads={self.num_heads}, head_size={self.head_size}, causal={self.causal}"
