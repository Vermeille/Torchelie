import torch


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
