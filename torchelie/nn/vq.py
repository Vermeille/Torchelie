import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Tuple

from .functional import quantize


class VQ(nn.Module):
    """
    Quantization layer from *Neural Discrete Representation Learning*

    Args:
        latent_dim (int): number of features along which to quantize
        num_tokens (int): number of tokens in the codebook
        dim (int): dimension along which to quantize
        return_indices (bool): whether to return the indices of the quantized
            code points
    """
    embedding: nn.Embedding
    dim: int
    commitment: float
    initialized: torch.Tensor
    return_indices: bool
    init_mode: str

    def __init__(self,
                 latent_dim: int,
                 num_tokens: int,
                 dim: int = 1,
                 commitment: float = 0.25,
                 init_mode: str = 'normal',
                 return_indices: bool = True):
        super(VQ, self).__init__()
        self.embedding = nn.Embedding(num_tokens, latent_dim)
        nn.init.normal_(self.embedding.weight, 0, 1.1)
        self.dim = dim
        self.commitment = commitment
        self.register_buffer('initialized', torch.Tensor([0]))
        self.return_indices = return_indices
        assert init_mode in ['normal', 'first']
        self.init_mode = init_mode
        self.register_buffer('usage', torch.zeros(num_tokens))

    def update_usage(self, indices):
        nb_codes = self.embedding.weight.shape[0]
        with torch.no_grad():
            id_flat = indices.view(indices.shape[0], -1)
            u = F.one_hot(id_flat, nb_codes).float().mean(0).sum(0)
            self.usage.mul_(0.99).add_(0.01 * nb_codes * u)

    def resample_dead(self, x):
        with torch.no_grad():
            dead = torch.nonzero(self.usage < 0.001, as_tuple=True)[0]
            if len(dead) > 0:
                print(f'{len(dead)} dead codes resampled')
                x_flat = x.view(-1, x.shape[-1])
                emb_weight = self.embedding.weight.data
                emb_weight[dead[:len(x_flat)]] = x_flat[torch.randperm(
                    len(x_flat))[:len(dead)]]

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass

        Args:
            x (tensor): input tensor

        Returns:
            quantized tensor, or (quantized tensor, indices) if
            `self.return_indices`
        """
        dim = self.dim
        nb_codes = self.embedding.weight.shape[0]

        codebook = self.embedding.weight
        if (self.init_mode == 'first' and self.initialized.item() == 0 and
                self.training):
            n_proto = self.embedding.weight.shape[0]

            ch_first = x.transpose(dim, -1).contiguous().view(-1, x.shape[dim])
            n_samples = ch_first.shape[0]
            idx = torch.randint(0, n_samples, (n_proto,))[:nb_codes]
            self.embedding.weight.data.copy_(ch_first[idx])
            self.initialized[:] = 1

        needs_transpose = dim != -1 or dim != x.dim() - 1
        if needs_transpose:
            x = x.transpose(-1, dim).contiguous()

        if self.training:
            self.resample_dead(x)

        codes, indices = quantize(x, codebook, self.commitment, self.dim)

        if self.training:
            self.update_usage(indices)

        if needs_transpose:
            codes = codes.transpose(-1, dim)
            indices = indices.transpose(-1, dim)

        if self.return_indices:
            return codes, indices
        else:
            return codes


class MultiVQ(nn.Module):
    """
    Multi codebooks quantization layer from *Neural Discrete Representation
    Learning*

    Args:
        latent_dim (int): number of features along which to quantize
        num_tokens (int): number of tokens in the codebook
        num_codebooks (int): number of parallel codebooks
        dim (int): dimension along which to quantize
            an angular distance
        return_indices (bool): whether to return the indices of the quantized
            code points
    """

    def __init__(self,
                 latent_dim: int,
                 num_tokens: int,
                 num_codebooks: int,
                 dim: int = 1,
                 commitment: float = 0.25,
                 init_mode: str = 'normal',
                 return_indices: bool = True):
        assert latent_dim % num_codebooks == 0, (
            "num_codebooks must divide evenly latent_dim")
        super(MultiVQ, self).__init__()
        self.dim = dim
        self.num_codebooks = num_codebooks
        self.return_indices = return_indices
        self.vqs = nn.ModuleList([
            VQ(latent_dim // num_codebooks,
               num_tokens,
               dim=dim,
               commitment=commitment,
               init_mode=init_mode,
               return_indices=return_indices) for _ in range(num_codebooks)
        ])

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x_chunks = torch.chunk(x, self.num_codebooks, dim=self.dim)
        quantized = [vq(chunk) for chunk, vq in zip(x_chunks, self.vqs)]
        if self.return_indices:
            q = torch.cat([q[0] for q in quantized], dim=self.dim)
            return q, torch.cat([q[1] for q in quantized], dim=self.dim)
        else:
            return torch.cat(quantized, dim=self.dim)
