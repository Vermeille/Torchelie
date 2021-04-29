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
        mode ('angular' or 'nearest'): whether the distance between the input
            vectors and the codebook vectors is computed with a L2 distance or
            an angular distance
        return_indices (bool): whether to return the indices of the quantized
            code points
    """
    embedding: nn.Embedding
    dim: int
    commitment: float
    initialized: torch.Tensor
    mode: str
    return_indices: bool
    init_mode: str

    def __init__(self,
                 latent_dim: int,
                 num_tokens: int,
                 dim: int = 1,
                 commitment: float = 0.25,
                 mode: str = 'nearest',
                 init_mode: str = 'normal',
                 return_indices: bool = True):
        super(VQ, self).__init__()
        self.embedding = nn.Embedding(num_tokens, latent_dim)
        nn.init.normal_(self.embedding.weight, 0, 1)
        self.dim = dim
        self.commitment = commitment
        self.register_buffer('initialized', torch.Tensor([0]))
        assert mode in ['nearest', 'angular']
        self.mode = mode
        self.return_indices = return_indices
        assert init_mode in ['normal', 'first']
        self.init_mode = init_mode

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
        if self.mode == 'angular':
            codebook = F.normalize(codebook)
            x = F.normalize(x, dim=dim)

        if (self.init_mode == 'first' and self.initialized.item() == 0
                and self.training):
            n_proto = self.embedding.weight.shape[0]

            ch_first = x.transpose(dim, -1).contiguous().view(-1, x.shape[dim])
            n_samples = ch_first.shape[0]
            idx = torch.randint(0, n_samples, (n_proto, ))[:nb_codes]
            self.embedding.weight.data.copy_(ch_first[idx])
            self.initialized[:] = 1

        needs_transpose = dim != -1 or dim != x.dim() - 1
        if needs_transpose:
            x = x.transpose(-1, dim).contiguous()

        codes, indices = quantize(x, codebook, self.commitment, self.dim)

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
        mode ('angular' or 'nearest'): whether the distance between the input
            vectors and the codebook vectors is computed with a L2 distance or
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
                 mode: str = 'nearest',
                 init_mode: str = 'normal',
                 return_indices: bool = True):
        assert latent_dim % num_codebooks == 0, (
            "num_codebooks must divide evenly latent_dim")
        super(MultiVQ, self).__init__()
        self.dim = dim
        self.num_codebooks = num_codebooks
        self.vqs = nn.ModuleList([
            VQ(latent_dim // num_codebooks,
               num_tokens,
               dim=dim,
               commitment=commitment,
               mode=mode,
               init_mode=init_mode,
               return_indices=return_indices) for _ in range(num_codebooks)
        ])

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x_chunks = torch.chunk(x, self.num_codebooks, dim=self.dim)
        return torch.cat([vq(chunk) for chunk, vq in zip(x_chunks, self.vqs)],
                         dim=self.dim)
