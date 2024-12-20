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
                 space="l2",
                 return_indices: bool = True,
                 max_age: int = 1000):
        super(VQ, self).__init__()
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(num_tokens, latent_dim)
        nn.init.normal_(self.embedding.weight, 0, 1.1)
        self.dim = dim
        self.commitment = commitment
        self.register_buffer('initialized', torch.Tensor([0]))
        self.return_indices = return_indices
        assert init_mode in ['normal', 'first']
        self.init_mode = init_mode
        self.register_buffer('age', torch.empty(num_tokens).fill_(max_age))
        self.max_age = max_age
        self.space = space
        assert space in ["l2", "angular"]

    def update_usage(self, indices):
        with torch.no_grad():
            self.age += 1
            if torch.distributed.is_initialized():
                n_gpu = torch.distributed.get_world_size()
                all_indices = [torch.empty_like(indices) for _ in range(n_gpu)]
                torch.distributed.all_gather(all_indices, indices)
                indices = torch.cat(all_indices)
            used = torch.unique(indices)
            self.age[used] = 0

    def resample_dead(self, x):
        with torch.no_grad():
            dead = torch.nonzero(self.age > self.max_age, as_tuple=True)[0]
            if len(dead) == 0:
                return

            print(f'{len(dead)} dead codes resampled')
            x_flat = x.view(-1, x.shape[-1])
            emb_weight = self.embedding.weight.data
            emb_weight[dead[:len(x_flat)]] = x_flat[torch.randperm(
                len(x_flat))[:len(dead)]].to(emb_weight.dtype)
            self.age[dead[:len(x_flat)]] = 0

            if torch.distributed.is_initialized():
                torch.distributed.broadcast(emb_weight, 0)

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
        if torch.is_floating_point(x):
            return self.quantize(x)
        else:
            return self.lookup(x)

    def lookup(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        dim = self.dim
        needs_transpose = dim != -1 or dim != x.dim() - 1

        x = self.embedding(x)
        if self.space == "angular":
            x = F.normalize(x, dim=-1)

        if needs_transpose:
            dims = list(range(x.ndim))
            dims.insert(dim, dims[-1])
            dims.pop()
            x = x.permute(*dims)
        return x

    def quantize(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        dim = self.dim
        nb_codes = self.embedding.weight.shape[0]

        codebook = self.embedding.weight
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

        if self.training:
            self.resample_dead(x)

        if self.space == "angular":
            codebook = F.normalize(codebook, dim=1)
            x = F.normalize(x, dim=-1)

        codes, indices = quantize(x, codebook, self.commitment, -1)

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
                 return_indices: bool = True,
                 max_age: int = 1000):
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
               return_indices=return_indices,
               max_age=max_age) for _ in range(num_codebooks)
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


class MultiVQ2(VQ):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.latent_dim
        dims = x.shape
        batched_dims = list(dims)
        batched_dims[self.dim] = d
        batched_dims[self.dim - 1] = -1
        out = super(MultiVQ2, self).forward(x.view(*batched_dims))
        return out.view(*dims).contiguous()
