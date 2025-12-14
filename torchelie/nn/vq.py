import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Tuple

from .functional import quantize


class VQ(nn.Module):
    """
    Quantization layer from *Neural Discrete Representation Learning*

    Args:
        embedding_dim (int): number of features along which to quantize
        num_embeddings (int): number of tokens in the codebook
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

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        dim: int = 1,
        commitment: float = 0.25,
        init_mode: str = "normal",
        space="l2",
        return_indices: bool = True,
        max_age: int = 1000,
    ):
        super(VQ, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.normal_(self.embedding.weight, 0, 1.1)
        self.dim = dim
        self.commitment = commitment
        self.register_buffer("initialized", torch.Tensor([0]))
        self.return_indices = return_indices
        assert init_mode in ["normal", "first"]
        self.init_mode = init_mode
        self.age = nn.Buffer(torch.empty(num_embeddings).fill_(max_age))
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

            print(f"{len(dead)} dead codes resampled")
            x_flat = x.view(-1, x.shape[-1])
            emb_weight = self.embedding.weight.data
            emb_weight[dead[: len(x_flat)]] = x_flat[
                torch.randperm(len(x_flat))[: len(dead)]
            ].to(emb_weight.dtype)
            self.age[dead[: len(x_flat)]] = 0

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

    def lookup(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., K)
        dim = self.dim
        needs_transpose = dim not in (-1, x.dim() - 1)

        x = self.embedding(x)
        if self.space == "angular":
            x = F.normalize(x, dim=-1)

        if needs_transpose:
            dims = list(range(x.ndim))
            dims.insert(dim, dims[-1])
            dims.pop()
            x = x.permute(*dims)
        # x: (..., D)
        return x

    def quantize(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        dim = self.dim
        nb_codes = self.embedding.weight.shape[0]

        codebook = self.embedding.weight
        if self.init_mode == "first" and self.initialized.item() == 0 and self.training:
            n_proto = self.embedding.weight.shape[0]

            ch_first = x.transpose(dim, -1).contiguous().view(-1, x.shape[dim])
            n_samples = ch_first.shape[0]
            idx = torch.randint(0, n_samples, (n_proto,))[:nb_codes]
            self.embedding.weight.data.copy_(ch_first[idx])
            self.initialized[:] = 1

        needs_transpose = dim not in (-1, x.dim() - 1)
        if needs_transpose:
            x = x.transpose(-1, dim).contiguous()

        if self.training:
            self.resample_dead(x)

        if self.space == "angular":
            codebook = F.normalize(codebook, dim=1)
            x = F.normalize(x, dim=-1)

        # x: (..., D)
        codes, indices = quantize(x, codebook, self.commitment)

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
        embedding_dim (int): number of features along which to quantize
        num_embeddings (int): number of tokens in the codebook
        num_codebooks (int): number of parallel codebooks
        dim (int): dimension along which to quantize
            an angular distance
        return_indices (bool): whether to return the indices of the quantized
            code points
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int,
        num_codebooks: int,
        dim: int = 1,
        commitment: float = 0.25,
        init_mode: str = "normal",
        return_indices: bool = True,
        max_age: int = 1000,
    ):
        assert (
            embedding_dim % num_codebooks == 0
        ), "num_codebooks must divide evenly embedding_dim"
        super(MultiVQ, self).__init__()
        self.dim = dim
        self.num_codebooks = num_codebooks
        self.return_indices = return_indices
        self.vqs = nn.ModuleList(
            [
                VQ(
                    embedding_dim // num_codebooks,
                    num_embeddings,
                    dim=dim,
                    commitment=commitment,
                    init_mode=init_mode,
                    return_indices=return_indices,
                    max_age=max_age,
                )
                for _ in range(num_codebooks)
            ]
        )

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


class RVQ(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_codebooks: int,
        *,
        dim: int = 1,
        commitment: float = 0.25,
        init_mode: str = "normal",
        return_indices: bool = True,
        max_age: int = 1000,
    ):
        super().__init__()
        self.dim = dim
        self.return_indices = return_indices
        self.codebooks = nn.ModuleList(
            [
                VQ(
                    num_embeddings,
                    embedding_dim,
                    dim=-1,
                    commitment=commitment,
                    init_mode=init_mode,
                    return_indices=True,
                    max_age=max_age,
                )
                for _ in range(num_codebooks)
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        dim = self.dim
        needs_transpose = dim not in (-1, x.dim() - 1)
        if needs_transpose:
            x = x.transpose(-1, dim).contiguous()

        out = torch.zeros_like(x)
        indices = []
        for i, cb in enumerate(self.codebooks):
            this_codes, this_indices = cb(x - out)
            out += this_codes
            print("residual", torch.norm(x - out).item())
            indices.append(this_indices)

        indices = torch.cat(indices, dim=-1)

        if needs_transpose:
            out = out.transpose(-1, dim).contiguous()
            indices = indices.transpose(-1, dim).contiguous()

        if self.return_indices:
            return out, indices
        else:
            return out

    def lookup(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., K)
        dim = self.dim
        needs_transpose = dim not in (-1, x.dim() - 1)

        x = torch.stack(
            [cb.lookup(xx) for cb, xx in zip(self.codebooks, x.split(1, dim=-1))],
            dim=-1,
        )
        x = x.sum(-1)

        if needs_transpose:
            dims = list(range(x.ndim))
            dims.insert(dim, dims[-1])
            dims.pop()
            x = x.permute(*dims)
        # x: (..., D)
        return x
