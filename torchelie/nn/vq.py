import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .functional import quantize


class VQ(nn.Module):
    def __init__(self,
                 latent_dim,
                 num_tokens,
                 dim=1,
                 commitment=0.25,
                 mode='angular',
                 return_indices=True):
        super(VQ, self).__init__()
        self.embedding = nn.Embedding(num_tokens, latent_dim)
        nn.init.normal_(self.embedding.weight, 0, 0.2)
        self.dim = dim
        self.commitment = commitment
        self.register_buffer('initialized', torch.ByteTensor([0]))
        assert mode in ['nearest', 'angular']
        self.mode = mode
        self.return_indices = return_indices

    def forward(self, x):
        dim = self.dim
        nb_codes = self.embedding.weight.shape[0]

        codebook = self.embedding.weight
        if self.mode == 'angular':
            codebook = F.normalize(codebook)
            x = F.normalize(x, dim=dim)

        if self.initialized.item() == 0 and self.training:
            ch_first = x.transpose(dim, -1).contiguous().view(-1, x.shape[1])
            idx = torch.randperm(ch_first.shape[0])[:nb_codes]
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
