import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class VectorQuantization(Function):
    @staticmethod
    def compute_indices(inputs_orig, codebook):
        """
        Compute the indices of the indices.

        Args:
            inputs_orig: (todo): write your description
            codebook: (str): write your description
        """
        bi = []
        SZ = 10000
        for i in range(0, inputs_orig.size(0), SZ):
            inputs = inputs_orig[i:i + SZ]
            # NxK
            distances_matrix = torch.cdist(inputs, codebook)
            # Nx1
            indic = torch.min(distances_matrix, dim=-1)[1].unsqueeze(1)
            bi.append(indic)
        return torch.cat(bi, dim=0)

    @staticmethod
    def flatten(x):
        """
        Flatten a nested array.

        Args:
            x: (todo): write your description
        """
        code_dim = x.size(-1)
        return x.view(-1, code_dim)

    @staticmethod
    def restore_shapes(codes, indices, target_shape):
        """
        Restore shapes of target_shape.

        Args:
            codes: (todo): write your description
            indices: (array): write your description
            target_shape: (todo): write your description
        """
        idx_shape = list(target_shape)
        idx_shape[-1] = 1
        return codes.view(*target_shape), indices.view(*idx_shape)

    @staticmethod
    def forward(ctx, inputs, codebook, commitment=0.25, dim=1):
        """
        Generate batch of inputs.

        Args:
            ctx: (todo): write your description
            inputs: (todo): write your description
            codebook: (str): write your description
            commitment: (todo): write your description
            dim: (int): write your description
        """
        inputs_flat = VectorQuantization.flatten(inputs)
        indices = VectorQuantization.compute_indices(inputs_flat, codebook)
        codes = codebook[indices.view(-1), :]
        codes, indices = VectorQuantization.restore_shapes(
            codes, indices, inputs.shape)

        ctx.save_for_backward(codes, inputs, torch.FloatTensor([commitment]),
                              codebook, indices)
        ctx.mark_non_differentiable(indices)
        return codes, indices

    @staticmethod
    def backward(ctx, straight_through, unused_indices):
        """
        Compute backward backends.

        Args:
            ctx: (todo): write your description
            straight_through: (todo): write your description
            unused_indices: (todo): write your description
        """
        codes, inputs, beta, codebook, indices = ctx.saved_tensors

        # TODO: figure out proper vq loss reduction
        vq_loss = F.mse_loss(inputs, codes).detach()

        # gradient of vq_loss
        diff = 2 * (inputs - codes) / inputs.numel()

        commitment = beta.item() * diff

        code_disp = VectorQuantization.flatten(-diff)
        indices = VectorQuantization.flatten(indices)
        code_disp = (torch.zeros_like(codebook).index_add_(
            0, indices.view(-1), code_disp))
        return straight_through + commitment, code_disp, None, None


try:
    # Sphinx doesn't import Function, so apply ain't defined
    quantize = VectorQuantization.apply
except Exception as e:
    print(str(e))
    quantize = lambda x:x
