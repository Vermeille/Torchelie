import torch
from torch.autograd import Function


class VectorQuantization(Function):

    @staticmethod
    def compute_indices(inputs_orig, codebook):
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
        code_dim = x.size(-1)
        return x.view(-1, code_dim)

    @staticmethod
    def restore_shapes(codes, indices, target_shape):
        idx_shape = list(target_shape)
        idx_shape[-1] = 1
        return codes.view(*target_shape), indices.view(*idx_shape)

    @staticmethod
    def forward(ctx, inputs, codebook, commitment=0.25, dim=1):
        inputs_flat = VectorQuantization.flatten(inputs)
        indices = VectorQuantization.compute_indices(inputs_flat, codebook)
        codes = codebook[indices.view(-1), :]
        codes, indices = VectorQuantization.restore_shapes(
            codes, indices, inputs.shape)

        ctx.save_for_backward(codes, inputs, torch.tensor([float(commitment)]),
                              codebook, indices)
        ctx.mark_non_differentiable(indices)
        return codes, indices

    @staticmethod
    def backward(ctx, straight_through, unused_indices):
        codes, inputs, beta, codebook, indices = ctx.saved_tensors

        # TODO: figure out proper vq loss reduction
        # vq_loss = F.mse_loss(inputs, codes).detach()

        # gradient of vq_loss
        diff = 2 * (inputs - codes) / inputs.numel()

        commitment = beta.item() * diff

        code_disp = VectorQuantization.flatten(-diff)
        indices = VectorQuantization.flatten(indices)
        code_disp = (torch.zeros_like(codebook).index_add_(
            0, indices.view(-1), code_disp))
        return straight_through + commitment, code_disp, None, None


quantize = VectorQuantization.apply
