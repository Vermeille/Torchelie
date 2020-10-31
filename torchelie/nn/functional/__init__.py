import torch
import torch.nn.functional as F
from torch.autograd import Function
from .vq import quantize


def laplacian(images, n_down=4):
    """
    Decompose a 4D images tensor into a laplacian pyramid.

    Args:
        images (4D Tensor): source images
        n_down (int): how many times to downscale the input

    Returns:
        A list of tensor: laplacian pyramid
    """
    lapls = []

    for i in range(n_down):
        n = F.interpolate(images,
                          scale_factor=0.5,
                          mode='bilinear',
                          align_corners=True)
        lapls.append(images - F.interpolate(
            n, size=images.shape[-2:], mode='bilinear', align_corners=True))
        images = n

    lapls.append(images)
    return lapls


def combine_laplacians(laplacians):
    """
    Recombine a list of of tensor as returned by :code:`laplacian()` into an
    image batch

    Args:
        laplacians (list of tensors: laplacian pyramid

    Returns:
        a tensor
    """
    biggest = laplacians[0]

    rescaled = [biggest]
    for im in laplacians[1:]:
        rescaled.append(
            F.interpolate(im,
                          size=biggest.shape[-2:],
                          mode='bilinear',
                          align_corners=True))

    mixed = torch.stack(rescaled, dim=-1)
    return mixed.sum(dim=-1)


class InformationBottleneckFunc(Function):
    @staticmethod
    def forward(ctx, mu, sigma, strength=1):
        z = torch.randn_like(mu)
        x = mu + z * sigma
        s = torch.tensor(strength, device=x.device)
        ctx.save_for_backward(mu, sigma, z, s)
        return x

    @staticmethod
    def backward(ctx, d_out):
        mu, sigma, z, strength = ctx.saved_tensors

        # kl = -0.5 * (1 + log(sigma^2) - mu^2 - sigma^2)
        # dkl/dmu = -0.5 * (-2*mu)
        #         = mu
        # dkl/dsig = -0.5 * (d 2*log(sigma) - d sigma^2)
        #          = -0.5 * (2/sigma - 2*sigma)
        #          = -1/sigma + sigma
        d_mu = d_out + strength * mu
        d_sigma = z * d_out
        d_sigma += -strength / sigma + strength * sigma
        return d_mu, d_sigma, None


information_bottleneck = InformationBottleneckFunc.apply
unit_gaussian_prior = InformationBottleneckFunc.apply
