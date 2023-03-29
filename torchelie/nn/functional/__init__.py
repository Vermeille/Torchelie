import torch
import torch.nn.functional as F
from torch.autograd import Function
from .vq import quantize
from .transformer import local_attention_2d


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


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x * random_tensor.div(keep_prob)
    return output


class SquaReLUFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, inplace=False):
        x = F.relu(input, inplace=inplace)
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * 2 * x, None


squarelu = SquaReLUFunc.apply


class StaReLUFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, inplace=False):
        x = F.relu(input)

        x = x * x * weight
        if bias is not None:
            x += bias

        if inplace:
            input.copy_(x)
            ctx.save_for_backward(input, weight, bias)
            return input
        ctx.save_for_backward(x, weight, bias)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        out, w, b = ctx.saved_tensors
        if b is None:
            d_bias = None
        else:
            d_bias = torch.sum(grad_output)
            out -= b

        out /= w
        d_weight = torch.sum(grad_output * out)
        d_input = grad_output * 2 * out.sqrt() * w
        return d_input, d_weight, d_bias, None


starelu = StaReLUFunc.apply
