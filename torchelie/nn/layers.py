import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchelie.utils as tu
import torchelie as tch
from typing import List, Optional, Tuple, Union
from .interpolate import InterpolateBilinear2d


def Conv2d(in_ch, out_ch, ks, stride=1, bias=True):
    """
    A Conv2d with 'same' padding
    """
    return nn.Conv2d(in_ch,
                     out_ch,
                     ks,
                     padding=ks // 2,
                     stride=stride,
                     bias=bias)


def Conv3x3(in_ch: int,
            out_ch: int,
            stride: int = 1,
            bias: bool = True) -> nn.Conv2d:
    """
    A 3x3 Conv2d with 'same' padding
    """
    return Conv2d(in_ch, out_ch, 3, stride=stride, bias=bias)


def Conv1x1(in_ch: int,
            out_ch: int,
            stride: int = 1,
            bias: bool = True) -> nn.Conv2d:
    """
    A 1x1 Conv2d
    """
    return Conv2d(in_ch, out_ch, 1, stride=stride, bias=bias)


class AdaptiveConcatPool2d(nn.Module):
    """
    Pools with AdaptiveMaxPool2d AND AdaptiveAvgPool2d and concatenates both
    results.

    Args:
        target_size: the target output size (single integer or
            double-integer tuple)
    """
    def __init__(self, target_size):
        super(AdaptiveConcatPool2d, self).__init__()
        self.target_size = target_size

    def forward(self, x):
        return torch.cat([
            nn.functional.adaptive_avg_pool2d(x, self.target_size),
            nn.functional.adaptive_max_pool2d(x, self.target_size),
        ],
                         dim=1)


class ModulatedConv(nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 noise_channels: int,
                 *args,
                 demodulate: bool = True,
                 **kwargs):
        super(ModulatedConv, self).__init__(in_channels, *args, **kwargs)
        self.make_s = tu.xavier(nn.Linear(noise_channels, in_channels))
        self.make_s.bias.data.fill_(1)
        self.demodulate = demodulate

    def condition(self, z: torch.Tensor) -> None:
        self.s = self.make_s(z)

    def forward(self,
                x: torch.Tensor,
                z: Optional[torch.Tensor] = None) -> torch.Tensor:
        if z is not None:
            self.condition(z)
        N, C, H, W = x.shape
        C_out, C_in = self.weight.shape[:2]
        w_prime = torch.einsum('oihw,bi->boihw', self.weight, self.s)
        if self.demodulate:
            w_prime_prime = torch.einsum('boihw,boihw->bo', w_prime, w_prime)
            w_prime_prime = w_prime_prime.add_(1e-8).rsqrt()
            w = w_prime * w_prime_prime[..., None, None, None]
        else:
            w = w_prime

        w = w.view(-1, *w.shape[2:])
        x = F.conv2d(x.view(1, -1, H, W), w, None, self.stride, self.padding,
                     self.dilation, N)
        x = x.view(N, C_out, H, W)
        if self.bias is not None:
            return x.add_(self.bias.view(-1, 1, 1))
        else:
            return x


class SelfAttention2d(nn.Module):
    """
    Self Attention such as used in SAGAN or BigGAN.

    Args:
        ch (int): number of input / output channels
    """
    def __init__(self, ch: int):
        super().__init__()
        self.key = nn.Conv1d(ch, ch // 8, 1)
        self.query = nn.Conv1d(ch, ch // 8, 1)
        self.value = nn.Conv1d(ch, ch, 1)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward
        """
        x_flat = x.view(*x.shape[:2], -1)
        k = self.key(x_flat)
        q = self.query(x_flat)
        v = self.value(x_flat)

        affinity = torch.einsum('bki,bkj->bij', q, k)
        attention = F.softmax(affinity, dim=1)
        out = torch.einsum('bci,bih->bch', v, affinity).view(*x.shape)
        return self.gamma * out + x


from torch.autograd import Function


class GaussianPriorFunc(Function):
    @staticmethod
    def forward(ctx, mu, sigma, mu2, sigma2, strength=1):
        z = torch.randn_like(mu)
        x = mu + z * sigma
        s = torch.tensor(strength, device=x.device)
        ctx.save_for_backward(mu, sigma, z, mu2, sigma2, s)
        return x

    @staticmethod
    def backward(ctx, d_out):
        mu, sigma, z, mu2, sigma2, strength = ctx.saved_tensors

        # kl = -0.5
        #      + log(sigma2) - log(sigma1)
        #      + (sigma1 ** 2) / (2 * sigma2 ** 2)
        #      + ((mu1 - mu2) ** 2) / (2 * sigma2 ** 2)
        # dkl/dmu  =  (mu - mu2) / (sigma**2)
        # dkl/dmu2 = -(mu - mu2) / (sigma**2)
        # dkl/dsig  = -1/sigma + sigma/(sigma2**2)
        # dkl/dsig2 = 1/sigma2 - ((sigma**2) + ((mu-mu2)**2)) / (sigma2 ** 3)
        diff_mu = mu - mu2
        s2sq = sigma2.pow(2)
        diff_mu_over_s2sq = diff_mu / s2sq
        d_mu = d_out + strength * diff_mu_over_s2sq
        d_mu2 = -strength * diff_mu_over_s2sq
        d_sigma = d_out * z - strength / sigma + strength * sigma / s2sq
        d_sigma2 = 1 / sigma2 - (sigma.pow(2) + diff_mu.pow(2)) / sigma2.pow(3)
        d_sigma2 *= strength
        return d_mu, d_sigma, d_mu2, d_sigma2


class UnitGaussianPrior(nn.Module):
    """
    Force a representation to fit a unit gaussian prior. It projects with a
    nn.Linear the input vector to a mu and sigma that represent a gaussian
    distribution from which the output is sampled. The backward pass includes a
    kl divergence loss between N(mu, sigma) and N(0, 1).

    This can be used to implement VAEs or information bottlenecks

    In train mode, the output is sampled from N(mu, sigma) but at test time mu
    is returned.

    Args:
        in_channels (int): dimension of input channels
        num_latents (int): dimension of output latents
        strength (float): strength of the kl loss. When using this to implement
            a VAE, set strength to :code:`1/number of output dim of the model`
            or set it to 1 but make sure that the loss for each output
            dimension is summed, but averaged over the batch.
        kl_reduction (str): how the implicit kl loss is reduced over the batch
            samples. 'sum' means the kl term of each sample is summed, while
            'mean' divides the loss by the number of examples.
    """
    def __init__(self,
                 in_channels,
                 num_latents,
                 strength=1,
                 kl_reduction='mean'):
        super().__init__()
        self.project = tu.kaiming(nn.Linear(in_channels, 2 * num_latents))
        self.project.bias.data[num_latents:].fill_(1)
        self.strength = strength
        assert kl_reduction in ['mean', 'sum']
        self.reduction = kl_reduction

    def forward(self, x):
        """
        Args:
            x (Tensor): A 2D (N, in_channels) tensor

        Returns:
            A 2D (N, num_channels) tensor sampled from the implicit gaussian
                distribution.
        """
        x = self.project(x)
        mu, sigma = torch.chunk(x, 2, dim=1)
        if self.training:
            sigma = torch.exp(0.5 * sigma)
            strength = self.strength
            if self.reduction == 'mean':
                strength = strength / x.shape[0]
            return tch.nn.functional.unit_gaussian_prior(mu, sigma, strength)
        else:
            return mu


class InformationBottleneck(UnitGaussianPrior):
    pass


class Const(nn.Module):
    """
    Return a constant learnable volume. Disregards the input except its batch
    size

    Args:
        *size (ints): the shape of the volume to learn
    """
    def __init__(self, *size: int) -> None:
        super().__init__()
        self.size = size
        self.weight = nn.Parameter(torch.randn(1, *size))

    def extra_repr(self):
        return repr(self.size)

    def forward(self, n: int) -> torch.Tensor:
        """
        Args:
            n (int): batch size to use
        """
        return self.weight.expand(n, *self.weight.shape[1:]).contiguous()


class MinibatchStddev(nn.Module):
    """Minibatch Stddev layer from Progressive GAN"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stddev_map = torch.sqrt(x.var(dim=0) + 1e-8).mean()
        stddev = stddev_map.expand(x.shape[0], 1, *x.shape[2:])
        return torch.cat([x, stddev], dim=1)
