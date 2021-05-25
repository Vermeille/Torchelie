import math

import torch
import torch.nn as nn
from torch.distributions import TransformedDistribution

from .utils import experimental


class Logistic(TransformedDistribution):
    """
    Logistic distribution

    Args:
        loc (tensor): mean of the distribution
        scale (tensor): scale of the distribution
    """

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor) -> None:
        td = torch.distributions
        base_distribution = td.Uniform(torch.zeros_like(loc),
                                       torch.ones_like(loc))
        transforms = [
            td.SigmoidTransform().inv,
            td.AffineTransform(loc=loc, scale=scale)
        ]
        super(Logistic, self).__init__(base_distribution, transforms)


class LogisticMixture:
    """
    Mixture of Logistic distributions. Each tensor contains an additional
    dimension with `number of distributions` elements.

    Args:
        weights (tensor): un-normalized weights of distributions
        loc (tensor): mean of the distributions
        scale (tensor): scale of the distributions
        dim (int): dimension reprenseting the various distributions, that will
            weighted and averaged on.
    """

    def __init__(self, weights, locs, scales, dim) -> None:
        self.weights = weights
        self.logistics = Logistic(locs, scales)
        self.locs = locs
        self.dim = dim - len(locs.shape) if dim >= 0 else dim

    @property
    def mean(self) -> torch.Tensor:
        w = nn.functional.softmax(self.weights, dim=self.dim)
        return torch.sum(self.locs * w, dim=self.dim)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_pis = nn.functional.log_softmax(self.weights, dim=self.dim)
        return torch.logsumexp(self.logistics.log_prob(x.unsqueeze(self.dim)) +
                               log_pis,
                               dim=self.dim)


class GaussianMixture:
    """
    Mixture of gaussian distributions. Each tensor contains an additional
    dimension with `number of distributions` elements.

    Args:
        weights (tensor): un-normalized weights of distributions
        loc (tensor): mean of the distributions
        scale (tensor): scale of the distributions
        dim (int): dimension reprenseting the various distributions, that will
            weighted and averaged on.
    """

    def __init__(self, weights: torch.Tensor, locs: torch.Tensor,
                 scales: torch.Tensor) -> None:
        self.weights = weights
        self.logistics = torch.distributions.Normal(locs, scales)
        self.locs = locs

    @property
    def mean(self) -> torch.Tensor:
        w = nn.functional.softmax(self.weights, dim=1)
        return torch.sum(self.locs * w, dim=1)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_pis = nn.functional.log_softmax(self.weights, dim=1)
        return torch.logsumexp(self.logistics.log_prob(x.unsqueeze(1)) +
                               log_pis,
                               dim=1)


@experimental
def parameterized_truncated_normal(uniform: torch.Tensor, mu: float,
                                   sigma: float, a: float,
                                   b: float) -> torch.Tensor:
    normal = torch.distributions.normal.Normal(0, 1)

    alpha = torch.tensor((a - mu) / sigma)
    beta = torch.tensor((b - mu) / sigma)

    alpha_normal_cdf = normal.cdf(alpha)
    p = alpha_normal_cdf + (normal.cdf(beta) - alpha_normal_cdf) * uniform

    one = torch.tensor(1, dtype=p.dtype)
    epsilon = 1e-8
    v = torch.clamp(2 * p - 1, -one + epsilon, one - epsilon)
    x = mu + sigma * math.sqrt(2) * torch.erfinv(v)
    x = torch.clamp(x, a, b)

    return x


@experimental
def truncated_normal(uniform: torch.Tensor) -> torch.Tensor:
    return parameterized_truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2)


@experimental
def sample_truncated_normal(*shape):
    return truncated_normal(torch.rand(shape))
