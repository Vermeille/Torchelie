import torch
import torch.nn as nn
from torch.distributions import TransformedDistribution


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
        return torch.logsumexp(self.logistics.log_prob(x.unsqueeze(self.dim))
                               + log_pis,
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
        return torch.logsumexp(self.logistics.log_prob(x.unsqueeze(1))
                               + log_pis,
                               dim=1)
