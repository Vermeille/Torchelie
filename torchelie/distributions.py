import torch
import torch.nn as nn


class Logistic(torch.distributions.TransformedDistribution):
    """
    Logistic distribution

    Args:
        loc (tensor): mean of the distribution
        scale (tensor): scale of the distribution
    """
    def __init__(self, loc, scale):
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
    def __init__(self, weights, locs, scales, dim):
        self.weights = weights
        self.logistics = Logistic(locs, scales)
        self.locs = locs
        self.dim = dim

    @property
    def mean(self):
        w = nn.functional.softmax(self.weights, dim=self.dim)
        return torch.sum(self.locs * w, dim=self.dim)

    def log_prob(self, x):
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
    def __init__(self, weights, locs, scales):
        self.weights = weights
        self.logistics = torch.distributions.Normal(locs, scales)
        self.locs = locs

    @property
    def mean(self):
        w = nn.functional.softmax(self.weights, dim=1)
        return torch.sum(self.locs * w, dim=1)

    def log_prob(self, x):
        log_pis = nn.functional.log_softmax(self.weights, dim=1)
        return torch.logsumexp(self.logistics.log_prob(x.unsqueeze(1)) +
                               log_pis,
                               dim=1)
