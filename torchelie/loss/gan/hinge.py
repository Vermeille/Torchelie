r"""
Hinge loss from Spectral Normalization GAN.

https://arxiv.org/abs/1802.05957

:math:`L_D(x_r, x_f) = \text{max}(0, 1 - D(x_r)) + \text{max}(0, 1 + D(x_f))`

:math:`L_G(x_f) = -D(x_f)`
"""

import torch
import torch.nn.functional as F


def real(x, reduction='mean'):
    """
    Calculate mean of x.

    Args:
        x: (todo): write your description
        reduction: (todo): write your description
    """
    out = F.relu(1 - x)
    if reduction == 'none':
        return out
    if reduction == 'mean':
        return out.mean()
    if reduction == 'sum':
        return out.sum()


def fake(x, reduction='mean'):
    """
    Fake - 1d function.

    Args:
        x: (todo): write your description
        reduction: (todo): write your description
    """
    out = F.relu(1 + x)
    if reduction == 'none':
        return out
    if reduction == 'mean':
        return out.mean()
    if reduction == 'sum':
        return out.sum()


def generated(x, reduction='mean'):
    """
    Calculate the kl divergence.

    Args:
        x: (str): write your description
        reduction: (str): write your description
    """
    out = -x
    if reduction == 'none':
        return out
    if reduction == 'mean':
        return out.mean()
    if reduction == 'sum':
        return out.sum()
