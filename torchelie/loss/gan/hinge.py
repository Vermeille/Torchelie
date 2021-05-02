r"""
Hinge loss from Spectral Normalization GAN.

https://arxiv.org/abs/1802.05957

:math:`L_D(x_r, x_f) = \text{max}(0, 1 - D(x_r)) + \text{max}(0, 1 + D(x_f))`

:math:`L_G(x_f) = -D(x_f)`
"""

import torch
import torch.nn.functional as F


def real(x: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    out = F.relu(1 - x)
    if reduction == 'none':
        return out
    if reduction == 'mean':
        return out.mean()
    if reduction == 'sum':
        return out.sum()
    assert False, f'{reduction} is not a valid reduction method'


def fake(x: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    out = F.relu(1 + x)
    if reduction == 'none':
        return out
    if reduction == 'mean':
        return out.mean()
    if reduction == 'sum':
        return out.sum()
    assert False, f'{reduction} is not a valid reduction method'


def generated(x: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    out = -x
    if reduction == 'none':
        return out
    if reduction == 'mean':
        return out.mean()
    if reduction == 'sum':
        return out.sum()
    assert False, f'{reduction} is not a valid reduction method'
