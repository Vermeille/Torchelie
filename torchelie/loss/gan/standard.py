r"""
Standard, non saturating, GAN loss from the original GAN paper

https://arxiv.org/abs/1406.2661

:math:`L_D(x_r, x_f) = - \log(1 - D(x_f)) - \log D(x_r)`

:math:`L_G(x_f) = -\log D(x_f)`
"""

import torch
import torch.nn.functional as F


def real(x: torch.Tensor, reduce: str = 'mean') -> torch.Tensor:
    out = F.softplus(-x)
    if reduce == 'none':
        return out
    if reduce == 'mean':
        return out.mean()
    if reduce == 'batch_mean':
        return out.view(out.shape[0], -1).sum(1).mean()
    assert False, f'reduction {reduce} invalid'


def fake(x: torch.Tensor, reduce: str = 'mean') -> torch.Tensor:
    out = F.softplus(x)
    if reduce == 'none':
        return out
    if reduce == 'mean':
        return out.mean()
    if reduce == 'batch_mean':
        return out.view(out.shape[0], -1).sum(1).mean()
    assert False, f'reduction {reduce} invalid'


def generated(x: torch.Tensor, reduce: str = 'mean') -> torch.Tensor:
    out = F.softplus(-x)
    if reduce == 'none':
        return out
    if reduce == 'mean':
        return out.mean()
    if reduce == 'batch_mean':
        return out.view(out.shape[0], -1).sum(1).mean()
    assert False, f'reduction {reduce} invalid'
