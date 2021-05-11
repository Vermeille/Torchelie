r"""
Least Square GAN
"""

import torch
import torch.nn.functional as F


def real(x: torch.Tensor, reduce: str = 'mean') -> torch.Tensor:
    out = F.mse_loss(x, torch.ones_like(x), reduce='none')
    if reduce == 'none':
        return out
    if reduce == 'mean':
        return out.mean()
    if reduce == 'batch_mean':
        return out.view(out.shape[0], -1).sum(1).mean()
    assert False, f'reduction {reduce} invalid'


def fake(x: torch.Tensor, reduce: str = 'mean') -> torch.Tensor:
    out = F.mse_loss(x, -torch.ones_like(x), reduce='none')
    if reduce == 'none':
        return out
    if reduce == 'mean':
        return out.mean()
    if reduce == 'batch_mean':
        return out.view(out.shape[0], -1).sum(1).mean()
    assert False, f'reduction {reduce} invalid'


def generated(x: torch.Tensor, reduce: str = 'mean') -> torch.Tensor:
    out = F.mse_loss(x, torch.ones_like(x), reduce='none')
    if reduce == 'none':
        return out
    if reduce == 'mean':
        return out.mean()
    if reduce == 'batch_mean':
        return out.view(out.shape[0], -1).sum(1).mean()
    assert False, f'reduction {reduce} invalid'
