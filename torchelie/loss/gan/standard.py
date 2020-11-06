"""
Standard, non saturating, GAN loss from the original GAN paper

https://arxiv.org/abs/1406.2661

:math:`L_D(x_r, x_f) = - \log(1 - D(x_f)) - \log D(x_r)`

:math:`L_G(x_f) = -\log D(x_f)`
"""

import torch
import torch.nn.functional as F


def real(x):
    return F.softplus(-x).mean()


def fake(x):
    return F.softplus(x).mean()


def generated(x):
    return F.softplus(-x).mean()
