r"""
Hinge loss from Spectral Normalization GAN.

https://arxiv.org/abs/1802.05957

:math:`L_D(x_r, x_f) = \text{max}(0, 1 - D(x_r)) + \text{max}(0, 1 + D(x_f))`

:math:`L_G(x_f) = -D(x_f)`
"""

import torch
import torch.nn.functional as F


def real(x):
    return F.relu(1 - x).mean()


def fake(x):
    return F.relu(1 + x).mean()


def generated(x):
    return -x.mean()
