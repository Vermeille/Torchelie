import torch
import torch.nn.functional as F


def real(x):
    return F.binary_cross_entropy_with_logits(x, torch.ones_like(x))


def fake(x):
    return F.binary_cross_entropy_with_logits(x, torch.zeros_like(x))


def generated(x):
    return F.binary_cross_entropy_with_logits(x, torch.ones_like(x))
