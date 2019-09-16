import torch
import torch.nn as nn
import torch.nn.functional as F


def ortho(w):
    r"""
    Returns the orthogonal loss for weight matrix `m`, from Big GAN.

    https://arxiv.org/abs/1809.11096

    :math:`R_{\beta}(W)= ||W^T W  \odot (1 - I)||_F^2`
    """
    cosine = torch.mm(w, w.t())
    no_diag = (1 - torch.eye(w.shape[0], device=w.device))
    return (cosine * no_diag).pow(2).sum(dim=1).mean()


def total_variation(i):
    """
    Returns the total variation loss for batch of images `i`
    """
    v = F.l1_loss(i[:, :, 1:, :], i[:, :, :-1, :])
    h = F.l1_loss(i[:, :, :, 1:], i[:, :, :, :-1])
    return v + h


def focal_loss(input, target, gamma=0):
    r"""
    Returns the focal loss between `target` and `input`

    :math:`\text{FL}(p_t)=-(1-p_t)^\gamma\log(p_t)`
    """
    if input.shape[1] == 1:
        logp = nn.functional.binary_cross_entropy_with_logits(input, target)
    else:
        logp = nn.functional.cross_entropy(input, target)
    p = torch.exp(-logp)
    loss = (1 - p)**gamma * logp
    return loss.mean()


def continuous_cross_entropy(pred, soft_targets):
    r"""
    Compute the cross entropy between the logits `pred` and a normalized
    distribution `soft_targets`. If `soft_targets` is a one-hot vector, this is
    equivalent to `nn.functional.cross_entropy` with a label
    """
    return torch.mean(torch.sum(-soft_targets * F.log_softmax(pred, 1), 1))
