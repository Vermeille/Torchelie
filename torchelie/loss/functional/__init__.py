from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchelie.utils import experimental


def ortho(w: torch.Tensor) -> torch.Tensor:
    r"""
    Returns the orthogonal loss for weight matrix `m`, from Big GAN.

    https://arxiv.org/abs/1809.11096

    :math:`R_{\beta}(W)= ||W^T W  \odot (1 - I)||_F^2`
    """
    cosine = torch.einsum('ij,ji->ij', w, w)
    no_diag = (1 - torch.eye(w.shape[0], device=w.device))
    return (cosine * no_diag).pow(2).sum(dim=1).mean()


def total_variation(i: torch.Tensor) -> torch.Tensor:
    """
    Returns the total variation loss for batch of images `i`
    """
    v = F.l1_loss(i[:, :, 1:, :], i[:, :, :-1, :])
    h = F.l1_loss(i[:, :, :, 1:], i[:, :, :, :-1])
    return v + h


def focal_loss(input: torch.Tensor,
               target: torch.Tensor,
               gamma: float = 0,
               weight: Optional[torch.Tensor] = None,
               dim: int = -1) -> torch.Tensor:
    r"""
    Returns the focal loss between `target` and `input`

    :math:`\text{FL}(p_t)=-(1-p_t)^\gamma\log(p_t)`
    """
    if input.shape[dim if dim != -1 else 1] == 1:
        logp = nn.functional.binary_cross_entropy_with_logits(input, target)
    else:
        logp = nn.functional.cross_entropy(input,
                                           target,
                                           weight=weight,
                                           dim=dim)
    p = torch.exp(-logp)
    loss = (1 - p)**gamma * logp
    return loss.mean()


def continuous_cross_entropy(pred: torch.Tensor,
                             soft_targets: torch.Tensor,
                             weights: Optional[torch.Tensor] = None,
                             reduction: str = 'mean') -> torch.Tensor:
    r"""
    Compute the cross entropy between the logits `pred` and a normalized
    distribution `soft_targets`. If `soft_targets` is a one-hot vector, this is
    equivalent to `nn.functional.cross_entropy` with a label
    """
    if weights is None:
        ce = torch.sum(-soft_targets * F.log_softmax(pred, 1), 1)
    else:
        ce = torch.sum(-weights * soft_targets * F.log_softmax(pred, 1), 1)

    if reduction == 'mean':
        return ce.mean()
    if reduction == 'sum':
        return ce.sum()
    if reduction == 'none':
        return ce
    assert False, f'{reduction} not a valid reduction method'


@experimental
def smoothed_cross_entropy(pred: torch.Tensor,
                           targets: torch.tensor,
                           smoothing: float = 0.9):
    """
    Cross entropy with label smoothing

    Args:
        pred (FloatTensor): a 2D logits prediction
        targets (LongTensor): 1D indices
        smoothing (float): target probability value for the correct class
    """
    prob = F.log_softmax(pred)
    n_classes = pred.shape[1]
    wrong_prob = (1 - smoothing) / (n_classes - 1)

    wrong_sum = prob.sum(1) * wrong_prob
    good = pred.gather(0, targets.unsqueeze(0)).squeeze(0)
    good *= (smoothing - wrong_prob)
    return -torch.mean(wrong_sum + good)
