import torch
import torch.nn.functional as F


def ortho(w):
    """
    Returns the orthogonal loss for weight matrix `m`, from Big GAN.

    https://arxiv.org/abs/1809.11096
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
    """
    Returns the focal loss between `target` and `input`
    """
    logp = nn.functional.cross_entropy(input, target)
    p = torch.exp(-logp)
    loss = (1 - p)**gamma * logp
    return loss.mean()


def continuous_cross_entropy(pred, soft_targets):
    """
    Compute the cross entropy between the logits `pred` and a normalized
    distribution `soft_targets`. If `soft_targets` is a one-hot vector, this is
    equivalent to `nn.functional.cross_entropy` with a label
    """
    return torch.mean(torch.sum(-soft_targets * F.log_softmax(pred), 1))
