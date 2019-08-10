import torch
import torch.nn.functional as F

import torchelie.loss.gan

from .perceptualloss import PerceptualLoss
from .neuralstyleloss import NeuralStyleLoss


def ortho(w):
    cosine = torch.mm(w, w.t())
    no_diag = (1 - torch.eye(w.shape[0], device=w.device))
    return (cosine * no_diag).pow(2).sum(dim=1).mean()


def total_variation(i):
    v = F.l1_loss(i[:, :, 1:, :] - i[:, :, :-1, :])
    h = F.l1_loss(i[:, :, :, 1:] - i[:, :, :, :-1])
    return v + h
