import torchelie.loss.gan

from .perceptualloss import PerceptualLoss

import torch

def ortho(w):
    cosine = torch.mm(w, w.t())
    no_diag = (1 - torch.eye(w.shape[0], device=w.device))
    return (cosine * no_diag).pow(2).sum(dim=1).mean()
