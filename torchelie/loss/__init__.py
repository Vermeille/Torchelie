import torch
import torch.nn as nn

import torchelie.loss.gan
import torchelie.loss.functional as tlf

from .perceptualloss import PerceptualLoss
from .neuralstyleloss import NeuralStyleLoss
from .focal import FocalLoss


class OrthoLoss(nn.Module):
    def forward(self, w):
        return tlf.ortho(w)


class TotalVariationLoss(nn.Module):
    def forward(self, x):
        return tlf.total_variation(x)
