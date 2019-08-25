import torch
import torch.nn as nn

import torchelie.loss.gan
from .functional import ortho, total_variation, continuous_cross_entropy

from .perceptualloss import PerceptualLoss
from .neuralstyleloss import NeuralStyleLoss
from .focal import FocalLoss


class OrthoLoss(nn.Module):
    def forward(self, w):
        return ortho(w)


class TotalVariationLoss(nn.Module):
    def forward(self, x):
        return total_variation(x)


class ContinuousCEWithLogits(nn.Module):
    def forward(self, pred, soft_targets):
        return continuous_cross_entropy(pred, soft_targets)
