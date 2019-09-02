import torch
import torch.nn as nn

import torchelie.loss.gan
from .functional import ortho, total_variation, continuous_cross_entropy
from .functional import focal_loss

from .perceptualloss import PerceptualLoss
from .neuralstyleloss import NeuralStyleLoss
from .focal import FocalLoss


class OrthoLoss(nn.Module):
    """
    Orthogonal loss

    See :func:`torchelie.loss.ortho` for details.
    """
    def forward(self, w):
        return ortho(w)


class TotalVariationLoss(nn.Module):
    """
    Total Variation loss

    See :func:`torchelie.loss.total_variation` for details.
    """
    def forward(self, x):
        return total_variation(x)


class ContinuousCEWithLogits(nn.Module):
    """
    Cross Entropy loss accepting continuous target values

    See :func:`torchelie.loss.continuous_cross_entropy` for details.
    """
    def forward(self, pred, soft_targets):
        return continuous_cross_entropy(pred, soft_targets)
