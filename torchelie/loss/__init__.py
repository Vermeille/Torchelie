import torch
import torch.nn as nn

import torchelie.loss.gan
from .functional import ortho, total_variation, continuous_cross_entropy
from .functional import focal_loss

from .perceptualloss import PerceptualLoss
from .neuralstyleloss import NeuralStyleLoss
from .deepdreamloss import DeepDreamLoss
from .focal import FocalLoss
from .bitempered import tempered_cross_entropy, TemperedCrossEntropyLoss
from .bitempered import tempered_softmax, tempered_nll_loss
from .bitempered import tempered_log_softmax


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


def binary_hinge(x, y):
    p = y * torch.clamp(1 - x, min=0) + (1 - y) * torch.clamp(1 + x, min=0)
    return p.mean()
