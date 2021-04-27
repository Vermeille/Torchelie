import torch
import torch.nn as nn
from .functional import focal_loss


class FocalLoss(nn.Module):
    """
    The focal loss

    https://arxiv.org/abs/1708.02002

    See :func:`torchelie.loss.focal_loss` for details.
    """
    def __init__(self, gamma: float = 0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.gamma)
