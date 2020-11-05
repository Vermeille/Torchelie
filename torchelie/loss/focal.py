import torch
import torch.nn as nn
from .functional import focal_loss


class FocalLoss(nn.Module):
    """
    The focal loss

    https://arxiv.org/abs/1708.02002

    See :func:`torchelie.loss.focal_loss` for details.
    """
    def __init__(self, gamma=0):
        """
        Initialize the gradient

        Args:
            self: (todo): write your description
            gamma: (float): write your description
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            target: (todo): write your description
        """
        return focal_loss(input, target, self.gamma)
