import torch
import torch.nn as nn
from .functional import focal_loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        return focal_loss(input, target, self.gamma)
