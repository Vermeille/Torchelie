import torch
import torch.nn as nn
import torchelie.loss.functional as tlf


class FocalLoss(nn.Module):
    def __init__(self, gamma=0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        return tlf.focal_loss(input, target, self.gamma)
