import torch
import torch.nn as nn
import torch.nn.functional as F

from torchelie.nn import WithSavedActivations
from torchelie.nn import ImageNetInputNorm
from torchelie.models import PerceptualNet


class PerceptualLoss(nn.Module):
    def __init__(self, l, rescale=False, loss=F.mse_loss):
        super(PerceptualLoss, self).__init__()
        self.m = WithSavedActivations(PerceptualNet(l))
        self.norm = ImageNetInputNorm()
        self.rescale = rescale
        self.loss = loss

    def forward(self, x, y):
        if self.rescale:
            y = F.interpolate(y, size=(224, 224), mode='nearest')
            x = F.interpolate(x, size=(224, 224), mode='nearest')

        ref = self.m(self.norm(y), detach=True)
        acts = self.m(self.norm(x), detach=False)
        loss = 0
        for k in acts.keys():
            loss += self.loss(acts[k], ref[k])
        return loss
