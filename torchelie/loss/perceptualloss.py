import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M

from torchelie.nn import WithSavedActivations
from torchelie.nn import ImageNetInputNorm


def PerceptualNet(l):
    m = M.vgg16(pretrained=True).eval()
    m = m.features[:l]
    m = WithSavedActivations(m)
    return m


class PerceptualLoss(nn.Module):
    def __init__(self, l, rescale=False):
        super(PerceptualLoss, self).__init__()
        self.m = PerceptualNet(l)
        self.norm = ImageNetInputNorm()
        self.rescale = rescale

    def forward(self, x, y):
        if self.rescale:
            y = F.interpolate(y, size=(224, 224), mode='nearest')
            x = F.interpolate(x, size=(224, 224), mode='nearest')

        ref = self.m(self.norm(y), detach=True)
        acts = self.m(self.norm(x), detach=False)
        loss = 0
        for k in acts.keys():
            loss += torch.nn.functional.mse_loss(acts[k], ref[k])
        return loss
