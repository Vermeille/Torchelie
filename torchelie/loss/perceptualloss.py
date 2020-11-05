import torch
import torch.nn as nn
import torch.nn.functional as F

from torchelie.nn import ImageNetInputNorm
from torchelie.models import PerceptualNet


class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss: the distance between a two images deep representation

    :math:`\text{Percept}(\text{input}, \text{target})=\sum_l^{layers}
    \text{loss_fn}(\text{Vgg}(\text{input})_l, \text{Vgg}(\text{target})_l)`

    Args:
        l (list of str): the layers on which to compare the representations
        rescale (bool): whether to scale images to 224x224 as expected by the
            underlying vgg net
        loss_fn (distance function): a distance function to compare the
            representations, like mse_loss or l1_loss
    """
    def __init__(self, l, rescale=False, loss_fn=F.mse_loss):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            l: (int): write your description
            rescale: (str): write your description
            loss_fn: (todo): write your description
            F: (int): write your description
            mse_loss: (bool): write your description
        """
        super(PerceptualLoss, self).__init__()
        self.m = PerceptualNet(l)
        self.norm = ImageNetInputNorm()
        self.rescale = rescale
        self.loss_fn = loss_fn

    def forward(self, x, y):
        """
        Return the perceptual loss between batch of images `x` and `y`
        """
        if self.rescale:
            y = F.interpolate(y, size=(224, 224), mode='nearest')
            x = F.interpolate(x, size=(224, 224), mode='nearest')

        _, ref = self.m(self.norm(y), detach=True)
        _, acts = self.m(self.norm(x), detach=False)
        loss = 0
        for k in acts.keys():
            loss += self.loss_fn(acts[k], ref[k])
        return loss
