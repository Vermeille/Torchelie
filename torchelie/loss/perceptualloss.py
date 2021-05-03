import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable, cast

from torchelie.nn import ImageNetInputNorm
from torchelie.models import PerceptualNet


class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss: the distance between a two images deep representation

    :math:`\text{Percept}(\text{input}, \text{target})=\sum_l^{layers}
    \text{loss_fn}(\text{Vgg}(\text{input})_l, \text{Vgg}(\text{target})_l)`

    Args:
        l (list of str): the layers on which to compare the representations
        rescale (bool): whether to scale images smaller side to 224 as
            expected by the underlying vgg net
        loss_fn (distance function): a distance function to compare the
            representations, like mse_loss or l1_loss
    """
    def __init__(self,
                 layers: List[str],
                 rescale: bool = False,
                 loss_fn: Callable[[torch.Tensor, torch.Tensor],
                                   torch.Tensor] = F.mse_loss,
                 use_avg_pool: bool = True,
                 remove_unused_layers: bool = True):
        super(PerceptualLoss, self).__init__()
        self.m = PerceptualNet(layers,
                               use_avg_pool=use_avg_pool,
                               remove_unused_layers=remove_unused_layers)
        self.norm = ImageNetInputNorm()
        self.rescale = rescale
        self.loss_fn = loss_fn

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Return the perceptual loss between batch of images `x` and `y`
        """
        if self.rescale:
            s = 224 / min(y.shape[-2:])
            y = F.interpolate(y, scale_factor=s, mode='bicubic')
            s = 224 / min(x.shape[-2:])
            x = F.interpolate(x, scale_factor=s, mode='bicubic')

        _, ref = self.m(self.norm(y), detach=True)
        _, acts = self.m(self.norm(x), detach=False)
        loss = cast(torch.Tensor,
                    sum(self.loss_fn(acts[k], ref[k]) for k in acts.keys()))
        return loss / len(acts)
