import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable, cast, Union, Tuple

from torchelie.nn.imagenetinputnorm import ImageNetInputNorm
from torchelie.models.perceptualnet import PerceptualNet


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
                 layers: Union[List[str], List[Tuple[str, float]]],
                 rescale: bool = False,
                 loss_fn: Callable[[torch.Tensor, torch.Tensor],
                                   torch.Tensor] = F.mse_loss,
                 use_avg_pool: bool = True,
                 remove_unused_layers: bool = True):
        super(PerceptualLoss, self).__init__()

        def key(l):
            if isinstance(l, (tuple, list)):
                return l[0]
            else:
                return l

        def weight(l):
            if isinstance(l, (tuple, list)):
                return l[1]
            else:
                return 1

        self.weight = {key(l): weight(l) for l in layers}
        self.m = PerceptualNet([key(l) for l in layers],
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
            y = F.interpolate(y, scale_factor=s, mode='area')
            s = 224 / min(x.shape[-2:])
            x = F.interpolate(x, scale_factor=s, mode='area')

        _, ref = self.m(self.norm(y), detach=True)
        _, acts = self.m(self.norm(x), detach=False)
        loss = cast(
            torch.Tensor,
            sum(self.weight[k] * self.loss_fn(acts[k], ref[k])
                for k in acts.keys()))
        return loss / len(acts)
