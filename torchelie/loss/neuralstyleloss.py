import torch
import torch.nn.functional as F
import torch.nn as nn
import random
from typing import Dict, Optional, List, cast, Tuple

from torchelie.utils import bgram

import torchelie as tch
import torchelie.utils as tu
from torchelie.nn import ImageNetInputNorm, WithSavedActivations
from torchelie.models import PerceptualNet


def normalize(x):
    var, mean = torch.var_mean(x, dim=(2, 3), unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + 1e-4)


def normalize_(x):
    return torch.utils.checkpoint.checkpoint(normalize, x)


class NeuralStyleLoss(nn.Module):
    """
    Style Transfer loss by Leon Gatys

    https://arxiv.org/abs/1508.06576

    set the style and content before performing a forward pass.
    """
    net: PerceptualNet

    def __init__(self) -> None:
        super(NeuralStyleLoss, self).__init__()
        self.style_layers = [
            'conv1_1',
            'conv2_1',
            'conv3_1',
            'conv4_1',
            'conv5_1',
        ]
        self.content_layers = ['conv3_2']
        self.content = {}
        self.style_maps = {}
        self.net = PerceptualNet(self.style_layers + self.content_layers,
                                 remove_unused_layers=False)
        self.norm = ImageNetInputNorm()
        tu.freeze(self.net)

    def get_style_content_(self, img: torch.Tensor,
                           detach: bool) -> Dict[str, Dict[str, torch.Tensor]]:
        activations: Dict[str, torch.Tensor]

        _, activations = self.net(self.norm(img), detach=detach)

        # this ain't a bug. This normalization is freakin *everything*.
        activations = {k: normalize(a.float()) for k, a in activations.items()}

        return activations

    def set_style(self,
                  style_img: torch.Tensor,
                  style_ratio: float,
                  style_layers: Optional[List[str]] = None) -> None:
        """
        Set the style.

        Args:
            style_img (3xHxW tensor): an image tensor
            style_ratio (float): a multiplier for the style loss to make it
                greater or smaller than the content loss
            style_layer (list of str, optional): the layers on which to compute
                the style, or `None` to keep them unchanged
        """
        if style_layers is not None:
            self.style_layers = style_layers
            self.net.set_keep_layers(names=self.style_layers +
                                     self.content_layers)

        self.ratio = torch.tensor(style_ratio)

        with torch.no_grad():
            out = self.get_style_content_(style_img, detach=True)
        self.style_maps = {k: bgram(out[k]) for k in self.style_layers}

    def set_content(self,
                    content_img: torch.Tensor,
                    content_layers: Optional[List[str]] = None) -> None:
        """
        Set the content.

        Args:
            content_img (3xHxW tensor): an image tensor
            content_layer (str, optional): the layer on which to compute the
                content representation, or `None` to keep it unchanged
        """
        if content_layers is not None:
            self.content_layers = content_layers
            self.net.set_keep_layers(names=self.style_layers +
                                     self.content_layers)

        with torch.no_grad():
            out = self.get_style_content_(content_img, detach=True)

        self.content = {a: out[a] for a in self.content_layers}

    def forward(
            self,
            input_img: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Actually compute the loss
        """
        out = self.get_style_content_(input_img, detach=False)

        c_ratio = 1. - self.ratio.squeeze()
        s_ratio = self.ratio.squeeze()

        style_loss = sum(
            F.l1_loss(self.style_maps[a], bgram(out[a]))
            for a in self.style_layers) / len(self.style_maps)

        content_loss = sum(
            F.mse_loss(self.content[a], out[a])
            for a in self.content_layers) / len(self.content_layers)

        loss = c_ratio * content_loss + s_ratio * style_loss

        return loss, {
            'style': style_loss.item(),
            'content': content_loss.item()
        }
