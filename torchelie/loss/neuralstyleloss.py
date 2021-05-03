import torch
import torch.nn.functional as F
import torch.nn as nn
import random
from typing import Dict, Optional, List, cast, Tuple

from torchelie.utils import bgram

import torchelie.utils as tu
from torchelie.nn import ImageNetInputNorm
from torchelie.models import PerceptualNet


@tu.experimental
def hist_match(source, template):
    # get the set of unique pixel values and their corresponding indices and
    # counts
    sm, sM = source.min(), source.max()
    tm, tM = template.min(), template.max()
    RES = 256
    ssz = (sM - sm) / (RES - 1)
    tsz = (tM - tm) / (RES - 1)
    # s_counts, s_values = np.histogram(source, bins=256, range=(m, M))
    # t_counts, t_values = np.histogram(template, bins=256, range=(m, M))
    s_counts = torch.histc(source, bins=RES, min=sm, max=sM)
    t_counts = torch.histc(template, bins=RES, min=tm, max=tM)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    # s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles = torch.cumsum(s_counts, dim=0).float()
    s_quantiles /= s_quantiles[-1].clone()
    # t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles = torch.cumsum(t_counts, dim=0).float()
    t_quantiles /= t_quantiles[-1].clone()

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    # interp_t_values = np.interp(s_quantiles, t_quantiles, t_values[1:])
    interp_t_values = (t_quantiles[:, None]
                       < s_quantiles[None, :]).sum(0).float().mul_(tsz).add_(tm)

    bin = (source - sm) / (ssz + 1e-8)
    bin_idx = bin.long()
    low = interp_t_values[bin_idx]
    diff = interp_t_values[torch.clamp(bin_idx + 1, max=RES - 1)] - low
    return low + (bin - bin_idx.float()) * diff


@tu.experimental
def hist_loss(src, tgt):
    N, C, H, W = src.shape

    with torch.no_grad():
        remapped = torch.zeros_like(src)
        for s, t, r in zip(src.view(N * C, -1), tgt.view(N * C, -1),
                           remapped.view(N * C, -1)):
            r[:] = hist_match(s, t)
    return F.mse_loss(src, remapped) * 1


class NeuralStyleLoss(nn.Module):
    """
    Style Transfer loss by Leon Gatys

    https://arxiv.org/abs/1508.06576

    set the style and content before performing a forward pass.
    """
    style_hists: Dict[str, torch.Tensor]
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
        self.style_weights = {'conv5_1': 1.0, 'conv4_1': 1.0}
        self.style_hists = {}
        self.content_layers = ['conv3_2']
        self.hists_layers = ['conv5_1']
        self.net = PerceptualNet(self.style_layers + self.content_layers,
                                 remove_unused_layers=False)
        tu.freeze(self.net)
        self.norm = ImageNetInputNorm()

    def get_style_content_(self, img: torch.Tensor,
                           detach: bool) -> Dict[str, Dict[str, torch.Tensor]]:
        activations: Dict[str, torch.Tensor]
        _, activations = self.net(self.norm(img), detach=detach)

        grams = {
            layer_id: bgram(layer_data)
            for layer_id, layer_data in activations.items()
            if layer_id in self.style_layers
        }
        act_names = list(activations.keys())
        for i in range(len(act_names)):
            for j in range(i, len(act_names)):
                comb_name = act_names[i] + ':' + act_names[j]
                if comb_name not in self.style_layers:
                    continue
                small = activations[act_names[i]]
                big = activations[act_names[j]]

                if small.shape[-1] > big.shape[-1]:
                    small, big = big, small

                small = F.interpolate(small,
                                      size=big.shape[-2:],
                                      mode='nearest')
                comb = torch.cat([big, small], dim=1)
                grams[comb_name] = bgram(comb)

        content = {
            layer: (a - a.mean((2, 3), keepdim=True))
            / torch.sqrt(a.std((2, 3), keepdim=True) + 1e-8)
            for layer, a in activations.items() if layer in self.content_layers
        }

        hists = {
            layer: a
            for layer, a in activations.items() if layer in self.hists_layers
        }
        return {'grams': grams, 'content': content, 'hists': hists}

    def set_style(self,
                  style_img: torch.Tensor,
                  style_ratio: float,
                  style_layers: Optional[List[str]] = None,
                  style_weights: Optional[Dict[str, float]] = None) -> None:
        """
        Set the style.

        Args:
            style_img (3xHxW tensor): an image tensor
            style_ratio (float): a multiplier for the style loss to make it
                greater or smaller than the content loss
            style_layer (list of str, optional): the layers on which to compute
                the style, or `None` to keep them unchanged
        """
        self.ratio = torch.tensor(style_ratio)

        if style_layers is not None:
            self.style_layers = style_layers
            self.net.set_keep_layers(names=self.style_layers
                                     + self.content_layers)
        if style_weights is not None:
            self.style_weights = style_weights

        with torch.no_grad():
            out = self.get_style_content_(style_img, detach=True)

        self.style_grams = out['grams']
        self.style_hists = out['hists']

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
            self.net.set_keep_layers(names=self.style_layers
                                     + self.content_layers)

        with torch.no_grad():
            acts = self.get_style_content_(content_img, detach=True)['content']
        self.photo_activations = acts

    def forward(
            self,
            input_img: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Actually compute the loss
        """
        out = self.get_style_content_(input_img, detach=False)
        style_grams, content_acts, hists = out['grams'], out['content'], out[
            'hists']

        c_ratio = 1. - self.ratio.squeeze()
        s_ratio = self.ratio.squeeze()
        losses = {}

        style_loss = cast(torch.Tensor, 0.)
        avg = 1 / len(style_grams)
        for j in style_grams:
            this_loss = F.l1_loss(style_grams[j],
                                  self.style_grams[j],
                                  reduction='none').sum((1, 2))

            w = self.style_weights.get(j, 1)
            this_loss = avg * w * this_loss
            losses['style:' + j] = (s_ratio * this_loss).mean().item()
            style_loss = style_loss + this_loss
        losses['style_loss'] = (s_ratio * style_loss).mean().item()

        content_loss = cast(torch.Tensor, 0.)
        avg = 1 / len(content_acts)
        for j in content_acts:
            this_loss = F.mse_loss(content_acts[j],
                                   self.photo_activations[j],
                                   reduction='none').mean((1, 2, 3))
            content_loss = content_loss + avg * this_loss
            losses['content:' + j] = (c_ratio * this_loss).mean().item()
        losses['content_loss'] = (c_ratio * content_loss).mean().item()

        hists_loss = cast(torch.Tensor, 0.)
        losses['hists_loss'] = 0
        if random.randint(0, 20) > 18:
            for layer in hists.keys():
                hists_loss = hists_loss + hist_loss(hists[layer],
                                                    self.style_hists[layer])
            losses['hists_loss'] = hists_loss.mean().item()
        loss = (c_ratio * content_loss + s_ratio
                * (style_loss + hists_loss)).mean()
        losses['loss'] = loss.item()

        return loss, losses
