import torch
import torch.nn.functional as F
import torch.nn as nn

from torchelie.utils import bgram
import torchelie.nn as tnn

from torchelie.nn import ImageNetInputNorm
from torchelie.models import PerceptualNet


class DeepDreamLoss(nn.Module):
    """
    The Deep Dream loss

    Args:
        model (nn.Module): a pretrained network on which to compute the
            activations
        dream_layer (str): the name of the layer on which the activations are
            to be maximized
        max_reduction (int): the maximum factor of reduction of the image, for
            multiscale generation
    """
    def __init__(self, model, dream_layer, max_reduction=3):
        super(DeepDreamLoss, self).__init__()
        self.dream_layer = dream_layer
        self.octaves = max_reduction
        model = model.eval()
        self.net = tnn.WithSavedActivations(model, names=[self.dream_layer])
        self.i = 0

    def get_acts_(self, img, detach):
        octave = (self.i % (self.octaves * 2)) / 2 + 1
        this_sz_img = F.interpolate(img, scale_factor=1 / octave)
        _, activations = self.net(this_sz_img, detach=detach)
        return activations[self.dream_layer]

    def forward(self, input_img):
        """
        Compute the Deep Dream loss on `input_img`
        """
        dream = self.get_acts_(input_img, detach=False)
        self.i += 1

        dream_loss = -dream.pow(2).sum()

        return dream_loss
