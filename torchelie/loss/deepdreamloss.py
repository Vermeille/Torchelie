import torch
import torch.nn.functional as F
import torch.nn as nn

from torchelie.utils import bgram

from torchelie.nn import ImageNetInputNorm
from torchelie.models import PerceptualNet


class DeepDreamLoss(nn.Module):
    def __init__(self, content_layer, dream_layer, ratio):
        super(DeepDreamLoss, self).__init__()
        self.content_layer = content_layer
        self.dream_layer = dream_layer
        self.net = PerceptualNet([self.content_layer, self.dream_layer])
        self.norm = ImageNetInputNorm()
        self.ratio = ratio

    def get_acts_(self, img, detach):
        contents = []
        dreams = []
        for octave in [1, 2, 4]:
            this_sz_img = F.interpolate(img, scale_factor=1 / octave)
            _, activations = self.net(self.norm(this_sz_img), detach=detach)
            contents.append(activations[self.content_layer])
            dreams.append(activations[self.dream_layer])
        return contents, dreams

    def set_content(self, content_img, content_layer=None):
        if content_layer is not None:
            self.content_layer = content_layer
            self.net.set_keep_layers(
                names=[self.style_layers, self.content_layer])

        with torch.no_grad():
            acts = self.get_acts_(content_img[None], detach=True)[0]
        self.photo_activations = acts

    def forward(self, input_img):
        contents, dreams = self.get_acts_(input_img, detach=False)

        dream_loss = -sum(dream.pow(2).mean() for dream in dreams)

        content_loss = sum(
            F.mse_loss(content, ref)
            for content, ref in zip(contents, self.photo_activations))

        return content_loss + self.ratio * dream_loss
