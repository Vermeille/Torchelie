import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaIN2d(nn.Module):
    def __init__(self, channels, cond_channels):
        super(AdaIN, self).__init__()
        self.make_weight = nn.Linear(cond_channels, channels)
        self.make_bias = nn.Linear(cond_channels, channels)

    def forward(self, x, z=None):
        if z is not None:
            self.condition(z)

        m = x.mean(dim=(2, 3), keepdim=True)
        s = x.std(dim=(2, 3), keepdim=True)

        weight = (1 + self.weight) / (s + 1e-8)
        bias = -m * weight + self.bias
        return weight * x + bias

    def condition(self, z):
        self.weight = self.make_weight(z)[:, :, None, None]
        self.bias = self.make_bias(z)[:, :, None, None]


class FiLM2d(nn.Module):
    def __init__(self, channels, cond_channels):
        super(AdaIN, self).__init__()
        self.make_weight = nn.Linear(cond_channels, channels)
        self.make_bias = nn.Linear(cond_channels, channels)

    def forward(self, x, z=None):
        if z is not None:
            self.condition(z)

        return self.weight * x + self.bias

    def condition(self, z):
        self.weight = self.make_weight(z)[:, :, None, None]
        self.bias = self.make_bias(z)[:, :, None, None]


