import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Conv3x3

class MovingAverageBN2dBase_(nn.Module):
    def __init__(self, channels, momentum=0.8):
        super(MovingAverageBN2dBase_, self).__init__()
        self.register_buffer('running_mean', torch.zeros(1, channels, 1, 1))
        self.register_buffer('running_var', torch.zeros(1, channels, 1, 1))
        self.register_buffer('step', torch.ones(1))
        self.momentum = momentum

    def update_moments(self, x):
        if self.training:
            m = x.mean(dim=(0, 2, 3), keepdim=True)
            v = x.var(dim=(0, 2, 3), keepdim=True)

            m = self.momentum * self.running_mean + (1 - self.momentum) * m
            self.running_mean.copy_(m.detach())
            m = m / (1 - self.momentum ** self.step)

            v = self.momentum * self.running_var + (1 - self.momentum) * v
            self.running_var.copy_(v.detach())
            v = v / (1 - self.momentum ** self.step)

            self.step += 1
        else:
            m = self.running_mean
            v = self.running_var
        return m, torch.sqrt(v)


class NoAffineMABN2d(MovingAverageBN2dBase_):
    def __init__(self, channels, momentum=0.8):
        super(NoAffineMABN2d, self).__init__()

    def forward(self, x):
        m, s = self.update_moments(x)
        return (x - m) / (s + 1e-8)


class MovingAverageBN2d(MovingAverageBN2dBase_):
    def __init__(self, channels, momentum=0.8):
        super(MovingAverageBN2d, self).__init__(channels, momentum)
        self.weight = nn.Parameter(torch.zeros(channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(channels, 1, 1))

    def forward(self, x):
        m, v = self.update_moments(x)
        weight = (1 + self.weight) / v
        bias = -m * weight + self.bias
        return weight * x + bias


class ConditionalMABN2d(MovingAverageBN2dBase_):
    def __init__(self, channels, cond_channels, momentum=0.6):
        super(ConditionalMABN2d, self).__init__(channels, momentum)
        self.make_weight = nn.Linear(cond_channels, channels)
        self.make_bias = nn.Linear(cond_channels, channels)

    def forward(self, x, z=None):
        if z is not None:
            self.condition(z)

        m, v = self.update_moments(x)
        weight = (1 + self.weight) / (v + 1e-8)
        bias = -m * weight + self.bias
        return weight * x + bias

    def condition(self, z):
        self.weight = self.make_weight(z)[:, :, None, None]
        self.bias = self.make_bias(z)[:, :, None, None]


class MovingAverageSpade2d(MovingAverageBN2dBase_):
    def __init__(self, channels, cond_channels, hidden, momentum=0.8):
        super(MovingAverageSpade2d, self).__init__(channels, momentum)
        self.initial = Conv3x3(cond_channels, hidden)
        self.make_weight = Conv3x3(hidden, channels)
        self.make_bias = Conv3x3(hidden, channels)

    def forward(self, x, z=None):
        if z is not None:
            self.condition(z, x)

        m, v = self.update_moments(x)
        weight = (1 + self.weight) / (v + 1e-8)
        bias = -m * weight + self.bias
        return weight * x + bias

    def condition(self, z, like):
        z = F.interpolate(z, size=like.shape[2:], mode='nearest')
        z = F.relu(self.initial(z), inplace=True)
        self.weight = self.make_weight(z)
        self.bias = self.make_bias(z)

