import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchNorm2dBase_(nn.Module):
    def __init__(self, channels, momentum=0.8):
        super(BatchNorm2dBase_, self).__init__()
        self.register_buffer('running_mean', torch.zeros(1, channels, 1, 1))
        self.register_buffer('running_var', torch.ones(1, channels, 1, 1))
        self.register_buffer('step', torch.ones(1))
        self.momentum = momentum

    def update_moments(self, x):
        if self.training:
            m = x.mean(dim=(0, 2, 3), keepdim=True)
            v = x.var(dim=(0, 2, 3), keepdim=True)

            self.running_mean.copy_((
                    self.momentum * self.running_mean
                    + (1 - self.momentum) * m).detach())

            self.running_var.copy_((
                    self.momentum * self.running_var
                    + (1 - self.momentum) * v).detach())

            self.step += 1
        else:
            m = self.running_mean
            v = self.running_var
        return m, torch.sqrt(v)


class NoAffineBN2d(BatchNorm2dBase_):
    def __init__(self, channels, momentum=0.8):
        super(NoAffineBN2d, self).__init__(channels, momentum)

    def forward(self, x):
        m, s = self.update_moments(x)
        return (x - m) / (s + 1e-8)


class BatchNorm2d(BatchNorm2dBase_):
    def __init__(self, channels, momentum=0.8):
        super(BatchNorm2d, self).__init__(channels, momentum)
        self.weight = nn.Parameter(torch.zeros(channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(channels, 1, 1))

    def forward(self, x):
        m, s = self.update_moments(x)
        weight = (1 + self.weight) / (s + 1e-8)
        bias = -m * weight + self.bias
        return weight * x + bias


class ConditionalBN2d(BatchNorm2dBase_):
    def __init__(self, channels, cond_channels, momentum=0.8):
        super(ConditionalBN2d, self).__init__(channels, momentum)
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





