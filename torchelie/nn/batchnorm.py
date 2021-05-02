import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv3x3
from torchelie.utils import kaiming, xavier, experimental

__all__ = []


class BatchNorm2dBase_(nn.Module):
    @experimental
    def __init__(self, channels, momentum=0.8):
        super(BatchNorm2dBase_, self).__init__()
        self.register_buffer('running_mean', torch.zeros(1, channels, 1, 1))
        self.register_buffer('running_var', torch.ones(1, channels, 1, 1))
        self.register_buffer('step', torch.ones(1))
        self.momentum = momentum

    def update_moments(self, x):
        if self.training:
            m = x.mean(dim=(0, 2, 3), keepdim=True)
            v = torch.sqrt(x.var(dim=(0, 2, 3), unbiased=False, keepdim=True) + 1e-8)

            self.running_mean.copy_((self.momentum * self.running_mean
                                     + (1 - self.momentum) * m))

            self.running_var.copy_((self.momentum * self.running_var
                                    + (1 - self.momentum) * v))

            self.step += 1
        else:
            m = self.running_mean
            v = self.running_var
        return m, v


__all__.append('BatchNorm2dBase_')


class MovingAverageBN2dBase_(nn.Module):
    @experimental
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
            m = m / (1 - self.momentum**self.step)

            v = self.momentum * self.running_var + (1 - self.momentum) * v
            self.running_var.copy_(v.detach())
            v = v / (1 - self.momentum**self.step)

            self.step += 1
        else:
            m = self.running_mean
            v = self.running_var
        return m, torch.sqrt(v)


__all__.append('MovingAverageBN2dBase_')


def make_no_affine(base, name):
    class NoAffineBN(base):
        @experimental
        def __init__(self, channels, momentum=0.8):
            super(NoAffineBN, self).__init__(channels, momentum)

        def forward(self, x):
            m, s = self.update_moments(x)
            return (x - m) / (s + 1e-8)

    NoAffineBN.__name__ = name
    return NoAffineBN


NoAffineBN2d = make_no_affine(BatchNorm2dBase_, 'NoAffineBN2d')
NoAffineMABN2d = make_no_affine(MovingAverageBN2dBase_, 'NoAffineMABN2d')
__all__.append('NoAffineBN2d')
__all__.append('NoAffineMABN2d')


def make_bn(base, name):
    class BatchNorm2d(base):
        @experimental
        def __init__(self, channels, momentum=0.8):
            super(BatchNorm2d, self).__init__(channels, momentum)
            self.weight = nn.Parameter(torch.zeros(channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(channels, 1, 1))

        def forward(self, x):
            m, s = self.update_moments(x)
            weight = (1 + self.weight) / (s + 1e-8)
            bias = -m * weight + self.bias
            return weight * x + bias

    BatchNorm2d.__name__ = name
    return BatchNorm2d


BatchNorm2d = make_bn(BatchNorm2dBase_, 'BatchNorm2d')
MovingAverageBN2d = make_bn(MovingAverageBN2dBase_, 'MovingAverageBN2d')
__all__.append('BatchNorm2d')
__all__.append('MovingAverageBN2d')


def make_cbn(base, name):
    class ConditionalBN2d(base):
        @experimental
        def __init__(self, channels, cond_channels, momentum=0.8):
            super(ConditionalBN2d, self).__init__(channels, momentum)
            self.make_weight = nn.Linear(cond_channels, channels)
            nn.init.normal_(self.make_weight.weight, 0.002)
            nn.init.zeros_(self.make_weight.bias)
            self.make_bias = nn.Linear(cond_channels, channels)
            nn.init.normal_(self.make_bias.weight, 0.002)
            nn.init.zeros_(self.make_bias.bias)
            self.bn = nn.BatchNorm2d(channels, affine=False)

            self.weight = torch.tensor([0])
            self.bias = torch.tensor([0])

        def forward(self, x, z=None):
            if z is not None:
                self.condition(z)
            return self.bn(x) * self.weight + self.bias

            m, v = self.update_moments(x)
            weight = self.weight / (v + 1e-6)
            bias = -m * weight + self.bias
            out = x * weight + bias
            print(out.mean().item(), out.std().item())
            return out

        def condition(self, z):
            self.weight = (self.make_weight(z)[:, :, None, None])+1
            self.bias = (self.make_bias(z)[:, :, None, None])

    ConditionalBN2d.__name__ = name
    return ConditionalBN2d


ConditionalBN2d = make_cbn(BatchNorm2dBase_, 'ConditionalBN2d')
ConditionalMABN2d = make_cbn(MovingAverageBN2dBase_, 'ConditionalMABN2d')
__all__.append('ConditionalBN2d')
__all__.append('ConditionalMABN2d')


def make_spade(base, name):
    class Spade2d(base):
        @experimental
        def __init__(self,
                     channels,
                     cond_channels,
                     hidden,
                     momentum=0.8):
            super(Spade2d, self).__init__(channels, momentum)
            self.initial = kaiming(Conv3x3(cond_channels, hidden, stride=2))
            self.make_weight = xavier(Conv3x3(hidden, channels))
            self.make_bias = xavier(Conv3x3(hidden, channels))
            self.register_buffer('weight', torch.ones(channels))

        def forward(self, x, z=None):
            if z is not None:
                self.z = z
            self.make_weight_bias(self.z, x.shape[2:])

            m, v = self.update_moments(x)
            weight = (1 + self.weight) / (v + 1e-8)
            bias = -m * weight + self.bias
            return weight * x + bias

        def condition(self, z):
            self.z = z

        def make_weight_bias(self, z, size):
            z = F.interpolate(z,
                              size=(size[0] * 2, size[1] * 2),
                              mode='nearest')
            z = F.relu(self.initial(z), inplace=True)
            self.weight = self.make_weight(z)
            self.bias = self.make_bias(z)

    Spade2d.__name__ = name
    return Spade2d


Spade2d = make_spade(BatchNorm2dBase_, 'Spade2d')
SpadeMA2d = make_spade(MovingAverageBN2dBase_, 'SpadeMA2d')
__all__.append('Spade2d')
__all__.append('SpadeMA2d')


class AttenNorm2d(nn.BatchNorm2d):
    """
    From https://arxiv.org/abs/1908.01259
    """
    def __init__(self,
                 num_features,
                 num_weights,
                 eps=1e-8,
                 momentum=0.8,
                 track_running_stats=True):
        super(AttenNorm2d, self).__init__(num_features,
                                          eps=eps,
                                          momentum=momentum,
                                          affine=False,
                                          track_running_stats=track_running_stats)
        self.gamma = nn.Parameter(torch.ones(num_weights, num_features))
        self.beta = nn.Parameter(torch.zeros(num_weights, num_features))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_features, num_weights)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = super(AttenNorm2d, self).forward(x)
        size = output.size()
        b, c, _, _ = x.size()

        y = self.avgpool(x).view(b, c)
        y = self.fc(y)
        y = self.sigmoid(y)

        gamma = torch.mm(y, self.gamma)
        beta = torch.mm(y, self.beta)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1).expand(size)
        beta = beta.unsqueeze(-1).unsqueeze(-1).expand(size)
        return gamma * output + beta


__all__.append('AttenNorm2d')
