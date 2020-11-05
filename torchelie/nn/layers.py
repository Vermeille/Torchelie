import torch
import torch.nn as nn
import torch.nn.functional as F
import torchelie.utils as tu


def Conv2d(in_ch, out_ch, ks, stride=1, bias=True):
    """
    A Conv2d with 'same' padding
    """
    return nn.Conv2d(in_ch, out_ch, ks, padding=ks // 2, stride=stride,
            bias=bias)


def Conv3x3(in_ch, out_ch, stride=1, bias=True):
    """
    A 3x3 Conv2d with 'same' padding
    """
    return Conv2d(in_ch, out_ch, 3, stride=stride, bias=bias)


def Conv1x1(in_ch, out_ch, stride=1, bias=True):
    """
    A 1x1 Conv2d
    """
    return Conv2d(in_ch, out_ch, 1, stride=stride, bias=bias)


class AdaptiveConcatPool2d(nn.Module):
    """
    Pools with AdaptiveMaxPool2d AND AdaptiveAvgPool2d and concatenates both
    results.

    Args:
        target_size: the target output size (single integer or
            double-integer tuple)
    """
    def __init__(self, target_size):
        """
        Initialize target_size.

        Args:
            self: (todo): write your description
            target_size: (int): write your description
        """
        super(AdaptiveConcatPool2d, self).__init__()
        self.target_size = target_size

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return torch.cat([
            nn.functional.adaptive_avg_pool2d(x, self.target_size),
            nn.functional.adaptive_max_pool2d(x, self.target_size),
        ], dim=1)


class ModulatedConv(nn.Conv2d):
    def __init__(self, in_channels, noise_channels, *args, **kwargs):
        """
        Initialize the channels.

        Args:
            self: (todo): write your description
            in_channels: (int): write your description
            noise_channels: (todo): write your description
        """
        super(ModulatedConv, self).__init__(in_channels, *args, **kwargs)
        self.make_s = tu.kaiming(nn.Linear(noise_channels, in_channels))
        self.make_s.bias.data.fill_(1)

    def condition(self, z):
        """
        Set the condition

        Args:
            self: (todo): write your description
            z: (todo): write your description
        """
        self.s = self.make_s(z)

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        N, C, H, W = x.shape
        C_out, C_in = self.weight.shape[:2]
        w_prime = torch.einsum('oihw,bi->boihw', self.weight, self.s)
        w_prime_prime = torch.einsum('boihw,boihw->bo', w_prime, w_prime)
        w_prime_prime = w_prime_prime.add_(1e-8).rsqrt()
        w = w_prime * w_prime_prime[..., None, None, None]

        w = w.view(-1, *w.shape[2:])
        x = F.conv2d(x.view(1, -1, H, W), w, None, self.stride, self.padding,
                       self.dilation, N)
        x = x.view(N, C_out, H, W)
        return x.add_(self.bias.view(-1, 1, 1))

