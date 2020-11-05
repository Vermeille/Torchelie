import torch
import torch.nn as nn
import torch.nn.functional as F
import torchelie.utils as tu

from typing import Optional


class AdaIN2d(nn.Module):
    """
    Adaptive InstanceNormalization from _Arbitrary Style Transfer in Real-time
    with Adaptive Instance Normalization_ (Huang et al, 2017)

    Args:
        channels (int): number of input channels
        cond_channels (int): number of conditioning channels from which bias
            and scale will be derived
    """
    def __init__(self, channels, cond_channels):
        """
        Initialize the channel.

        Args:
            self: (todo): write your description
            channels: (list): write your description
            cond_channels: (todo): write your description
        """
        super(AdaIN2d, self).__init__()
        self.make_weight = nn.Linear(cond_channels, channels)
        self.make_bias = nn.Linear(cond_channels, channels)
        self.register_buffer('weight', torch.zeros(0))
        self.register_buffer('bias', torch.zeros(0))

    def forward(self, x, z: Optional[torch.Tensor] = None):
        """
        Forward pass

        Args:
            x (4D tensor): input tensor
            z (2D tensor, optional): conditioning vector. If not present,
            `condition(z)` must be called first

        Returns:
            x, renormalized
        """
        if z is not None:
            self.condition(z)

        m = x.mean(dim=(2, 3), keepdim=True)
        s = torch.sqrt(x.var(dim=(2, 3), keepdim=True) + 1e-8)

        weight = self.weight / (s + 1e-5)
        bias = -m * weight + self.bias
        out = weight * x + bias
        return out

    def condition(self, z):
        """
        Conditions the layer before the forward pass if z will not be present
        when calling forward

        Args:
            z (2D tensor, optional): conditioning vector
        """
        self.weight = self.make_weight(z)[:, :, None, None] + 1
        self.bias = self.make_bias(z)[:, :, None, None]


class FiLM2d(nn.Module):
    """
    Feature-wise Linear Modulation from
    https://distill.pub/2018/feature-wise-transformations/
    The difference with AdaIN is that FiLM does not uses the input's mean and
    std in its calculations

    Args:
        channels (int): number of input channels
        cond_channels (int): number of conditioning channels from which bias
            and scale will be derived
    """
    weight: Optional[torch.Tensor]
    bias: Optional[torch.Tensor]

    def __init__(self, channels, cond_channels):
        """
        Initialize the weights.

        Args:
            self: (todo): write your description
            channels: (list): write your description
            cond_channels: (todo): write your description
        """
        super(FiLM2d, self).__init__()
        self.make_weight = tu.normal_init(nn.Linear(cond_channels, channels),
                                          0.01)
        self.make_bias = tu.normal_init(nn.Linear(cond_channels, channels),
                                        0.01)
        self.weight = None
        self.bias = None

    def forward(self, x, z: Optional[torch.Tensor] = None):
        """
        Forward pass

        Args:
            x (4D tensor): input tensor
            z (2D tensor, optional): conditioning vector. If not present,
            `condition(z)` must be called first

        Returns:
            x, conditioned
        """
        if z is not None:
            self.condition(z)

        w = self.weight
        b = self.bias
        assert w is not None and b is not None
        return w * x + b

    def condition(self, z):
        """
        Conditions the layer before the forward pass if z will not be present
        when calling forward

        Args:
            z (2D tensor, optional): conditioning vector
        """
        self.weight = self.make_weight(z)[:, :, None, None].add_(1)
        self.bias = self.make_bias(z)[:, :, None, None]
