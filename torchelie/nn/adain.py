import torch
import torch.nn as nn
import torch.nn.functional as F


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
        super(AdaIN, self).__init__()
        self.make_weight = nn.Linear(cond_channels, channels)
        self.make_bias = nn.Linear(cond_channels, channels)

    def forward(self, x, z=None):
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
        s = x.std(dim=(2, 3), keepdim=True)

        weight = (1 + self.weight) / (s + 1e-8)
        bias = -m * weight + self.bias
        return weight * x + bias

    def condition(self, z):
        """
        Conditions the layer before the forward pass if z will not be present
        when calling forward

        Args:
            z (2D tensor, optional): conditioning vector
        """
        self.weight = self.make_weight(z)[:, :, None, None]
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
    def __init__(self, channels, cond_channels):
        super(AdaIN, self).__init__()
        self.make_weight = nn.Linear(cond_channels, channels)
        self.make_bias = nn.Linear(cond_channels, channels)

    def forward(self, x, z=None):
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

        return self.weight * x + self.bias

    def condition(self, z):
        """
        Conditions the layer before the forward pass if z will not be present
        when calling forward

        Args:
            z (2D tensor, optional): conditioning vector
        """
        self.weight = self.make_weight(z)[:, :, None, None]
        self.bias = self.make_bias(z)[:, :, None, None]
