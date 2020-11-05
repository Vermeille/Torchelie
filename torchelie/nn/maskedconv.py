import torch
import torch.nn as nn
import torch.nn.functional as F

from torchelie.utils import kaiming


class MaskedConv2d(nn.Conv2d):
    """
    A masked 2D convolution for PixelCNN

    Args:
        in_chan (int): number of input channels
        out_chan (int): number of output channels
        ks (int): kernel size
        center (bool): whereas central pixel is masked or not
        stride (int): stride, defaults to 1
        bias (2-tuple of ints): A spatial bias. Either the spatial dimensions
            of the input for a different bias at each location, or (1, 1) for
            the same bias everywhere (default)
    """

    def __init__(self, in_chan, out_chan, ks, center, stride=1, bias=(1, 1)):
        """
        Initialize the channel.

        Args:
            self: (todo): write your description
            in_chan: (int): write your description
            out_chan: (str): write your description
            ks: (int): write your description
            center: (list): write your description
            stride: (int): write your description
            bias: (float): write your description
        """
        super(MaskedConv2d, self).__init__(in_chan,
                                           out_chan, (ks // 2 + 1, ks),
                                           padding=0,
                                           stride=stride,
                                           bias=False)
        self.register_buffer('mask', torch.ones(ks // 2 + 1, ks))
        self.mask[-1, ks // 2 + (1 if center else 0):] = 0

        self.spatial_bias = None
        if bias is not None:
            self.spatial_bias = nn.Parameter(torch.zeros(out_chan, *bias))

        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        self.weight_orig = self.weight
        del self.weight
        self.weight = self.weight_orig * self.mask
        ks = self.weight.shape[-1]

        x = F.pad(x, (ks // 2, ks // 2, ks // 2, 0))
        res = super(MaskedConv2d, self).forward(x)

        self.weight = self.weight_orig
        del self.weight_orig
        if self.spatial_bias is not None:
            return res + self.spatial_bias
        else:
            return res


class TopLeftConv2d(nn.Module):
    """
    A 2D convolution for PixelCNN made of a convolution above the current pixel
    and another on the left.

    Args:
        in_chan (int): number of input channels
        out_chan (int): number of output channels
        ks (int): kernel size
        center (bool): whereas central pixel is masked or not
        stride (int): stride, defaults to 1
        bias (2-tuple of ints): A spatial bias. Either the spatial dimensions
            of the input for a different bias at each location, or (1, 1) for
            the same bias everywhere (default)
    """

    def __init__(self, in_chan, out_chan, ks, center, stride=1, bias=(1, 1)):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            in_chan: (int): write your description
            out_chan: (str): write your description
            ks: (int): write your description
            center: (list): write your description
            stride: (int): write your description
            bias: (float): write your description
        """
        super(TopLeftConv2d, self).__init__()
        self.top = kaiming(
            nn.Conv2d(in_chan,
                      out_chan, (ks // 2, ks),
                      bias=False,
                      stride=stride))
        self.left = kaiming(
            nn.Conv2d(in_chan,
                      out_chan, (1, ks // 2 + (1 if center else 0)),
                      stride=stride,
                      bias=False))
        self.ks = ks
        self.center = center
        self.bias = nn.Parameter(torch.zeros(out_chan, *bias))

    def forward(self, x):
        """
        Calculate of the graph.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        top = self.top(
            F.pad(x[:, :, :-1, :],
                  (self.ks // 2, self.ks // 2, self.ks // 2, 0)))
        if not self.center:
            left = self.left(F.pad(x[:, :, :, :-1], (self.ks // 2, 0, 0, 0)))
        else:
            left = self.left(F.pad(x, (self.ks // 2, 0, 0, 0)))
        return top + left + self.bias
