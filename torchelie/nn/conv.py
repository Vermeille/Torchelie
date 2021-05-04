import torch.nn as nn
from typing import Tuple, Optional, Union, cast
from .utils import remove_batchnorm, insert_after, insert_before
from .condseq import CondSeq
from torchelie.utils import kaiming, experimental
from .interpolate import InterpolateBilinear2d


def Conv2d(in_ch, out_ch, ks, stride=1, bias=True) -> nn.Conv2d:
    """
    A Conv2d with 'same' padding
    """
    return nn.Conv2d(in_ch,
                     out_ch,
                     ks,
                     padding=(ks - 1) // 2,
                     stride=stride,
                     bias=bias)


def Conv3x3(in_ch: int,
            out_ch: int,
            stride: int = 1,
            bias: bool = True) -> nn.Conv2d:
    """
    A 3x3 Conv2d with 'same' padding
    """
    return Conv2d(in_ch, out_ch, 3, stride=stride, bias=bias)


def Conv1x1(in_ch: int,
            out_ch: int,
            stride: int = 1,
            bias: bool = True) -> nn.Conv2d:
    """
    A 1x1 Conv2d
    """
    return Conv2d(in_ch, out_ch, 1, stride=stride, bias=bias)


class ConvBlock(CondSeq):
    """
    A packed block with Conv-BatchNorm-ReLU and various operations to alter it.

    Args:
        in_channels (int): input channels
        out_channels (int): output channels
        kernel_size (int): kernel size
        stride (int): stride of the conv

    Returns:
        A packed block with Conv-Norm-ReLU as a CondSeq
    """
    conv: Union[nn.Conv2d, nn.ConvTranspose2d]
    norm: Optional[nn.Module]
    relu: Optional[nn.Module]

    out_channels: int
    in_channels: int
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.reset()

    def reset(self) -> None:
        """
        Recreate the block as a simple conv-BatchNorm-ReLU
        """
        self.add_module(
            'conv',
            kaiming(
                Conv2d(self.in_channels,
                       self.out_channels,
                       self.kernel_size[0],
                       stride=self.stride,
                       bias=False)))
        self.add_module('norm', nn.BatchNorm2d(self.out_channels))
        self.add_module('relu', nn.ReLU(inplace=True))

    def to_input_specs(self, in_channels: int) -> 'ConvBlock':
        """
        Recreate a convolution with :code:`in_channels` input channels
        """
        self.in_channels = in_channels

        c = self.conv
        self.conv = kaiming(
            Conv2d(in_channels,
                   c.out_channels,
                   c.kernel_size[0],
                   stride=c.stride,
                   bias=c.bias))
        return self

    @experimental
    def to_transposed_conv(self) -> 'ConvBlock':
        """
        Transform the convolution into a hopefully equivalent transposed
        convolution
        """
        c = self.conv
        assert c.kernel_size[0] in [
            3, 4
        ], ('ConvBlock.to_transposed_conv()'
            ' not supported with kernel_size other than 2 or 3 (please add the'
            ' support!')
        if c.kernel_size[0] == 4:
            self.conv = kaiming(
                nn.ConvTranspose2d(c.in_channels,
                                   c.out_channels,
                                   cast(Tuple[int, int], c.kernel_size),
                                   stride=cast(Tuple[int, int], c.stride),
                                   padding=cast(Tuple[int, int], c.padding),
                                   bias=c.bias is not None,
                                   padding_mode=c.padding_mode))
        else:
            self.conv = kaiming(
                nn.ConvTranspose2d(c.in_channels,
                                   c.out_channels,
                                   cast(Tuple[int, int], c.kernel_size),
                                   stride=cast(Tuple[int, int], c.stride),
                                   padding=cast(Tuple[int, int], c.padding),
                                   output_padding=cast(Tuple[int, int],
                                                       c.padding),
                                   bias=c.bias is not None,
                                   padding_mode=c.padding_mode))
        return self

    def remove_batchnorm(self) -> 'ConvBlock':
        """
        Remove the BatchNorm, restores the bias term in conv.

        Returns:
            self
        """
        remove_batchnorm(self)
        if hasattr(self, 'norm'):
            del self.norm
        self.norm = None
        return self

    def add_upsampling(self) -> 'ConvBlock':
        """
        Add a bilinear upsampling layer before the conv that doubles the
        spatial size
        """
        insert_before(self, 'conv', InterpolateBilinear2d(scale_factor=2),
                      'upsample')
        return self

    def restore_batchnorm(self) -> 'ConvBlock':
        """
        Restore BatchNorm if deleted
        """
        if self.norm is not None:
            return self
        insert_after(self, 'conv', nn.BatchNorm2d(self.out_channels), 'norm')
        return self

    def no_bias(self) -> 'ConvBlock':
        """
        Remove the bias term.

        Returns:
            self
        """
        if hasattr(self.norm, 'bias') and self.norm is not None:
            self.norm.bias = None  # type: ignore
        self.conv.bias = None
        return self

    def leaky(self, leak: float = 0.2) -> 'ConvBlock':
        """
        Change the ReLU to a LeakyReLU, also rescaling the weights in the conv
        to preserve the variance.

        Returns:
            self
        """
        if not hasattr(self, 'relu') or self.relu is None:
            return self

        new_gain = nn.init.calculate_gain('leaky_relu', param=leak)
        if isinstance(self.relu, nn.LeakyReLU):
            old_gain = nn.init.calculate_gain('leaky_relu',
                                              param=self.relu.negative_slope)
        else:
            old_gain = nn.init.calculate_gain('relu')

        self.relu = nn.LeakyReLU(leak, inplace=True)
        self.conv.weight.data *= new_gain / old_gain
        return self

    def no_relu(self) -> 'ConvBlock':
        """
        Remove the ReLU
        """
        del self.relu
        self.relu = None
        return self

    def to_preact(self) -> 'ConvBlock':
        """
        Place the normalization and ReLU before the convolution.
        """
        self.reset()
        c = self.conv
        del self.conv
        self.add_module('conv', c)
        return self
