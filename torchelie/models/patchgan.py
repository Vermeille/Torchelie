import torch.nn as nn
import torchelie.nn as tnn
from torchelie.utils import kaiming
from typing import List
from torchelie.transforms.differentiable import BinomialFilter2d


class PatchDiscriminator(nn.Module):
    def __init__(self, arch: List[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            tnn.Conv2dBNReLU(3, arch[0], kernel_size=4,
                             stride=2).remove_batchnorm().leaky()
        ]

        in_ch = arch[0]
        self.in_channels = in_ch
        for next_ch in arch[1:]:
            layers.append(
                tnn.Conv2dBNReLU(in_ch, next_ch, kernel_size=4,
                                 stride=2).leaky())
            in_ch = next_ch
        assert isinstance(layers[-1], tnn.Conv2dBNReLU)
        layers[-1].conv.stride = (1, 1)

        self.features = tnn.CondSeq(*layers)
        self.classifier = tnn.Conv2d(in_ch, 1, 4)

    def forward(self, x):
        return self.classifier(self.features(x))

    def to_equal_lr(self, leak=0.2) -> 'ResidualDiscriminator':
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                kaiming(m, dynamic=True, a=leak)

        return self

    def to_binomial_downsampling(self) -> 'PatchDiscriminator':
        for m in self.features.modules():
            if isinstance(m, tnn.Conv2dBNReLU):
                if m.conv.stride[0] != 2:
                    continue
                tnn.utils.insert_before(m, 'conv', BinomialFilter2d(2), 'pool')
                m.conv.stride = (1, 1)
        return self

    def to_avg_pool(self) -> 'PatchDiscriminator':
        for m in self.features.modules():
            if isinstance(m, tnn.Conv2dBNReLU):
                if m.conv.stride[0] != 2:
                    continue
                tnn.utils.insert_before(m, 'conv', nn.AvgPool2d(2), 'pool')
                m.conv.stride = (1, 1)
        return self

    def set_input_specs(self, in_channels: int) -> 'PatchDiscriminator':
        c = self.features[0].conv
        assert isinstance(c, nn.Conv2d)
        self.features[0].conv = kaiming(nn.Conv2d(in_channels,
                                                     c.out_channels,
                                                     4,
                                                     stride=2,
                                                     padding=c.padding,
                                                     bias=c.bias is not None),
                                           a=0.2)
        return self

    def set_kernel_size(self, kernel_size: int) -> 'PatchDiscriminator':
        def change_ks(m):
            if isinstance(m, nn.Conv2d) and m.kernel_size[0] != 1:
                return kaiming(
                    nn.Conv2d(m.in_channels,
                              m.out_channels,
                              kernel_size,
                              m.stride,
                              padding=kernel_size // 2))
            return m

        tnn.utils.edit_model(self.features, change_ks)
        return self


def patch286() -> PatchDiscriminator:
    """
    Patch Discriminator from pix2pix
    """
    return PatchDiscriminator([64, 128, 256, 512, 512, 512])


def patch70() -> PatchDiscriminator:
    """
    Patch Discriminator from pix2pix
    """
    return PatchDiscriminator([64, 128, 256, 512])


def patch34() -> PatchDiscriminator:
    """
    Patch Discriminator from pix2pix
    """
    return PatchDiscriminator([64, 128, 256])


def patch16() -> PatchDiscriminator:
    """
    Patch Discriminator from pix2pix
    """
    return PatchDiscriminator([64, 128])
