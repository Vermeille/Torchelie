from typing import List, Tuple, cast

import torchelie.nn as tnn
import torchelie.utils as tu
import torch.nn as nn


@tu.experimental
class UNet(nn.Module):
    def __init__(self, arch: List[int]) -> None:
        super().__init__()
        self.arch = arch
        self.in_channels = 3
        self.out_channels = arch[-1]

        feats = tnn.CondSeq()
        feats.input = tnn.Conv2dBNReLU(3, arch[0], 3)

        encdec: nn.Module = tnn.Conv2dBNReLU(arch[-1], arch[-1] * 2, 3)
        for outer, inner in zip(arch[-2::-1], arch[:0:-1]):
            encdec = tnn.UBlock(outer, inner, encdec)
        feats.encoder_decoder = encdec
        self.features = feats
        self.classifier = tnn.CondSeq()
        assert isinstance(encdec.out_channels, int)
        self.classifier.conv = tnn.Conv2dBNReLU(encdec.out_channels, 3,
                                                3).remove_batchnorm()

    def forward(self, x):
        return self.classifier(self.features(x))

    def set_input_specs(self, in_channels: int) -> 'UNet':
        assert isinstance(self.features.input, tnn.Conv2dBNReLU)
        c = self.features.input.conv
        self.features.input.conv = tu.kaiming(
            nn.Conv2d(in_channels,
                      c.out_channels,
                      cast(Tuple[int, int], c.kernel_size),
                      bias=c.bias is not None,
                      padding=cast(Tuple[int, int], c.padding)))
        return self

    def remove_first_batchnorm(self) -> 'UNet':
        assert isinstance(self.features.input, tnn.Conv2dBNReLU)
        self.features.input.remove_batchnorm()
        return self

    def remove_batchnorm(self) -> 'UNet':
        for m in self.modules():
            if isinstance(m, tnn.Conv2dBNReLU):
                m.remove_batchnorm()
        return self


