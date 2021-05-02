import torch
import torch.nn as nn
import torchelie.utils as tu
import torchelie.nn as tnn


@tu.experimental
class Hourglass(nn.Module):
    """
    Hourglass model from Deep Image Prior.
    """

    def __init__(self,
                 noise_dim=32,
                 down_channels=[128, 128, 128, 128, 128],
                 skip_channels=4,
                 down_kernel=[3, 3, 3, 3, 3],
                 up_kernel=[3, 3, 3, 3, 3],
                 upsampling='bilinear') -> None:
        super().__init__()

        assert (len(down_channels) == len(down_kernel)), (len(down_channels),
                                                          len(down_kernel))
        assert (len(down_channels) == len(up_kernel)), (len(down_channels),
                                                        len(up_kernel))

        self.upsampling = upsampling
        self.downs = nn.ModuleList(
            [self.down(noise_dim, down_channels[0], down_kernel[0])] + [
                self.down(d1, d2, down_kernel[0]) for d1, d2, k in zip(
                    down_channels[:-1], down_channels[1:], down_kernel[1:])
            ])

        self.ups = nn.ModuleList([
            self.up(down_channels[-1]
                    + skip_channels, down_channels[-1], up_kernel[0])
        ] + [
            self.up(d1 + skip_channels, d2, k) for d1, d2, k in zip(
                down_channels[:0:-1], down_channels[-2::-1], up_kernel[1:])
        ])

        if skip_channels != 0:
            self.skips = nn.ModuleList(
                [self.skip(d, skip_channels) for d in down_channels])

        self.to_rgb = tnn.ConvBlock(down_channels[0], 3, up_kernel[-1])
        self.to_rgb.no_relu()

        tnn.utils.make_leaky(self)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.padding_mode = 'reflect'

    def down(self, in_ch, out_ch, ks) -> nn.Sequential:
        return nn.Sequential(
            tnn.ConvBlock(in_ch, out_ch, ks, stride=2),
            tnn.ConvBlock(out_ch, out_ch, ks, stride=1),
        )

    def up(self, in_ch, out_ch, ks) -> nn.Sequential:
        conv = tnn.ConvBlock(in_ch, out_ch, ks)
        tnn.utils.insert_before(conv, 'conv', nn.BatchNorm2d(in_ch), 'pre_bn')
        return conv

    def skip(self, in_ch, out_ch) -> nn.Sequential:
        return tnn.ConvBlock(in_ch, out_ch, 1)

    def forward(self, x) -> torch.Tensor:
        acts = [x]
        for d in self.downs:
            acts.append(d(acts[-1]))
        acts = acts[1:]

        if hasattr(self, 'skips'):
            skips = [s(a) for s, a in zip(self.skips, acts)]

            x = acts[-1]
            for u, s in zip(self.ups, reversed(skips)):
                x = nn.functional.interpolate(x,
                                              size=s.shape[2:],
                                              mode=self.upsampling)
                x = u(torch.cat([x, s], dim=1))
        else:
            x = acts[-1]
            for u, x2 in zip(self.ups, reversed(acts)):
                x = nn.functional.interpolate(x,
                                              size=x2.shape[2:],
                                              mode=self.upsampling)
                x = u(x)

        x = nn.functional.interpolate(x, scale_factor=2, mode=self.upsampling)
        return torch.sigmoid(self.to_rgb(x))
