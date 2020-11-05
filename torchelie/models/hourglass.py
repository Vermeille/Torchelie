import torch
import torch.nn as nn
import torchelie.utils as tu
import torchelie.nn as tnn
import numpy as np


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
                 upsampling='bilinear',
                 pad=nn.ReflectionPad2d,
                 relu=nn.LeakyReLU(0.2, True)):
        """
        Initialize the kernel.

        Args:
            self: (todo): write your description
            noise_dim: (str): write your description
            down_channels: (todo): write your description
            skip_channels: (todo): write your description
            down_kernel: (todo): write your description
            up_kernel: (str): write your description
            upsampling: (todo): write your description
            pad: (todo): write your description
            nn: (todo): write your description
            ReflectionPad2d: (str): write your description
            relu: (todo): write your description
            nn: (todo): write your description
            LeakyReLU: (todo): write your description
        """
        super(Hourglass, self).__init__()

        assert (len(down_channels) == len(down_kernel)), (len(down_channels),
                                                          len(down_kernel))
        assert (len(down_channels) == len(up_kernel)), (len(down_channels),
                                                        len(up_kernel))

        self.upsampling = upsampling
        self.pad = pad
        self.relu = relu
        self.downs = nn.ModuleList(
            [self.down(noise_dim, down_channels[0], down_kernel[0])] + [
                self.down(d1, d2, down_kernel[0]) for d1, d2, k in zip(
                    down_channels[:-1], down_channels[1:], down_kernel[1:])
            ])

        self.ups = nn.ModuleList([
            self.up(down_channels[-1] +
                    skip_channels, down_channels[-1], up_kernel[0])
        ] + [
            self.up(d1 + skip_channels, d2, k) for d1, d2, k in zip(
                down_channels[:0:-1], down_channels[-2::-1], up_kernel[1:])
        ])

        if skip_channels != 0:
            self.skips = nn.ModuleList(
                [self.skip(d, skip_channels) for d in down_channels])

        self.to_rgb = nn.Sequential(
            self.pad(up_kernel[-1] // 2),
            (nn.Conv2d(down_channels[0], 3, up_kernel[-1])), nn.BatchNorm2d(3))

    def down(self, in_ch, out_ch, ks):
        """
        Downsample the output ).

        Args:
            self: (todo): write your description
            in_ch: (int): write your description
            out_ch: (str): write your description
            ks: (int): write your description
        """
        return nn.Sequential(
            self.pad(ks // 2),
            (nn.Conv2d(in_ch, out_ch, ks, stride=2)),
            nn.BatchNorm2d(out_ch),
            self.relu,
            self.pad(ks // 2),
            (nn.Conv2d(out_ch, out_ch, ks)),
            nn.BatchNorm2d(out_ch),
            self.relu,
        )

    def up(self, in_ch, out_ch, ks):
        """
        Updates the output tensor.

        Args:
            self: (todo): write your description
            in_ch: (int): write your description
            out_ch: (todo): write your description
            ks: (todo): write your description
        """
        return nn.Sequential(
            nn.BatchNorm2d(in_ch),
            self.pad(ks // 2),
            (nn.Conv2d(in_ch, out_ch, ks)),
            nn.BatchNorm2d(out_ch),
            self.relu,
        )

    def skip(self, in_ch, out_ch):
        """
        Skip the output from_chunk.

        Args:
            self: (todo): write your description
            in_ch: (int): write your description
            out_ch: (str): write your description
        """
        return nn.Sequential(
            (nn.Conv2d(in_ch, out_ch, 1)),
            nn.BatchNorm2d(out_ch),
            self.relu,
        )

    def forward(self, x):
        """
        Forward computation. forward.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
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
