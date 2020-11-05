import torch
import torch.nn as nn
import torch.nn.functional as F

import torchelie.nn as tnn
from torchelie.utils import xavier, kaiming


class AutoGAN(nn.Module):
    """
    Generator discovered in AutoGAN: Neural Architecture Search for Generative
    Adversarial Networks.

    Args:
        arch (list): architecture specification: a list of output channel for
            each block. Each block doubles the resolution of the generated
            image. Example: `[512, 256, 128, 64, 32]`.
        n_skip_max (int): how many blocks far back will be used for the skip
            connections maximum.
        in_noise (int): dimension of the input noise vector
        out_ch (int): number of channels on the image
        batchnorm_in_output (bool): whether to have a batchnorm just before
            projecting to RGB. I have found it better on False, but the
            official AutoGAN repo has it.
    """

    def __init__(self,
                 arch,
                 n_skip_max=2,
                 in_noise=256,
                 out_ch=3,
                 batchnorm_in_output=False):
        """
        Initialize the architecture.

        Args:
            self: (todo): write your description
            arch: (todo): write your description
            n_skip_max: (float): write your description
            in_noise: (int): write your description
            out_ch: (str): write your description
            batchnorm_in_output: (int): write your description
        """
        super(AutoGAN, self).__init__()
        self.n_skip_max = n_skip_max
        self.make_noise = xavier(nn.Linear(in_noise, 4 * 4 * arch[0]))

        in_ch = arch[0]
        blocks = []
        lasts = []
        for i, out in enumerate(arch[1:]):
            mode = 'nearest' if i % 2 == 0 else 'bilinear'
            blocks.append(tnn.AutoGANGenBlock(in_ch, out, lasts, mode=mode))
            lasts = ([out] + lasts)[:n_skip_max]
            in_ch = out
        self.blocks = nn.ModuleList(blocks)
        if batchnorm_in_output:
            self.to_rgb = nn.Sequential(nn.BatchNorm2d(arch[-1]),
                                        nn.ReLU(True),
                                        xavier(tnn.Conv3x3(arch[-1], out_ch)))
        else:
            self.to_rgb = nn.Sequential(nn.ReLU(True),
                                        xavier(tnn.Conv3x3(arch[-1], out_ch)))

    def forward(self, z):
        """
        Forward pass

        Args:
            z (tensor): A batch of noise vectors

        Returns:
            generated batch of images
        """
        x = self.make_noise(z)
        x = x.view(x.shape[0], -1, 4, 4)

        skips = []
        for b in self.blocks:
            x, sk = b(x, skips)
            skips = ([sk] + skips)[:self.n_skip_max]
        return torch.sigmoid(self.to_rgb(F.leaky_relu(x, 0.2)))


def autogan_128(in_noise, out_ch=3):
    """
    Autogan autogan autogan noise.

    Args:
        in_noise: (int): write your description
        out_ch: (todo): write your description
    """
    return AutoGAN(arch=[512, 512, 256, 128, 64, 32],
                   n_skip_max=3,
                   in_noise=in_noise,
                   out_ch=out_ch)


def autogan_64(in_noise, out_ch=3):
    """
    Autogan autogan noise.

    Args:
        in_noise: (int): write your description
        out_ch: (todo): write your description
    """
    return AutoGAN(arch=[512, 256, 128, 64, 32],
                   n_skip_max=3,
                   in_noise=in_noise,
                   out_ch=out_ch)


def autogan_32(in_noise, out_ch=3):
    """
    Autogan autogan noise noise.

    Args:
        in_noise: (int): write your description
        out_ch: (todo): write your description
    """
    return AutoGAN(arch=[256, 128, 64, 32],
                   n_skip_max=3,
                   in_noise=in_noise,
                   out_ch=out_ch)
