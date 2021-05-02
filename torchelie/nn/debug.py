import crayons

import torch.nn as nn
from torchelie.utils import experimental


class Debug(nn.Module):
    """
    An pass-through layer that prints some debug info during forward pass.
    It prints its name, the input's shape, mean of channels means, mean,
    mean of channels std, and std.

    Args:
        name (str): this layer's name
    """
    @experimental
    def __init__(self, name):
        super(Debug, self).__init__()
        self.name = name

    def forward(self, x):
        print(crayons.yellow(self.name))
        print(crayons.yellow('----'))
        print('Shape {}'.format(x.shape))
        if x.ndim == 2:
            print("Stats mean {:.2f} var {:.2f}".format(x.mean().item(),
                  x.std().item()))
        if x.ndim == 4:
            print("Stats mean {:.2f} {:.2f} var s{:.2f} {:.2f}".format(
                x.mean(dim=[0, 2, 3]).mean().item(),
                x.mean().item(),
                x.std(dim=[0, 2, 3]).mean().item(),
                x.std().item()))
        print()
        return x


class Dummy(nn.Module):
    """
    A pure pass-through layer
    """
    def forward(self, x):
        return x
