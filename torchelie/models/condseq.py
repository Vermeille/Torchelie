import torch.nn as nn


class ConditionalSequential(nn.Sequential):
    def forward(self, x, z=None):
        for m in self._modules.values():
            if hasattr(m, 'condition'):
                x = m(x, z)
            else:
                x = m(x)
        return x
