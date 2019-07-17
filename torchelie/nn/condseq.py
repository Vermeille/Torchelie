import torch.nn as nn


class CondSeq(nn.Sequential):
    def condition(self, z):
        for m in self._modules.values():
            if hasattr(m, 'condition'):
                m.condition(z)

    def forward(self, x, z=None):
        for m in self._modules.values():
            if hasattr(m, 'condition'):
                x = m(x, z)
            else:
                x = m(x)
        return x
