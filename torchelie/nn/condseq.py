import torch.nn as nn
from typing import Dict, Any


class CondSeq(nn.Sequential):
    """
    An extension to torch's Sequential that allows conditioning either as a
    second forward argument or `condition()`
    """
    _modules: Dict[str, nn.Module]

    def condition(self, z: Any):
        """
        Conditions all the layers on z

        Args:
            z: conditioning
        """
        for m in self._modules.values():
            if hasattr(m, 'condition'):
                m.condition(z)

    def forward(self, x, z=None):
        """
        Forward pass

        Args:
            x: input
            z (optional): conditioning. condition() must be called first if
                left None
        """
        for m in self._modules.values():
            if hasattr(m, 'condition') and z is not None:
                x = m(x, z)
            else:
                x = m(x)
        return x
