import torch.nn as nn
from typing import Any, Optional, Callable, cast


class CondSeq(nn.Sequential):
    """
    An extension to torch's Sequential that allows conditioning either as a
    second forward argument or `condition()`
    """
    def condition(self, z: Any) -> None:
        """
        Conditions all the layers on z

        Args:
            z: conditioning
        """
        for m in self:
            if hasattr(m, 'condition') and m is not self:
                cast(Callable, m.condition)(z)

    def forward(self, x: Any, z: Optional[Any] = None) -> Any:
        """
        Forward pass

        Args:
            x: input
            z (optional): conditioning. condition() must be called first if
                left None
        """
        for m in self:
            if hasattr(m, 'condition') and z is not None:
                x = m(x, z)
            else:
                x = m(x)
        return x
