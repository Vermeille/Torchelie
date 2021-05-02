"""
Those classes are different ways of averaging metrics.
"""

import torchelie.utils as tu
from typing import List, Optional


class RunningAvg(tu.AutoStateDict):
    """
    Average by keeping the whole sum and number of elements of the data logged.
    Useful when the metrics come per batch and an accurate number for the whole
    epoch is needed.
    """
    def __init__(self) -> None:
        super(RunningAvg, self).__init__()
        self.count = 0.0
        self.val = 0.0

    def log(self, x: float, total: int = 1):
        """
        Log metric

        Args:
            x: the metric
            total: how many element does this represent
        """
        self.count += total
        self.val += x

    def get(self) -> float:
        """
        Get the average so far
        """
        if self.count == 0:
            return float('nan')
        return self.val / self.count


class WindowAvg(tu.AutoStateDict):
    """
    Average a window containing the `k` previous logged values

    Args:
        k (int): the window's length
    """
    def __init__(self, k: int = 100) -> None:
        super(WindowAvg, self).__init__()
        self.vals: List[float] = []
        self.k = k

    def log(self, x: float) -> None:
        """
        Log `x`
        """
        if len(self.vals) == self.k:
            self.vals = self.vals[1:]
        self.vals.append(x)

    def get(self) -> float:
        """
        Return the value averaged over the window
        """
        if len(self.vals) == 0:
            return float("nan")
        return sum(self.vals) / len(self.vals)


class ExponentialAvg(tu.AutoStateDict):
    r"""
    Keep an exponentially decaying average of the values according to

    :math:`y := \beta y + (1 - \beta) x`

    Args:
        beta (float): the decay rate
    """
    def __init__(self, beta: float = 0.6):
        super(ExponentialAvg, self).__init__()
        self.beta = beta
        self.val: Optional[float] = None

    def log(self, x: float) -> None:
        """Log `x`"""
        if self.val is None:
            self.val = x
        else:
            self.val = self.beta * self.val + (1 - self.beta) * x

    def get(self) -> float:
        """Return the exponential average at this time step"""
        assert self.val is not None, 'no value yet'
        return self.val
