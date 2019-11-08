"""
Those classes are different ways of averaging metrics.
"""

import torchelie.utils as tu

class RunningAvg(tu.AutoStateDict):
    """
    Average by keeping the whole sum and number of elements of the data logged.
    Useful when the metrics come per batch and an accurate number for the whole
    epoch is needed.
    """
    def __init__(self):
        super(RunningAvg, self).__init__()
        self.count = 0
        self.val = 0

    def log(self, x, total=1):
        """
        Log metric

        Args:
            x: the metric
            total: how many element does this represent
        """
        self.count += total
        self.val += x

    def get(self):
        """
        Get the average so far
        """
        return self.val / self.count


class WindowAvg(tu.AutoStateDict):
    """
    Average a window containing the `k` previous logged values

    Args:
        k (int): the window's length
    """
    def __init__(self, k=100):
        super(WindowAvg, self).__init__()
        self.vals = []
        self.k = k

    def log(self, x):
        """
        Log `x`
        """
        if len(self.vals) == self.k:
            self.vals = self.vals[1:]
        self.vals.append(x)

    def get(self):
        """
        Return the value averaged over the window
        """
        return sum(self.vals) / len(self.vals)


class ExponentialAvg(tu.AutoStateDict):
    r"""
    Keep an exponentially decaying average of the values according to

    :math:`y := \beta y + (1 - \beta) x`

    Args:
        beta (float): the decay rate
    """
    def __init__(self, beta=0.6):
        super(ExponentialAvg, self).__init__()
        self.beta = beta
        self.val = None

    def log(self, x):
        """Log `x`"""
        if self.val is None:
            self.val = x
        else:
            self.val = self.beta * self.val + (1 - self.beta) * x

    def get(self):
        """Return the exponential average at this time step"""
        return self.val
