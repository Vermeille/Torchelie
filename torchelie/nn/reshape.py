import torch.nn as nn


class Lambda(nn.Module):
    """
    Applies a lambda function on forward()

    Args:
        lamb (fn): the lambda function
    """
    def __init__(self, lam):
        """
        Initialize the lambda function

        Args:
            self: (todo): write your description
            lam: (array): write your description
        """
        super(Lambda, self).__init__()
        self.lam = lam

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return self.lam(x)


class Reshape(nn.Module):
    """
    Reshape the input volume

    Args:
        *shape (ints): new shape, WITHOUT specifying batch size as first
        dimension, as it will remain unchanged.
    """
    def __init__(self, *shape):
        """
        Initialize shape.

        Args:
            self: (todo): write your description
            shape: (int): write your description
        """
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return x.view(x.shape[0], *self.shape)
