import torch.nn as nn


class Lambda(nn.Module):
    def __init__(self, lam):
        super(Lambda, self).__init__()
        self.lam = lam

    def forward(self, x):
        return self.lam(x)


class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)
