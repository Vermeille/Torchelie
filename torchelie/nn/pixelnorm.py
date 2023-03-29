import torch


class PixelNorm(torch.nn.Module):
    """
    PixelNorm from ProgressiveGAN
    """

    def forward(self, x):
        return x / (x.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-8)


class ChannelNorm(torch.nn.Module):

    def __init__(self, dim=1, affine=True):
        super().__init__()
        self.affine = affine
        if affine:
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.bias = torch.nn.Parameter(torch.zeros(1))
        self.dim = dim

    def forward(self, x):
        mean = x.mean(dim=self.dim, keepdim=True)
        var = x.var(dim=self.dim, keepdim=True, unbiased=False)
        if not self.affine:
            return (x - mean) * var.rsqrt()
        w_ = var.rsqrt() * self.weight
        return x * w_ - (mean * w_ + self.bias)
