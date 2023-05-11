import torch


class PixelNorm(torch.nn.Module):
    """
    PixelNorm from ProgressiveGAN
    """
    def forward(self, x):
        return x / (x.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-8)


class ChannelNorm(torch.nn.Module):
    def __init__(self, dim=1, affine=True, channels=1):
        super().__init__()
        self.affine = affine
        if affine:
            self.weight = torch.nn.Parameter(torch.ones(channels))
            #self.bias = torch.nn.Parameter(torch.zeros(channels))

        if isinstance(dim, int):
            dim = [dim]
        self.dim = dim

        if isinstance(channels, int):
            channels = [channels]
        self.channels = channels

    def forward(self, x):
        var = x.var(dim=self.dim, keepdim=True, unbiased=False)
        if not self.affine:
            return (x ) * var.rsqrt()
        expand = [(self.channels[self.dim.index(i)] if i in self.dim else 1)
                  for i in range(x.dim())]
        w = self.weight.view(expand)
        w_ = var.rsqrt() * w
        return x * w_

        mean = x.mean(dim=self.dim, keepdim=True)
        var = x.var(dim=self.dim, keepdim=True, unbiased=False)
        if not self.affine:
            return (x - mean) * var.rsqrt()
        expand = [(self.channels[self.dim.index(i)] if i in self.dim else 1)
                  for i in range(x.dim())]
        w = self.weight.view(expand)
        b = self.bias.view(expand)
        w_ = var.rsqrt() * w
        return x * w_ - (mean * w_ + b)
