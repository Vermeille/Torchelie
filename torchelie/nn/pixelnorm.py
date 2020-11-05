import torch

class PixelNorm(torch.nn.Module):
    """
    PixelNorm from ProgressiveGAN
    """
    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return x / (x.mean(dim=1, keepdim=True).sqrt() + 1e-8)
