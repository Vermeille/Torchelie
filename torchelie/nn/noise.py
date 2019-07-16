import torch
import torch.nn as nn


class Noise(nn.Module):
    def __init__(self, ch):
        super(Noise, self).__init__()
        self.a = nn.Parameter(torch.zeros(ch, 1, 1))

    def forward(self, x):
        return x + self.a * torch.randn_like(x)
