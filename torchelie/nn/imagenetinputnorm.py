import torch
import torch.nn as nn


class ImageNetInputNorm(nn.Module):
    """
    Normalize images channels as torchvision models expects, in a
    differentiable way
    """
    def __init__(self):
        super(ImageNetInputNorm, self).__init__()
        self.register_buffer(
            'norm_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))

        self.register_buffer(
            'norm_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input):
        return (input - self.norm_mean) / self.norm_std
