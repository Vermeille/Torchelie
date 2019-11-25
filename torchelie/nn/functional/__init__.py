from .vq import quantize

import torch
import torch.nn.functional as F


def laplacian(images, n_down=4):
    """
    Decompose a 4D images tensor into a laplacian pyramid.

    Args:
        images (4D Tensor): source images
        n_down (int): how many times to downscale the input

    Returns:
        A list of tensor: laplacian pyramid
    """
    lapls = []

    for i in range(n_down):
        n = F.interpolate(images, scale_factor=0.5, mode='bilinear',
                align_corners=True)
        lapls.append(images -
                     F.interpolate(n, size=images.shape[-2:], mode='bilinear',
                         align_corners=True))
        images = n

    lapls.append(images)
    return lapls


def combine_laplacians(laplacians):
    """
    Recombine a list of of tensor as returned by :code:`laplacian()` into an
    image batch

    Args:
        laplacians (list of tensors: laplacian pyramid

    Returns:
        a tensor
    """
    biggest = laplacians[0]

    rescaled = [biggest]
    for im in laplacians[1:]:
        rescaled.append(
                F.interpolate(im, size=biggest.shape[-2:], mode='bilinear',
                    align_corners=True))

    mixed = torch.stack(rescaled, dim=-1)
    return mixed.sum(dim=-1)
