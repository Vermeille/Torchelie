import random
import math
import torch

from enum import Enum
from torch import Tensor
from typing import List, Tuple, Optional, Callable

import PIL
from PIL.Image import Image as PILImage

from torchvision.transforms import functional as F, InterpolationMode
import torchvision.transforms as TF
from torchelie.utils import indent

from .augments import *

__all__ = ['RandAugment']


class RandAugment(torch.nn.Module):
    """
    RandAugment policy from RandAugment: Practical automated data
    augmentation with a reduced search space.

    Args:
        n_transforms (int): how many transforms to apply
        magnitude (float): magnitude of the transforms. 10 is base rate, can
            be set to more.
        interpolation: interpolation to use for suitable transforms
        fill: fill value to use for suitable transforms
    """

    def __init__(self,
                 n_transforms: int,
                 magnitude: float,
                 interpolation: InterpolationMode = InterpolationMode.BILINEAR,
                 fill: Optional[List[float]] = None):
        super().__init__()
        assert 0 <= magnitude and magnitude <= 30
        self.interpolation = interpolation
        self.fill = fill
        self.n_transforms = n_transforms
        magnitude /= 30
        self.magnitude = magnitude

        self.clear()
        self.add_colors()
        self.add_geometric()

    def clear(self):
        self.transforms = [Identity()]
        return self

    def add_extended_colors(self):
        magnitude = self.magnitude
        self.transforms += [
            Solarize(magnitude * 256),
            TF.RandomEqualize(p=1),
            TF.ColorJitter(hue=0.5 * magnitude),
        ]
        return self

    def add_colors(self):
        magnitude = self.magnitude
        self.transforms += [
            TF.RandomAutocontrast(p=1),
            Posterize(min_bits=8 - magnitude * 6, max_bits=8),
            TF.ColorJitter(saturation=0.9 * magnitude),
            TF.ColorJitter(contrast=0.9 * magnitude),
            TF.ColorJitter(brightness=0.9 * magnitude),
            TF.RandomAdjustSharpness(0.9 * magnitude, p=1),
            Lighting(1 * magnitude),
        ]
        return self

    def add_geometric(self):
        magnitude = self.magnitude
        self.transforms += [
            TF.RandomAffine(degrees=magnitude * 30,
                            interpolation=self.interpolation,
                            fill=self.fill),
            TF.RandomAffine(0,
                            shear=(-15 * magnitude, 15 * magnitude, 0, 0),
                            interpolation=self.interpolation,
                            fill=self.fill),
            TF.RandomAffine(0,
                            shear=(0, 0, -15 * magnitude, 15 * magnitude),
                            interpolation=self.interpolation,
                            fill=self.fill),
            Cutout(0., 0.1 + magnitude * 0.6),
            TF.RandomAffine(0,
                            translate=(0.3 * magnitude, 0),
                            interpolation=self.interpolation,
                            fill=self.fill),
            TF.RandomAffine(0,
                            translate=(0, 0.3 * magnitude),
                            interpolation=self.interpolation,
                            fill=self.fill),
        ]
        return self

    def forward(self, img: PILImage) -> PILImage:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: RandAugmented image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F._get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        for op in random.choices(self.transforms, k=self.n_transforms):
            img = op(img)

        return img

    def add_transform(self, tfm: Callable[[PILImage],
                                          PILImage]) -> 'RandAugment':
        self.transforms.append(tfm)
        return self

    def add_scale(self) -> 'RandAugment':
        s = 0.3 * self.magnitude
        return self.add_transform(
            TF.RandomAffine(0,
                            scale=(max(0, 1 - s), 1 + s),
                            interpolation=self.interpolation,
                            fill=self.fill))

    def add_greyscale(self) -> 'RandAugment':
        return self.add_transform(TF.RandomGrayscale(p=1))

    def add_subsampling(self) -> 'RandAugment':
        return self.add_transform(
            Subsample(int(self.magnitude * 8), 1., self.interpolation))

    def add_jpeg(self):
        return self.add_transform(JPEGArtifacts(1 - self.magnitude, p=1))

    def berserk_mode(self) -> 'RandAugment':
        """
        Load even more transforms
        """
        self.add_scale()
        self.add_greyscale()
        self.add_perspective()
        self.add_subsampling()
        self.add_jpeg()
        #self.add_transform(SampleMixup())
        #self.add_transform(SampleCutmix())
        return self

    def add_perspective(self) -> 'RandAugment':
        return self.add_transform(
            TF.RandomPerspective(max(0, min(self.magnitude * 0.3, 1)), 1.0,
                                 self.interpolation, self.fill))

    def __repr__(self) -> str:
        return (self.__class__.__name__ + '(n_transforms={}, magnitude={},'
                ' fill={}, transforms=[\n{}\n])'.format(
                    self.n_transforms, self.magnitude, self.fill,
                    indent(',\n'.join([repr(t) for t in self.transforms]))))
