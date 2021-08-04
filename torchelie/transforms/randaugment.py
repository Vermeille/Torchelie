import random
import math
import torch

from enum import Enum
from torch import Tensor
from typing import List, Tuple, Optional

import PIL
import numpy as np

from torchvision.transforms import functional as F, InterpolationMode

__all__ = ['RandAugment']


class RandAugment(torch.nn.Module):
    transforms = [
        ('Identity', None, None),
        ('AutoContrast', None, None),
        ('Equalize', None, None),
        ('Invert', None, None),
        ('Rotate', 0, 30),
        ('Posterize', 4, 8),
        ('Solarize', 0, 256),
        ('SolarizeAdd', 0, 110),
        ('Color', 0., 0.9),
        ('Contrast', 0., 0.9),
        ('Brightness', 0., 0.9),
        ('Sharpness', 0., 0.9),
        ('ShearX', 0., 0.3),
        ('ShearY', 0., 0.3),
        ('Cutout', 0, 0.5),
        ('TranslateX', 0, 0.2),
        ('TranslateY', 0, 0.2),
    ]

    def __init__(self,
                 n_transforms: int,
                 magnitude: int,
                 interpolation: InterpolationMode = InterpolationMode.NEAREST,
                 fill: Optional[List[float]] = None):
        super().__init__()
        self.interpolation = interpolation
        self.fill = fill
        self.n_transforms = n_transforms
        self.magnitude = magnitude

    def forward(self, img: Tensor):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: AutoAugmented image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F._get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        for op_name, minv, maxv in random.sample(self.transforms,
                                                 self.n_transforms):

            if maxv is None:
                magnitude = None
            else:
                magnitude = random.uniform(minv, maxv * self.magnitude / 10)
                signed_magnitude = magnitude * 2 - self.magnitude * maxv / 10

            if op_name == 'Identity':
                pass
            elif op_name == "ShearX":
                img = F.affine(img,
                               angle=0.0,
                               translate=[0, 0],
                               scale=1.0,
                               shear=[math.degrees(signed_magnitude), 0.0],
                               interpolation=self.interpolation,
                               fill=fill)
            elif op_name == "ShearY":
                img = F.affine(img,
                               angle=0.0,
                               translate=[0, 0],
                               scale=1.0,
                               shear=[0.0, math.degrees(signed_magnitude)],
                               interpolation=self.interpolation,
                               fill=fill)
            elif op_name == "TranslateX":
                img = F.affine(
                    img,
                    angle=0.0,
                    translate=[
                        int(F._get_image_size(img)[0] * signed_magnitude), 0
                    ],
                    scale=1.0,
                    interpolation=self.interpolation,
                    shear=[0.0, 0.0],
                    fill=fill)
            elif op_name == "TranslateY":
                img = F.affine(
                    img,
                    angle=0.0,
                    translate=[
                        0, int(F._get_image_size(img)[1] * signed_magnitude)
                    ],
                    scale=1.0,
                    interpolation=self.interpolation,
                    shear=[0.0, 0.0],
                    fill=fill)
            elif op_name == "Rotate":
                img = F.rotate(img,
                               signed_magnitude,
                               interpolation=self.interpolation,
                               fill=fill)
            elif op_name == "Brightness":
                img = F.adjust_brightness(img, 1.0 + signed_magnitude)
            elif op_name == "Color":
                img = F.adjust_saturation(img, 1.0 + signed_magnitude)
            elif op_name == "Contrast":
                img = F.adjust_contrast(img, 1.0 + signed_magnitude)
            elif op_name == "Sharpness":
                img = F.adjust_sharpness(img, 1.0 + signed_magnitude)
            elif op_name == "Posterize":
                img = F.posterize(img, int(magnitude))
            elif op_name == "Solarize":
                img = F.solarize(img, magnitude)
            elif op_name == "SolarizeAdd":
                img_np = np.array(img).astype(np.int)
                img_np = img_np + signed_magnitude
                img_np = np.clip(img_np, 0, 255)
                img_np = img_np.astype(np.uint8)
                img = PIL.Image.fromarray(img_np)
                img = F.solarize(img, 128)
            elif op_name == "AutoContrast":
                img = F.autocontrast(img)
            elif op_name == "Equalize":
                img = F.equalize(img)
            elif op_name == "Invert":
                img = F.invert(img)
            elif op_name == 'Cutout':
                w, h = img.size
                x0 = np.random.uniform(w)
                y0 = np.random.uniform(h)

                v = int(magnitude * min(w, h))
                x0 = int(max(0, x0 - v / 2.))
                y0 = int(max(0, y0 - v / 2.))
                x1 = min(w, x0 + v)
                y1 = min(h, y0 + v)

                xy = (x0, y0, x1, y1)
                color = (125, 123, 114)
                img = img.copy()
                PIL.ImageDraw.Draw(img).rectangle(xy, color)
            else:
                raise ValueError(
                    "The provided operator {} is not recognized.".format(
                        op_name))

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(n_transforms={}, magnitude={}, fill={})'.format(
            self.n_transforms, self.magnitude, self.fill)
