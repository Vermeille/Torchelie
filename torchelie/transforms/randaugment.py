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

        self.transforms = [
            (Identity, None, None),
            (AutoContrast, None, None),
            (Equalize, None, None),
            (Invert, None, None),
            (Rotate, 0, 30),
            (Posterize, 4, 8),
            (Solarize, 0, 256),
            (SolarizeAdd, 0, 110),
            (Color, 0., 0.9),
            (Contrast, 0., 0.9),
            (Brightness, 0., 0.9),
            (Sharpness, 0., 0.9),
            (ShearX, 0., 0.3),
            (ShearY, 0., 0.3),
            (Cutout, 0, 0.5),
            (TranslateX, 0, 0.45),
            (TranslateY, 0, 0.45),
        ]

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

        for op, minv, maxv in random.choices(self.transforms,
                                             k=self.n_transforms):
            if maxv is None:
                magnitude = None
                signed_magnitude = None
            else:
                magnitude = random.uniform(minv, maxv * self.magnitude / 10)
                signed_magnitude = magnitude * 2 - self.magnitude * maxv / 10

            img = op(img, magnitude, signed_magnitude, self.interpolation,
                     self.fill)

        return img

    def add_transform(self, tfm, minv=None, maxv=None):
        self.transforms.append(tfm, minv, maxv)
        return self

    def __repr__(self):
        return self.__class__.__name__ + '(n_transforms={}, magnitude={}, fill={})'.format(
            self.n_transforms, self.magnitude, self.fill)


def Identity(x, *_):
    return x


def AutoContrast(x, *_):
    return F.autocontrast(x)


def Equalize(x, *_):
    return F.equalize(x)


def Invert(x, *_):
    return F.invert(x)


def Rotate(x, _, signed_magnitude, interpolation, fill):
    return F.rotate(x, signed_magnitude, interpolation=interpolation, fill=fill)


def Posterize(x, magnitude, *_):
    return F.posterize(x, int(magnitude))


def Solarize(x, magnitude, *_):
    return F.solarize(x, magnitude)


def SolarizeAdd(x, magnitude, signed_magnitude, *_):
    img_np = np.array(x).astype(np.int)
    img_np = img_np + signed_magnitude
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = PIL.Image.fromarray(img_np)
    return F.solarize(img, 128)


def Color(x, magnitude, signed_magnitude, *_):
    return F.adjust_saturation(x, 1.0 + signed_magnitude)


def Contrast(x, magnitude, signed_magnitude, *_):
    return F.adjust_contrast(x, 1.0 + signed_magnitude)


def Brightness(x, magnitude, signed_magnitude, *_):
    return F.adjust_brightness(x, 1.0 + signed_magnitude)


def Sharpness(x, magnitude, signed_magnitude, *_):
    return F.adjust_sharpness(x, 1.0 + signed_magnitude)


def ShearX(x, _, signed_magnitude, interpolation, fill):
    return F.affine(x,
                    angle=0.0,
                    translate=[0, 0],
                    scale=1.0,
                    shear=[math.degrees(signed_magnitude), 0.0],
                    interpolation=interpolation,
                    fill=fill)


def ShearY(x, _, signed_magnitude, interpolation, fill):
    return F.affine(x,
                    angle=0.0,
                    translate=[0, 0],
                    scale=1.0,
                    shear=[0.0, math.degrees(signed_magnitude)],
                    interpolation=interpolation,
                    fill=fill)


def Cutout(x, magnitude, *_):
    w, h = x.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    v = int(magnitude * min(w, h))
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    img = x.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def TranslateX(x, _, signed_magnitude, interpolation, fill):
    return F.affine(
        x,
        angle=0.0,
        translate=[int(F._get_image_size(x)[0] * signed_magnitude), 0],
        scale=1.0,
        interpolation=interpolation,
        shear=[0.0, 0.0],
        fill=fill)


def TranslateY(x, _, signed_magnitude, interpolation, fill):
    return F.affine(
        x,
        angle=0.0,
        translate=[0, int(F._get_image_size(x)[1] * signed_magnitude)],
        scale=1.0,
        interpolation=interpolation,
        shear=[0.0, 0.0],
        fill=fill)
