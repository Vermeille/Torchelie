from typing import Optional
import random
import math
from io import BytesIO

import torch
import numpy as np
import PIL
from PIL.Image import Image as PILImage

from torchvision.transforms import functional as F, InterpolationMode

try:
    import cv2
    HAS_CV = True
except:
    HAS_CV = False

__all__ = [
    'Posterize',
    'Solarize',
    'Cutout',
    'Identity',
    'Subsample',
    'JPEGArtifacts',
    'Canny',
]


class Posterize:
    """
    Apply a Posterize filter with a random number of bits.

    Args:
        min_bits (int): minimum color encoding bits
        max_bits (int): maximum color encoding bits
    """

    def __init__(self, min_bits: int = 4, max_bits: int = 8):
        self.min_bits = int(min_bits)
        self.max_bits = int(max_bits)

    def __call__(self, x: PILImage) -> PILImage:
        return F.posterize(x, random.randint(self.min_bits, self.max_bits))

    def __repr__(self) -> str:
        return f'Posterize(min_bits={self.min_bits}, max_bits={self.max_bits})'


class Solarize:
    """
    Apply a Solarize filter with a random threshold.

    Args:
        max_thresh (int): upper bound for the random threshold.
    """

    def __init__(self, max_thresh: int = 128):
        self.max_thresh = int(max_thresh)

    def __call__(self, x: PILImage) -> PILImage:
        return F.solarize(x, 255 - random.randint(0, self.max_thresh))

    def __repr__(self) -> str:
        return f'Solarize(max_thresh={self.max_thresh})'


class Cutout:
    """
    Applies a random Cutout filter erasing at most :code:`max_size*100`% of
    the picture.

    Args:
        max_size (float): the maximum ratio that can be erased. 0 means no
            erasure, 1 means up to the whole image can be erased.
    """

    def __init__(self, min_size: float, max_size: float):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, x: PILImage) -> PILImage:
        w, h = x.size
        v = int(np.random.uniform(self.min_size, self.max_size) * min(w, h))

        x0 = np.random.uniform(0, w - v)
        y0 = np.random.uniform(0, h - v)

        x0 = int(max(0, x0))
        y0 = int(max(0, y0))
        x1 = int(min(w, x0 + v))
        y1 = int(min(h, y0 + v))

        xy = (x0, y0, x1, y1)
        color = (125, 123, 114)
        img = x.copy()
        PIL.ImageDraw.Draw(img).rectangle(xy, color)
        return img

    def __repr__(self) -> str:
        return f'Cutout(min_size={self.min_size}, max_size={self.max_size})'


class Identity:
    """
    Do nothing
    """

    def __call__(self, x: PILImage) -> PILImage:
        return x

    def __repr__(self) -> str:
        return 'Identity()'


class Subsample:
    """
    Randomly subsample images.

    Args:
        p (float): the transform is applied with probability p
        max_ratio (int): maximum subscaling factor
        interpolation (InterpolationMode): interpolation mode
    """

    def __init__(self,
                 max_ratio: int = 3,
                 p: float = 0.5,
                 interpolation: InterpolationMode = InterpolationMode.BILINEAR):
        self.max_ratio = max_ratio
        self.p = p
        self.interpolation = interpolation

    def __call__(self, x: PILImage) -> PILImage:
        if random.uniform(0, 1) >= self.p:
            return x

        size = min(x.size)
        x = F.resize(x, int(size // random.uniform(1, self.max_ratio)),
                     self.interpolation)
        x = F.resize(x, size, self.interpolation)
        return x

    def __repr__(self) -> str:
        return "SubsampleResize(p={}, max_ratio={})".format(
            self.p, self.max_ratio)


class JPEGArtifacts:
    """
    Add some random jpeg compression artifacts

    Args:
        p (float): probability of applying the filter
        min_compression (float): minimum quality (1: maximum quality)
    """

    def __init__(self, min_compression: float = 0.5, p: float = 0.5):
        self.p = p
        self.min_compression = min_compression

    def __call__(self, x: PILImage) -> PILImage:
        if random.uniform(0, 1) > self.p:
            return x
        f = BytesIO()
        x.save(f,
               format='JPEG',
               quality=random.randint(int(self.min_compression * 100), 100))
        f.seek(0)
        return PIL.Image.open(f)

    def __repr__(self) -> str:
        return f'JPEGArtifacts(min_compression={self.min_compression})'


class Canny:
    """
    Run Canny edge detector over an image. Requires OpenCV to be installed

    Args:
        thresh_low (int): lower threshold (default: 100)
        thresh_high (int): upper threshold (default: 200)
    """

    is_available: bool = HAS_CV

    def __init__(self, thresh_low: int = 100, thresh_high: int = 200):
        self.thresh_low = thresh_low
        self.thresh_high = thresh_high
        if not HAS_CV:
            print("Can't import OpenCV. Canny will notwork properly")

    def __call__(self, img: PILImage) -> PILImage:
        """
        Detect edges

        Args:
            img (PIL.Image): the image

        Returns:
            edges detected in `img` as PIL Image
        """
        if not HAS_CV:
            return img

        img = np.array(img)
        img = cv2.Canny(img, self.thresh_low, self.thresh_high)
        return PIL.Image.fromarray(img).convert('RGB')

    def __repr__(self) -> str:
        return 'Canny()'


class SampleMixup:
    last_img: Optional[PILImage]

    def __init__(self, alpha=0.4):
        self.last_img = None
        self.mixer = torch.distributions.Beta(alpha, alpha)

    def __call__(self, x):
        if self.last_img is None:
            self.last_img = x
            return x

        t = self.mixer.sample(torch.tensor([1])).item()
        second = self.last_img.resize(x.size, PIL.Image.BICUBIC)
        self.last_img = x
        return PIL.Image.blend(x, second, alpha=min(t, 1 - t))

    def __repr__(self):
        return 'SampleMixup()'


class SampleCutmix:
    last_img: Optional[PILImage]

    def __init__(self, alpha=0.4):
        self.last_img = None

    def __call__(self, x):
        if self.last_img is None:
            self.last_img = x
            return x

        w, h = x.size
        v = int(np.random.uniform(0.1, 0.5) * min(w, h))

        x0 = int(max(0, np.random.uniform(0, w - v)))
        y0 = int(max(0, np.random.uniform(0, h - v)))

        x1 = int(min(w, x0 + v))
        y1 = int(min(h, y0 + v))

        xy = (x0, y0, x1, y1)
        color = (125, 123, 114)
        img = x.copy()
        second = self.last_img.resize(x.size, PIL.Image.BICUBIC)
        second = second.crop(xy)
        img.paste(second, xy)
        self.last_img = x
        return img

    def __repr__(self):
        return 'SampleCutmix()'
