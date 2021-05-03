import math

import torch
import torchvision.transforms as TF
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np

try:
    import cv2
    HAS_CV = True
except:
    pass


class ResizeNoCrop:
    """
    Resize a PIL image so that its longer border is of size `size`

    Args:
        size (int): max size of the image
    """

    def __init__(self, size, mode=Image.BILINEAR):
        self.size = size
        self.mode = mode

    def __call__(self, x):
        """
        Args:
            x (PIL.Image): the image to transform

        Returns:
            resized image
        """
        size = self.size
        width = x.width
        height = x.height
        if width > height:
            return x.resize((size, int(size * height / width)), self.mode)
        else:
            return x.resize((int(size * width / height), size), self.mode)


class AdaptPad:
    """
    Pad an input image so that it reaches size `size`

    Args:
        sz ((int, int)): target size
        padding_mode (str): one of the modes of `torchvision.transforms.pad`
    """

    def __init__(self, sz, padding_mode='constant', fill=0):
        self.sz = sz
        self.padding_mode = padding_mode
        self.fill = fill

    def __call__(self, img):
        """
        Pad the image

        Args:
            img (PIL.Image): The image to pad

        Returns:
            Padded image
        """
        w, h = img.width, img.height

        pl = max(0, self.sz[1] - w) // 2
        pr = self.sz[1] - w - pl

        pt = max(0, self.sz[0] - h) // 2
        pd = self.sz[0] - h - pt

        return TF.functional.pad(img, (pl, pt, pr, pd),
                                 padding_mode=self.padding_mode,
                                 fill=self.fill)


class MultiBranch:
    """
    Transform an image with multiple transforms

    Args:
        transforms (list of transforms): the parallel set of transforms
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        """
        Transform the image

        Args:
            x: image

        Returns:
            A tuple `out` so that `out[i] = transforms[i](x)`
        """
        return tuple(tf(x) for tf in self.transforms)


class Canny:
    """
    Run Canny edge detector over an image. Requires OpenCV to be installed

    Args:
        thresh_low (int): lower threshold (default: 100)
        thresh_high (int): upper threshold (default: 200)
    """

    def __init__(self, thresh_low=100, thresh_high=200):
        self.thresh_low = thresh_low
        self.thresh_high = thresh_high

    def __call__(self, img):
        """
        Detect edges

        Args:
            img (PIL.Image): the image

        Returns:
            edges detected in `img` as PIL Image
        """
        assert HAS_CV, ("Can't import OpenCV. Some transforms will not "
                        "work properly")
        img = np.array(img)
        img = cv2.Canny(img, self.thresh_low, self.thresh_high)
        return Image.fromarray(img)


class ResizedCrop(object):
    """Crop the given PIL Image to size.
    A crop of size of the original size is made. This crop
    is finally resized to given size.

    Args:
        size: expected output size of each edge
        scale: size of the origin size cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=0.54, ratio=1,
                 interpolation=Image.BILINEAR):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

        self.ratio = ratio
        self.interpolation = interpolation
        self.scale = scale

    @staticmethod
    def _get_image_size(img):
        if F._is_pil_image(img):
            return img.size
        elif isinstance(img, torch.Tensor) and img.dim() > 2:
            return img.shape[-2:][::-1]
        else:
            raise TypeError("Unexpected type {}".format(type(img)))

    @staticmethod
    def get_params(img, scale, ratio=1):
        """Get parameters for ``crop``.
        Args:
            img (PIL Image): Image to be cropped.
            scale (float): range of size of the origin size cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop``.
        """
        scale = math.sqrt(scale)
        width, height = ResizedCrop._get_image_size(img)

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < 1):
            w = width * scale
            h = int(round(w / ratio))
        elif (in_ratio > 1):
            h = height * scale
            w = int(round(h * ratio))
        else:  # whole image
            w = width * scale
            h = height * scale
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(round(self.scale, 4))
        format_string += ', interpolation={0})'.format(self.interpolation)
        return format_string


class Noise:
    def __init__(self, std):
        self.std = std

    def __call__(self, x):
        return x + torch.randn_like(x) * self.std


def patches(img, patch_size=128):
    """
    Cut an image into square patches of equal size. Padding is added if needed.

    Args:
        img (PIL.Image): the image to split
        patch_size (int): the size od the square patch

    Returns:
        A list of split and coordinates like [((y pos, x pos), img patch)]
    """
    h, w = img.height, img.width
    padder = AdaptPad((patch_size, patch_size))
    for row, h_off in enumerate(range(0, h, patch_size)):
        for c, w_off in enumerate(range(0, w, patch_size)):
            p = img.crop(
                (w_off, h_off, w_off + patch_size, h_off + patch_size))
            yield (row, c), padder(p)


def paste_patches(patches):
    """
    Collate a list of patches and their coordinates such as returned by
    `patches` into an image.

    Args:
        patches (list): a list of patches and coordinates such as returned by
            `patches`

    Returns:
        the assembled PIL.Image.
    """
    patch_size = patches[0][1].width
    total_height = patch_size * (1 + max(p[0][0] for p in patches))
    total_width = patch_size * (1 + max(p[0][1] for p in patches))
    new_im = Image.new('RGB', (total_width, total_height))
    for (l, c), p in patches:
        new_im.paste(p, (c * patch_size, l * patch_size))
    return new_im
