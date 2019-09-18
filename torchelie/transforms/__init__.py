import torchvision.transforms as TF
from PIL import Image
import numpy as np

try:
    import cv2
except:
    print("Can't import OpenCV. Some transforms will not work properly")


class ResizeNoCrop:
    """
    Resize a PIL image so that its longer border is of size `size`

    Args:
        size (int): max size of the image
    """
    def __init__(self, size):
        self.size = size

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
            return x.resize((size, int(size * height / width)), Image.BILINEAR)
        else:
            return x.resize((int(size * width / height), size), Image.BILINEAR)


class AdaptPad:
    """
    Pad an input image so that it reaches size `size`

    Args:
        sz ((int, int)): target size
        padding_mode (str): one of the modes of `torchvision.transforms.pad`
    """
    def __init__(self, sz, padding_mode='constant'):
        self.sz = sz
        self.padding_mode = padding_mode

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
                                 padding_mode=self.padding_mode)


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
        img = np.array(img)
        img = cv2.Canny(img, self.thresh_low, self.thresh_high)
        return Image.fromarray(img)


