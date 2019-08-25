import torchvision.transforms as TF
from PIL import Image
import numpy as np

try:
    import cv2
except:
    print("Can't import OpenCV. Some transforms will not work properly")


class ResizeNoCrop:
    def __init__(self, tgt):
        self.tgt = tgt

    def __call__(self, x):
        size = self.tgt
        width = x.width
        height = x.height
        if width > height:
            return x.resize((size, int(size * height / width)), Image.BILINEAR)
        else:
            return x.resize((int(size * width / height), size), Image.BILINEAR)


class AdaptPad:
    def __init__(self, sz, padding='constant'):
        self.sz = sz
        self.padding_mode = padding_mode

    def __call__(self, img):
        w, h = img.width, img.height

        pl = max(0, self.sz[1] - w) // 2
        pr = self.sz[1] - w - pl

        pt = max(0, self.sz[0] - h) // 2
        pd = self.sz[0] - h - pt

        return TF.functional.pad(img, (pl, pt, pr, pd),
                                 padding_mode=padding_mode)


class MultiBranch:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        return tuple(tf(x) for tf in self.transforms)


class Canny:
    def __init__(self, thresh_low=100, thresh_high=200):
        self.thresh_low = thresh_low
        self.thresh_high = thresh_high

    def __call__(self, img):
        img = np.array(img)
        img = cv2.Canny(img, self.thresh_low, self.thresh_high)
        return Image.fromarray(img)


