import torchvision.transforms as TF
from PIL import Image


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

