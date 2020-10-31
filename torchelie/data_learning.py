import numpy as np
import torch
import torch.nn as nn
from typing import List


def _rfft2d_freqs(h, w):
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[:w // 2 + (1 if w % 2 == 0 else 2)]
    return np.sqrt(fx * fx + fy * fy)


class PixelImage(nn.Module):
    """
    A learnable image parameterized by its pixel values

    Args:
        shape (tuple of int): a tuple like (channels, height, width)
        sd (float): pixels are initialized with a random gaussian value of mean
            0.5 and standard deviation `sd`
        init_img (tensor, optional): an image tensor to initialize the pixel
            values
    """
    def __init__(self,
                 shape: List[int],
                 sd: float = 0.01,
                 init_img: torch.Tensor = None) -> None:
        super(PixelImage, self).__init__()
        self.shape = shape
        n, ch, h, w = shape
        self.pixels = torch.nn.Parameter(sd * torch.randn(n, ch, h, w),
                requires_grad=True)

        if init_img is not None:
            self.init_img(init_img)

    def init_img(self, init_img):
        self.pixels.data.copy_(init_img - 0.5)

    def forward(self):
        """
        Return the image
        """
        return self.pixels


class SpectralImage(nn.Module):
    """
    A learnable image parameterized by its Fourier representation.

    See https://distill.pub/2018/differentiable-parameterizations/

    Implementation ported from
    https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/spatial.py

    Args:
        shape (tuple of int): a tuple like (channels, height, width)
        sd (float): amplitudes are initialized with a random gaussian value of
            mean 0.5 and standard deviation `sd`
        init_img (tensor, optional): an image tensor to initialize the image
    """
    def __init__(self, shape, sd=0.01, decay_power=1, init_img=None):
        super(SpectralImage, self).__init__()
        self.shape = shape
        n, ch, h, w = shape
        freqs = _rfft2d_freqs(h, w)
        fh, fw = freqs.shape
        self.decay_power = decay_power

        init_val = sd * torch.randn(n, ch, fh, fw, 2)
        spectrum_var = torch.nn.Parameter(init_val)
        self.spectrum_var = spectrum_var

        spertum_scale = 1.0 / np.maximum(freqs,
                                         1.0 / max(h, w))**self.decay_power
        spertum_scale *= np.sqrt(w * h)
        spertum_scale = torch.tensor(spertum_scale).unsqueeze(-1)
        self.register_buffer('spertum_scale', spertum_scale)

        if init_img is not None:
            self.init_img(init_img)

    def init_img(self, init_img: torch.Tensor):
        if init_img.shape[2] % 2 == 1:
            init_img = nn.functional.pad(init_img, [1, 0, 0, 0])
        fft = torch.rfft(init_img * 4, 2, onesided=True, normalized=False)
        self.spectrum_var.data.copy_(fft / self.spertum_scale)

    def forward(self) -> torch.Tensor:
        """
        Return the image
        """
        n, ch, h, w = self.shape

        scaled_spectrum = self.spectrum_var * self.spertum_scale
        img = torch.irfft(scaled_spectrum, 2, onesided=True, normalized=False)

        img = img[:, :ch, :h, :w]
        return img / 4.


class CorrelateColors(torch.nn.Module):
    """
    Takes an learnable image and applies the inverse color decorrelation from
    ImageNet (ie, it correlates the color like ImageNet to ease optimization)
    """
    def __init__(self):
        super(CorrelateColors, self).__init__()
        color_correlation_svd_sqrt = torch.tensor([[0.26, 0.09, 0.02],
                                                   [0.27, 0.00, -0.05],
                                                   [0.27, -0.09, 0.03]])

        max_norm_svd_sqrt = float(
            np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0)))

        cc = color_correlation_svd_sqrt / max_norm_svd_sqrt
        self.register_buffer('color_correlation', cc)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Correlate the color of the image `t` and return the result
        """
        t_flat = t.view(t.shape[0], 3, -1).transpose(2, 1)
        t_flat = torch.matmul(t_flat, self.color_correlation.t())
        t = t_flat.transpose(2, 1).view(t.shape)
        return t

    def invert(self, t: torch.Tensor) -> torch.Tensor:
        """
        Decorrelate the color of the image `t` and return the result
        """
        t_flat = t.view(t.shape[0], 3, -1).transpose(2, 1)
        t_flat = torch.matmul(t_flat,
                              self.color_correlation.inverse().t()[None])
        t = t_flat.transpose(2, 1).view(t.shape)
        return t


class RGB:
    def __call__(self, x):
        return x

    def invert(self, x):
        return x


class ParameterizedImg(nn.Module):
    """
    A convenient wrapper around `PixelImage` and `SpectralImage` and
    `CorrelateColors` to make a learnable image.

    Args:
        *shape (int): shape of the image: channel, height, width
        init_sd (float): standard deviation for initializing the image if
            `init_img` is `None`
        init_img (tensor): an image to use as initialization
        space (str): either `"pixel"` or `"spectral"` to have the underlying
            representation be a `PixelImage` or a `SpectralImage`
        colors (str): either `"corr"` or `"uncorr"`, to use a correlated or
            decorrelated color space
    """
    def __init__(self,
                 *shape,
                 init_sd=0.06,
                 init_img=None,
                 space='spectral',
                 colors='uncorr'):
        super(ParameterizedImg, self).__init__()

        assert colors in ['uncorr', 'corr']
        if colors == 'uncorr':
            self.color = CorrelateColors()
        else:
            self.color = RGB()

        assert space in ['spectral', 'pixel']
        if space == 'spectral':
            self.img = SpectralImage(shape, sd=init_sd)
        else:
            self.img = PixelImage(shape, sd=init_sd)

        if init_img is not None:
            self.init_img(init_img)

    def init_img(self, init_img):
        init_img = init_img.clamp(0.01, 0.99)
        init_img = -torch.log(((1 - init_img) / init_img))

        init_img = self.color.invert(init_img)
        self.img.init_img(init_img)

    def forward(self):
        """
        Return the tensor
        """
        t = self.color(self.img())
        return torch.sigmoid(t) + 0.01 * t

    def render(self):
        """
        Return the tensor on cpu and detached, ready to be transformed to a PIL
        image
        """
        return self.forward().cpu().detach()[0]
