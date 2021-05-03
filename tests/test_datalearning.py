import torch
from torchelie.data_learning import *


def test_pixel_image():
    pi = PixelImage((1, 3, 128, 128), 0.01)
    pi()

    start = torch.randn(3, 128, 128)
    pi = PixelImage((1, 3, 128, 128), init_img=start)

    assert start.allclose(pi() + 0.5, atol=1e-7)


def test_spectral_image():
    pi = SpectralImage((1, 3, 128, 128), 0.01)
    pi()

    start = torch.randn(1, 3, 128, 128)
    pi = SpectralImage((1, 3, 128, 128), init_img=start)


def test_correlate_colors():
    corr = CorrelateColors()
    start = torch.randn(1, 3, 64, 64)
    assert start.allclose(corr.invert(corr(start)), atol=1e-5)


def test_parameterized_img():
    start = torch.clamp(torch.randn(1, 3, 128, 128) + 0.5, min=0, max=1)

    ParameterizedImg(1, 3, 128, 128, space='spectral', colors='uncorr')()
    ParameterizedImg(1, 3,
                     128,
                     128,
                     space='spectral',
                     colors='uncorr',
                     init_img=start)()

    ParameterizedImg(1, 3, 128, 128, space='spectral', colors='uncorr')()

    start = torch.clamp(torch.randn(1, 3, 128, 129) + 0.5, min=0, max=1)
    ParameterizedImg(1, 3,
                     128,
                     129,
                     space='spectral',
                     colors='uncorr',
                     init_img=start)()
    start = torch.clamp(torch.randn(1, 3, 128, 128) + 0.5, min=0, max=1)
    ParameterizedImg(1, 3, 128, 128, space='pixel', colors='uncorr')()
    ParameterizedImg(1, 3,
                     128,
                     128,
                     space='pixel',
                     colors='uncorr',
                     init_img=start)()

    ParameterizedImg(1, 3, 128, 128, space='spectral', colors='corr')()
    ParameterizedImg(1, 3,
                     128,
                     128,
                     space='spectral',
                     colors='corr',
                     init_img=start)()

    ParameterizedImg(1, 3, 128, 128, space='pixel', colors='corr')()
    ParameterizedImg(1, 3, 128, 128, space='pixel', colors='corr',
                     init_img=start)()
