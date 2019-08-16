import numpy as np
import torch
import torch.nn as nn


def _rfft2d_freqs(h, w):
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[:w // 2 + (1 if w % 2 == 0 else 2)]
    return np.sqrt(fx * fx + fy * fy)


class PixelImage(nn.Module):
    def __init__(self, shape, sd=0.01, init_img=None):
        super(PixelImage, self).__init__()
        ch, h, w = shape
        self.pixels = torch.nn.Parameter(sd * torch.randn(1, ch, h, w))

        if init_img is not None:
            self.pixels.data.copy_(init_img - 0.5)

    def forward(self):
        return (self.pixels)# + 0.5).clamp(0, 1)


class SpectralImage(nn.Module):
    def __init__(self, shape, sd=0.01, decay_power=1, init_img=None):
        super(SpectralImage, self).__init__()
        self.shape = shape
        ch, h, w = shape
        freqs = _rfft2d_freqs(h, w)
        fh, fw = freqs.shape
        self.decay_power = decay_power

        init_val = sd * torch.randn(ch, fh, fw, 2)
        spectrum_var = torch.nn.Parameter(init_val)
        self.spectrum_var = spectrum_var

        spertum_scale = 1.0 / np.maximum(freqs,
                                         1.0 / max(h, w))**self.decay_power
        spertum_scale *= np.sqrt(w * h)
        spertum_scale = torch.FloatTensor(spertum_scale).unsqueeze(-1)
        self.register_buffer('spertum_scale', spertum_scale)

        if init_img is not None:
            fft = torch.rfft(init_img * 4, 2, onesided=True, normalized=False)
            self.spectrum_var.data.copy_(fft / spertum_scale)

    def forward(self):
        ch, h, w = self.shape

        scaled_spectrum = self.spectrum_var * self.spertum_scale
        img = torch.irfft(scaled_spectrum, 2, onesided=True, normalized=False)

        img = img[:ch, :h, :w]
        return img.unsqueeze(0) / 4.


class CorrelateColors(torch.nn.Module):
    def __init__(self):
        super(CorrelateColors, self).__init__()
        color_correlation_svd_sqrt = torch.FloatTensor([[0.26, 0.09, 0.02],
                                                        [0.27, 0.00, -0.05],
                                                        [0.27, -0.09, 0.03]])

        max_norm_svd_sqrt = float(
            np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0)))

        self.register_buffer('color_correlation',
                             color_correlation_svd_sqrt / max_norm_svd_sqrt)

    def forward(self, t):
        t_flat = t.view(t.shape[0], 3, -1).transpose(2, 1)
        t_flat = torch.matmul(t_flat, self.color_correlation.t())
        t = t_flat.transpose(2, 1).view(t.shape)
        return t

    def invert(self, t):
        t_flat = t.view(3, -1).transpose(1, 0)
        t_flat = torch.matmul(t_flat, self.color_correlation.inverse().t())
        t = t_flat.transpose(1, 0).view(t.shape)
        return t


class ParameterizedImg(nn.Module):
    def __init__(self, *shape, init_sd=0.06, init_img=None, space='spectral',
            colors='uncorr'):
        super(ParameterizedImg, self).__init__()

        if init_img is not None:
            init_img = init_img.clamp(0.0001, 0.999)
            init_img = -torch.log(((1 - init_img) / init_img))

        assert colors in ['uncorr', 'corr']
        self.corr = lambda x: x
        if colors == 'uncorr':
            self.corr = CorrelateColors()
            if init_img is not None:
                init_img = self.corr.invert(init_img)

        assert space in ['spectral', 'pixel']
        if space == 'spectral':
            self.img = SpectralImage(shape, sd=init_sd, init_img=init_img)
        else:
            self.img = PixelImage(shape, sd=init_sd, init_img=init_img)


    def forward(self):
        t = self.corr(self.img())
        return torch.sigmoid(t)

    def render(self):
        return self.forward().cpu().detach()[0]
