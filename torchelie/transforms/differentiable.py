import math
import random
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, cast, List

# FIXME: The API is not that great, improve it.


def _rollx(img, begin):
    return torch.cat([img[..., :, begin:], img[..., :, :begin]], dim=-1)


def _rolly(img, begin):
    return torch.cat([img[..., begin:, :], img[..., :begin, :]], dim=-2)


def roll(img, x_roll, y_roll):
    """
    Wrap an image

    Args:
        img (3D or 4D image(s) tensor): an image tensor
        x_roll (int): how many pixels to roll on the x axis
        y_roll (int): how many pixels to roll on the y axis

    Returns:
        The rolled tensor
    """
    return _rollx(_rolly(img, y_roll), x_roll)


def center_crop(batch, size):
    """
    Crop the center of a 4D images tensor

    Args:
        batch (4D images tensor): the tensor to crop
        size ((int, int)): size of the resulting image as (height, width)

    Returns:
        The cropped image
    """
    if batch.shape[-2] == size[0] and batch.shape[-1] == size[1]:
        return batch
    y_off = (batch.shape[-2] - size[0]) // 2
    x_off = (batch.shape[-1] - size[1]) // 2
    return batch[..., y_off:y_off + size[0], x_off:x_off + size[1]]


def crop(img, warped=True, sub_img_factor=2):
    """
    Randomly crop a `sub_img_factor` smaller part of `img`.

    Args:
        img (3D or 4D image(s) tensor): input image(s)
        warped (bool): Whether the image should be considered warped (default:
            True)
        sub_img_factor (float): fraction of the image to take. For instance, 2
            will crop a quarter of the image (half the width, half the height).
            (default: 2)
    """
    sz = img.shape
    h = np.random.randint(sz[-2] // sub_img_factor, sz[-2])
    w = np.random.randint(sz[-1] // sub_img_factor, sz[-1])

    if warped:
        y = np.random.randint(sz[-2])
        x = np.random.randint(sz[-1])
        out = _rolly(_rollx(img, x), y)
        return out[..., :h, :w]
    else:
        y = np.random.randint(sz[-2] - h)
        x = np.random.randint(sz[-1] - w)
        return img[:, y:y + h, x:x + w]


def _gblur_kernel_2d(c):
    gaussian = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    kernel = np.zeros((3, 3, 3))
    kernel[c, :, :] = gaussian
    return kernel


def _gblur_kernel():
    return np.stack(
        [_gblur_kernel_2d(0),
         _gblur_kernel_2d(1),
         _gblur_kernel_2d(2)]).astype(np.float32)


def gblur(input):
    """
    Gaussian blur with kernel size 3

    Args:
        input (3D or 4D image(s) tensor): input image

    Returns:
        the blurred tensor
    """
    print('gblur', input.dtype, torch.from_numpy(_gblur_kernel()).dtype)
    return F.conv2d(input,
                    torch.from_numpy(_gblur_kernel()).to(input.device),
                    padding=1)


def _mblur_kernel_2d(c):
    gaussian = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 8
    kernel = np.zeros((3, 3, 3))
    kernel[c, :, :] = gaussian
    return kernel


def _mblur_kernel():
    return np.stack(
        [_mblur_kernel_2d(0),
         _mblur_kernel_2d(1),
         _mblur_kernel_2d(2)]).astype(np.float32)


class BinomialFilter2d(torch.nn.Module):
    def __init__(self, stride: int):
        super().__init__()
        self.stride = stride
        self.register_buffer(
            'weight',
            torch.tensor([[[1.0, 2, 1], [2, 4, 2], [1, 2, 1]]]) / 16)

    def forward(self, x):
        x = torch.nn.functional.pad(x, (1, 1, 1, 1), mode='replicate')
        return torch.nn.functional.conv2d(x,
                                          self.weight.expand(x.shape[1], 1, -1, -1),
                                          groups=x.shape[1],
                                          stride=self.stride,
                                          padding=0)


def mblur(input):
    """
    Mean (or average) blur with kernel size 3

    Args:
        input (3D or 4D image(s) tensor): input image

    Returns:
        the blurred tensor
    """
    return F.conv2d(input,
                    torch.from_numpy(_mblur_kernel()).to(input.device),
                    padding=1)


class AllAtOnceGeometric:
    """
    Various geometric transforms packed up into an affine transformation
    matrix. Transformations can be stacked then applied on a 4D tensor to
    reduce artifact, memory usage, and compute. Fully differentiable.

    >>> img = torch.randn(10, 3, 32, 32)
    >>> transformed = AllAtOnceGeometric(10)\
            .translate(5, 5).scale(0.9).apply(img)

    Note: the transformations get sampled at creation, so that each call to
    apply() runs the same transforms. Construct another AllAtOnceGeometric
    object for another set of transform. This allows to easily run the same
    transforms on paired datasets.

    Note2: Each transform has a prob argument which specifies whether to use
    the transform or bypass it. This makes it easy to implement StyleGAN2-ADA.

    Args:
        B (int): batch size
        init (torch.Tensor): an initial user supplied transformation matrix. If
            not provided, default to identity.
    """
    B: int
    m: torch.Tensor

    def __init__(self, B: int, init: Optional[torch.Tensor] = None) -> None:
        if init is None:
            self.m = torch.stack([torch.eye(3, 3) for _ in range(B)], dim=0)
        else:
            self.m = init
        self.B = B

    def _mix(self, m: torch.Tensor, prob: float) -> None:
        RND = (m.shape[0], 1, 1)
        self.m = torch.where(
            torch.rand(RND) < prob, torch.bmm(m, self.m), self.m)

    def translate(self, x: float, y: float, prob: float = 1.) -> 'AllAtOnceGeometric':
        """
        Randomly translate image horizontally with an offset sampled in [-x, x]
        and vertically [-y, y]. Note that the coordinate are not pixel
        coordinate but texel coordinate between [-1, 1]
        """
        tfm = torch.tensor([[[1, 0., random.uniform(-x, x)],
                             [0, 1, random.uniform(-y, y)], [0, 0, 1]]
                            for _ in range(self.B)])
        self._mix(tfm, prob)
        return self

    def scale(self, x: float, y: float, prob: float = 1.) -> 'AllAtOnceGeometric':
        """
        Randomly scale the image horizontally by a factor [1 - x; 1 + x] and
        vertically by a factor of [1 - y; 1 + y].

        Args:
            x (float): horizontal factor
            y (float): vertical factor
        """
        tfm = torch.tensor([[[random.uniform(1 - x, 1 + x), 0., 0],
                             [0, random.uniform(1 - y, 1 + y), 0], [0, 0, 1]]
                            for _ in range(self.B)])
        self._mix(tfm, prob)
        return self

    def rotate(self, theta: float, prob: float = 1.) -> 'AllAtOnceGeometric':
        """
        Rotate the image by an angle randomly sampled between [-theta, theta]

        Args:
            theta (float): an angle in degrees
        """
        theta = theta * 3.14 / 180
        rot = []
        for _ in range(self.B):
            t = random.uniform(-theta, theta)
            rot.append([[math.cos(t), math.sin(-t), 0],
                        [math.sin(t), math.cos(t), 0], [0, 0, 1]])
        tfm = torch.tensor(rot)
        self._mix(tfm, prob)
        return self

    def flip_x(self, p: float, prob: float = 1.) -> 'AllAtOnceGeometric':
        tfm = []
        for _ in range(self.B):
            p_rot = math.copysign(1, random.random() - (1 - p))
            tfm.append([[p_rot, 0., 0], [0, 1, 0], [0, 0, 1]])
        tfm_matrix = torch.tensor(tfm)
        self._mix(tfm_matrix, prob)
        return self

    def flip_y(self, p: float, prob: float = 1.) -> 'AllAtOnceGeometric':
        tfm = []
        for _ in range(self.B):
            p_rot = math.copysign(1, random.random() - (1 - p))
            tfm.append([[1, 0., 0], [0, p_rot, 0], [0, 0, 1]])
        tfm_matrix = torch.tensor(tfm)
        self._mix(tfm_matrix, prob)
        return self

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        grid = F.affine_grid(self.m[:, :2, :], cast(List[int], x.shape))
        grid = grid.to(x.device)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection')


class AllAtOnceColor:
    """
    Similar to AllAtOnceGeometric, performs multiple color transforms at once.

    Args:
        B (int): batch size
        init (torch.Tensor): an initial user supplied transformation matrix. If
            not provided, default to identity.
    """
    B: int
    m: torch.Tensor

    def __init__(self, B: int, init: Optional[torch.Tensor] = None):
        self.B = B
        if init is None:
            self.m = torch.stack([torch.eye(4, 4) for _ in range(B)], dim=0)
        else:
            self.m = init

    def _mix(self, m: torch.Tensor, prob: float) -> None:
        RND = (self.B, 1, 1)
        self.m = torch.where(
            torch.rand(RND) < prob, torch.bmm(m, self.m), self.m)

    def brightness(self, alpha: float, prob: float = 1.) -> 'AllAtOnceColor':
        """
        Change brightness by a factor alpha

        Args:
            alpha (float): scale factor
        """
        tfm = torch.stack([
            torch.eye(4, 4) * random.uniform(1 - alpha, 1 + alpha)
            for _ in range(self.B)
        ], dim=0)
        tfm[:, 3, 3] = 1
        self._mix(tfm, prob)
        return self

    def contrast(self, alpha: float, prob: float = 1.) -> 'AllAtOnceColor':
        """
        Scale contrast by factor alpha

        Args:
            alpha (float): scale factor
        """
        rot = []
        for _ in range(self.B):
            t = random.uniform(1 - alpha, 1 + alpha)
            rot.append([[t, 0, 0, (1 - t) / 2], [0, t, 0, (1 - t) / 2],
                        [0, 0, t, (1 - t) / 2], [0, 0, 0, 1]])
        rot_matrix = torch.tensor(rot)
        self._mix(rot_matrix, prob)
        return self

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies transforms on x

        Args:
            x (torch.Tensor): input

        Returns:
            transformed x
        """
        B, C, H, W = x.shape
        w = self.m[:, :3, :3].reshape(3 * self.B, 3, 1, 1)
        b = self.m[:, :3, 3].reshape(self.B, 3, 1, 1)
        x = x.view(1, B * C, H, W)
        w = w.to(x.device)
        b = b.to(x.device)
        out = F.conv2d(x, w, None, groups=self.B)
        out = out.view(B, C, H, W)
        return out + b
