import torch
import torch.nn.functional as F
import numpy as np

# FIXME: The API is not that great, improve it.


def _rollx(img, begin):
    return torch.cat(
            [img[..., :, begin:], img[..., :, :begin]], dim=-1)


def _rolly(img, begin):
    return torch.cat(
            [img[..., begin:, :], img[..., :begin, :]], dim=-2)


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
        return out [..., :h, :w]
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
         _gblur_kernel_2d(2)]).astype(float)


def gblur(input):
    """
    Gaussian blur with kernel size 3

    Args:
        input (3D or 4D image(s) tensor): input image

    Returns:
        the blurred tensor
    """
    return F.conv2d(input,
                    torch.FloatTensor(_gblur_kernel()).to(input.device),
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
         _mblur_kernel_2d(2)]).astype(float)


def mblur(input):
    """
    Mean (or average) blur with kernel size 3

    Args:
        input (3D or 4D image(s) tensor): input image

    Returns:
        the blurred tensor
    """
    return F.conv2d(input,
                    torch.FloatTensor(_mblur_kernel()).to(input.device),
                    padding=1)
