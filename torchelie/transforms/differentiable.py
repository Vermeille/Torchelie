import torch
import torch.nn.functional as F
import numpy as np

# FIXME: The API is not that great, improve it.


def _rollx(img, begin, to):
    if to <= img.shape[1]:
        return img[:, :, begin:to]
    else:
        return torch.cat([img[:, :, begin:], img[:, :, :to - img.shape[2]]],
                         dim=2)


def _rolly(img, begin, to):
    if to <= img.shape[0]:
        return img[:, begin:to, :]
    else:
        return torch.cat([img[:, begin:, :], img[:, :to - img.shape[1], :]],
                         dim=1)


def crop(batch_img, warped=True, sub_img_factor=2):
    imgs = []
    for img in batch_img:
        sz = img.shape
        h = np.random.randint(sz[1] // sub_img_factor, sz[1])
        w = np.random.randint(sz[2] // sub_img_factor, sz[2])

        if warped:
            y = np.random.randint(sz[1])
            x = np.random.randint(sz[2])
            img_crop = _rolly(_rollx(img, x, x + w), y, y + h)
        else:
            y = np.random.randint(sz[1] - h)
            x = np.random.randint(sz[2] - w)
            img_crop = img[:, y:y + h, x:x + w]
        imgs.append(img_crop)
    return torch.stack(imgs)


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
    return F.conv2d(input,
                    torch.FloatTensor(_gblur_kernel()).cuda(),
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
    return F.conv2d(input,
                    torch.FloatTensor(_mblur_kernel()).cuda(),
                    padding=1)
