import torch
from torchvision.transforms import ToPILImage
from torchelie.transforms import *
import torchelie.transforms.differentiable as dtf


def test_resizenocrop():
    """
    Resizenocropocropropropropicropocropicropicropropropicropropicropicropropicropic

    Args:
    """
    img = ToPILImage()(torch.clamp(torch.randn(3, 32, 16) + 1, min=0, max=1))
    tf = ResizeNoCrop(16)
    assert tf(img).width == 8
    assert tf(img).height == 16


def test_adaptpad():
    """
    Pushes a pil image size.

    Args:
    """
    img = ToPILImage()(torch.clamp(torch.randn(3, 30, 16) + 1, min=0, max=1))
    tf = AdaptPad((32, 32))
    assert tf(img).width == 32
    assert tf(img).height == 32


def test_multibranch():
    """
    Test if a multibid.

    Args:
    """
    tf = MultiBranch([
        lambda x: x + 1,
        lambda x: x * 3,
        lambda x: x - 1,
    ])
    assert tf(1) == (2, 3, 0)


def test_canny():
    """
    Convert the image to canny.

    Args:
    """
    img = ToPILImage()(torch.clamp(torch.randn(3, 30, 16) + 1, min=0, max=1))
    tf = Canny()
    tf(img)


def test_resizedcrop():
    """
    Resized crop is a crop.

    Args:
    """
    img = ToPILImage()(torch.clamp(torch.randn(3, 30, 16) + 1, min=0, max=1))
    tf = ResizedCrop(48)
    tf(img)


def test_diff():
    """
    Compute the difference between two images.

    Args:
    """
    dtf.roll(torch.randn(3, 16, 16), 3, 3)
    dtf.roll(torch.randn(1, 3, 16, 16), 3, 3)
    dtf.center_crop(torch.randn(3, 16, 16), (4, 4))
    dtf.crop(torch.randn(1, 3, 16, 16))
    dtf.gblur(torch.randn(1, 3, 16, 16))
    dtf.mblur(torch.randn(1, 3, 16, 16))
