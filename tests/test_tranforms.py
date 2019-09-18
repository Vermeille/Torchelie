import torch
from torchvision.transforms import ToPILImage
from torchelie.transforms import *


def test_resizenocrop():
    img = ToPILImage()(torch.clamp(torch.randn(3, 32, 16) + 1, min=0, max=1))
    tf = ResizeNoCrop(16)
    assert tf(img).width == 8
    assert tf(img).height == 16


def test_adaptpad():
    img = ToPILImage()(torch.clamp(torch.randn(3, 30, 16) + 1, min=0, max=1))
    tf = AdaptPad((32, 32))
    assert tf(img).width == 32
    assert tf(img).height == 32


def test_multibranch():
    tf = MultiBranch([
        lambda x: x + 1,
        lambda x: x * 3,
        lambda x: x - 1,
    ])
    assert tf(1) == (2, 3, 0)


def test_canny():
    img = ToPILImage()(torch.clamp(torch.randn(3, 30, 16) + 1, min=0, max=1))
    tf = Canny()
    tf(img)
