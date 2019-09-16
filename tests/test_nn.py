import torch

from torchelie.nn import *


def test_adain():
    m = AdaIN2d(16, 8)
    m(torch.randn(5, 16, 8, 8), torch.randn(5, 8))


def test_film():
    m = FiLM2d(16, 8)
    m(torch.randn(5, 16, 8, 8), torch.randn(5, 8))


def test_bn():
    for M in [NoAffineBN2d, NoAffineMABN2d, BatchNorm2d, MovingAverageBN2d]:
        m = M(16)
        m(torch.randn(5, 16, 8, 8))

    for M in [ConditionalBN2d, ConditionalMABN2d]:
        m = M(16, 8)
        m(torch.randn(5, 16, 8, 8), torch.randn(5, 8))


def test_spade():
    for M in [Spade2d, SpadeMA2d]:
        m = M(16, 8, 4)
        m(torch.randn(5, 16, 8, 8), torch.randn(5, 8, 8, 8))


def test_attnnorm():
    m = AttenNorm2d(16, 8)
    m(torch.randn(1, 16, 8, 8))
