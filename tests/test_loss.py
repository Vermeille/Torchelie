import torch

from torchelie.loss import *
import torchelie.loss.gan as gan
import torchelie.loss.functional as tlf


def test_bitempered():
    x = torch.randn(3, 5)
    y = torch.arange(3)

    tsm = tempered_softmax(x, 1)
    sm = torch.nn.functional.softmax(x, dim=1)
    assert sm.allclose(tsm)

    tsm = tempered_log_softmax(x, 1)
    sm = torch.nn.functional.log_softmax(x, dim=1)
    assert sm.allclose(tsm)

    tnll = tempered_nll_loss(sm, y, 1, 1)
    nll = torch.nn.functional.nll_loss(sm, y)
    assert nll.allclose(tnll)

    temp_loss = tempered_cross_entropy(x, y, 1, 1)
    ce_loss = torch.nn.functional.cross_entropy(x, y)
    assert ce_loss.allclose(temp_loss)

    tce = TemperedCrossEntropyLoss(1, 1)
    assert ce_loss.allclose(tce(x, y))


def test_deepdream():
    m = nn.Sequential(nn.Conv2d(1, 1, 3))
    dd = DeepDreamLoss(m, '0', max_reduction=1)
    dd(m(torch.randn(1, 1, 10, 10)))


def test_focal():
    y = (torch.randn(10, 1) < 0).float()
    x = torch.randn(10, 1)

    foc = FocalLoss()
    fl = foc(x, y)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(x, y)
    assert torch.allclose(fl, loss)

    y = torch.randint(4, (10,))
    x = torch.randn(10, 5)

    fl = foc(x, y)
    loss = torch.nn.functional.cross_entropy(x, y)
    assert torch.allclose(fl, loss)

    focal_loss(torch.randn(10, 5), torch.randint(4, (10,)))


def test_funcs():
    f = OrthoLoss()
    f(torch.randn(10, 10))
    ortho(torch.randn(10, 10))

    f = TotalVariationLoss()
    f(torch.randn(1, 1, 10, 10))
    total_variation(torch.randn(1, 1, 10, 10))

    f = ContinuousCEWithLogits()
    continuous_cross_entropy(torch.randn(10, 5),
                             torch.nn.functional.softmax(torch.randn(10, 5), 1))
    f(torch.randn(10, 5), torch.nn.functional.softmax(torch.randn(10, 5), 1))


def test_neural_style():
    ns = NeuralStyleLoss()
    ns.set_content(torch.randn(3, 128, 128))
    ns.set_style(torch.randn(3, 128, 128), 1)
    ns(torch.randn(1, 3, 128, 128))


def test_perceptual():
    pl = PerceptualLoss(['conv1_1'], rescale=True)
    pl(torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))


def test_gan():
    x = torch.randn(5, 5)

    gan.hinge.real(x)
    gan.hinge.fake(x)
    gan.hinge.generated(x)

    gan.standard.real(x)
    gan.standard.fake(x)
    gan.standard.generated(x)
