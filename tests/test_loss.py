import torch

from torchelie.loss import *

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
    loss = dd(m(torch.randn(1, 1, 10, 10)))

def test_focal():
    y = (torch.randn(10, 1) < 0).float()
    x = torch.randn(10, 1)

    foc = FocalLoss()
    fl = foc(x, y)
    l = torch.nn.functional.binary_cross_entropy_with_logits(x, y)
    assert torch.allclose(fl, l)

    y = torch.randint(4, (10,))
    x = torch.randn(10, 5)

    fl = foc(x, y)
    l = torch.nn.functional.cross_entropy(x, y)
    assert torch.allclose(fl, l)
