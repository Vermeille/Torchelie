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

