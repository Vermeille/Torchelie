from torchelie.optim import *


def test_deepdream():
    a = torch.randn(5, requires_grad=True)
    opt = DeepDreamOptim([a])
    a.mean().backward()
    opt.step()


def test_addsign():
    a = torch.randn(5, requires_grad=True)
    opt = AddSign([a])
    a.mean().backward()
    opt.step()


def test_radamw():
    a = torch.randn(5, requires_grad=True)
    opt = RAdamW([a])
    a.mean().backward()
    opt.step()


def test_adabelief():
    a = torch.randn(5, requires_grad=True)
    opt = AdaBelief([a])
    a.mean().backward()
    opt.step()


def test_lookahead():
    a = torch.randn(5, requires_grad=True)
    b = torch.randn(5, requires_grad=True)
    opt = Lookahead(AdaBelief([a, b]))
    for _ in range(10):
        a.mean().backward()
        opt.step()
