from torchelie.optim import *


def test_deepdream():
    """
    Test the mean

    Args:
    """
    a = torch.randn(5, requires_grad=True)
    opt = DeepDreamOptim([a])
    a.mean().backward()
    opt.step()


def test_addsign():
    """
    Add a set of the mean

    Args:
    """
    a = torch.randn(5, requires_grad=True)
    opt = AddSign([a])
    a.mean().backward()
    opt.step()


def test_radamw():
    """
    Perform the mean step.

    Args:
    """
    a = torch.randn(5, requires_grad=True)
    opt = AddSign([a])
    a.mean().backward()
    opt.step()


def test_lookahead():
    """
    Perform a lookup.

    Args:
    """
    a = torch.randn(5, requires_grad=True)
    opt = Lookahead(AddSign([a]))
    a.mean().backward()
    opt.step()
