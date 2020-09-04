import torch

from torchelie.utils import *


def test_net():
    m = torch.nn.Linear(10, 4)
    freeze(m)
    unfreeze(m)
    kaiming(m)
    xavier(m)
    normal_init(m, 0.02)
    nb_parameters(m)
    assert m is layer_by_name(torch.nn.Sequential(m), '0')
    assert layer_by_name(torch.nn.Sequential(m), 'test') is None
    send_to_device([{'a': [m]}], 'cpu')

    fm = FrozenModule(m)
    fm.train()
    assert not fm.weight.requires_grad
    fm.weight

    fm = DetachedModule(m)
    fm.weight


def test_utils():
    entropy(torch.randn(1, 10))
    gram(torch.randn(4, 10))
    bgram(torch.randn(3, 4, 10))
    assert dict_by_key({'a': [{'b': 42}]}, 'a.0.b') == 42
    assert lerp(0, 2, 0.5) == 1
    assert ilerp(0, 2, 1) == 0.5
