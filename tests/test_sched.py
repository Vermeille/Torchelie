import torch
from torch.optim import SGD
from torchelie.lr_scheduler import *


def test_curriculum():
    x = torch.randn(3, requires_grad=True)
    opt = SGD([x], 1)
    sched = CurriculumScheduler(opt, [[0, 1, 1], [10, 0, 0]])
    opt.step()
    sched.step()


def test_onecycle():
    x = torch.randn(3, requires_grad=True)
    opt = SGD([x], 1)
    sched = OneCycle(opt, (1e-3, 1e-2), 10)
    opt.step()
    sched.step()
